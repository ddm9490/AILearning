import uuid

from flask import Flask, g, jsonify, request

import arxiv_service
import context_providers
import curriculum_service
import curriculum_store
import llm_service
import pdf_service
from keyword_catalog import TIERS, extract_keywords, resolve_keyword

app = Flask(__name__, static_folder="frontend", static_url_path="")
curriculum_store.init_db()

MIN_COUNT = 3
MAX_COUNT = 15
DEFAULT_COUNT = 4
CANDIDATE_POOL_SIZE = 20
KNOWLEDGE_BASE_HITS_LIMIT = 30
TOTAL_SEARCH_KEYWORDS = 5

# 로그인이 없는 앱이라, 브라우저별로 발급하는 익명 랜덤 쿠키 하나로 "누구 커리큘럼인지"를
# 구분한다. 서버를 하나 켜두고 여러 명이 같이 쓰는 배포 환경(해커톤 등)에서, 이게 없으면
# curriculum_store가 완전히 공유 저장소라 다른 사람이 만든 커리큘럼/진행 상황이 전부
# 보이고 조작까지 가능했다. Flask session(서명 쿠키)을 안 쓴 이유: 그러려면 SECRET_KEY를
# 안정적으로 관리해야 하는데(재시작마다 바뀌면 로그인 풀림), 여긴 민감정보가 아니라 그냥
# "이 브라우저"를 구분하는 용도라 서명 없는 랜덤 쿠키로 충분하다.
OWNER_COOKIE_NAME = "bypp_uid"
OWNER_COOKIE_MAX_AGE = 60 * 60 * 24 * 180  # 180일


def _get_owner_id():
    owner_id = request.cookies.get(OWNER_COOKIE_NAME)
    if not owner_id:
        owner_id = uuid.uuid4().hex
        g._new_owner_id = owner_id  # after_request에서 이 값이 있으면 쿠키로 내려보낸다
    return owner_id


@app.after_request
def _set_owner_cookie(response):
    new_owner_id = getattr(g, "_new_owner_id", None)
    if new_owner_id:
        response.set_cookie(
            OWNER_COOKIE_NAME, new_owner_id, max_age=OWNER_COOKIE_MAX_AGE, httponly=True, samesite="Lax"
        )
    return response


def _clamp_recommend_count(raw_count):
    try:
        count = int(raw_count)
    except (TypeError, ValueError):
        return DEFAULT_COUNT
    return max(MIN_COUNT, min(MAX_COUNT, count))


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/tiers")
def tiers():
    return jsonify(TIERS)


@app.post("/api/recommend")
def recommend():
    body = request.get_json(silent=True) or {}
    interest = str(body.get("interest", "")).strip()
    selected_keywords = [str(k).strip() for k in body.get("keywords", []) if str(k).strip()]
    count = _clamp_recommend_count(body.get("count"))

    if not interest and not selected_keywords:
        return jsonify({"error": "관심 분야를 입력하거나 키워드를 선택해주세요."}), 400

    # 사용자가 직접 고르거나 입력한 키워드(예: "ConvNeXt")는 LLM을 거치지 않고 그대로
    # 검색어에 쓴다 — LLM에게 "다듬어 달라"고 맡기면 특정 아키텍처/모델 이름 같은
    # 고유명사를 다른 키워드로 잘못 해석해버리는 문제를 실제로 겪어서(사용자 리포트),
    # 자연어 관심사(interest)만 LLM으로 보내고 정확한 키워드는 그대로 합친다.
    exact_keywords = list(dict.fromkeys(selected_keywords))  # 순서 유지 + 중복 제거
    remaining_slots = max(0, TOTAL_SEARCH_KEYWORDS - len(exact_keywords))

    refined_keywords = []
    if interest and remaining_slots > 0:
        try:
            refined_keywords = llm_service.refine_search_keywords(interest, remaining_slots)
        except llm_service.QuotaExhaustedError as exc:
            return jsonify({"error": str(exc)}), 429
        except Exception:
            return jsonify({"error": "AI가 검색 키워드를 만드는 데 실패했어요. 잠시 후 다시 시도해주세요."}), 502

    exact_lower = {k.lower() for k in exact_keywords}
    search_keywords = exact_keywords + [k for k in refined_keywords if k.lower() not in exact_lower]

    try:
        candidate_results = arxiv_service.search_candidates(search_keywords, CANDIDATE_POOL_SIZE)
    except Exception:
        return jsonify({"error": "arXiv 검색에 실패했어요. 잠시 후 다시 시도해주세요."}), 502

    if not candidate_results:
        return jsonify({"search_keywords": search_keywords, "papers": []})

    candidates_by_id = {}
    candidate_summaries = []
    candidate_text_parts = []
    for result in candidate_results:
        paper = arxiv_service.to_dict(result)
        candidates_by_id[paper["id"]] = paper
        candidate_summaries.append({"id": paper["id"], "title": paper["title"], "summary": paper["summary"]})
        candidate_text_parts.append(f"{paper['title']} {paper['summary']}")

    # RAG: 후보 논문들 텍스트에서 우리 지식 베이스(keyword_catalog)에 실제로 등장하는
    # 용어만 뽑아서 큐레이션 프롬프트에 참고 자료로 얹는다. LLM 혼자 판단하게 두면
    # SwiGLU/Flash Attention처럼 새로 나온 용어를 놓치거나 표기가 매번 달라질 수 있어서,
    # 우리가 미리 정의해둔 정확한 명칭을 우선 참고하도록 근거를 붙여주는 것.
    known_keywords = extract_keywords(" ".join(candidate_text_parts), limit=KNOWLEDGE_BASE_HITS_LIMIT)

    try:
        curated = llm_service.curate_papers(interest, selected_keywords, candidate_summaries, count, known_keywords)
    except llm_service.QuotaExhaustedError as exc:
        return jsonify({"error": str(exc)}), 429
    except Exception:
        return jsonify({"error": "AI가 논문을 추천하는 데 실패했어요. 잠시 후 다시 시도해주세요."}), 502

    papers = []
    for item in curated:
        paper = candidates_by_id.get(item.get("id"))
        if not paper:
            continue
        paper = {
            **paper,
            "keywords": [resolve_keyword(name) for name in item.get("keywords", [])],
            "prerequisites": [resolve_keyword(name) for name in item.get("prerequisites", [])],
        }
        papers.append(paper)

    return jsonify({"search_keywords": search_keywords, "papers": papers})


@app.post("/api/curriculum")
def curriculum():
    """논문 하나(target_type="paper")든, "Transformer" 같은 순수 키워드
    (target_type="keyword")든 같은 엔드포인트로 커리큘럼 DAG를 만든다. 실제 로직은
    curriculum_service가 다 처리하고, 여기서는 입력 검증과 provider 이름 결정만 한다.
    """
    body = request.get_json(silent=True) or {}
    target_type = str(body.get("target_type", "")).strip()
    interest = str(body.get("interest", "")).strip()
    use_rag = body.get("use_rag", True)

    if target_type == "paper":
        pdf_url = str(body.get("pdf_url", "")).strip()
        title = str(body.get("title", "")).strip()
        if not pdf_url or not title:
            return jsonify({"error": "논문 제목과 PDF 링크가 필요해요."}), 400
        target = {"pdf_url": pdf_url, "title": title}
        target_label = title
        target_description = str(body.get("summary", "")).strip()
    elif target_type == "keyword":
        keyword = str(body.get("keyword", "")).strip()
        if not keyword:
            return jsonify({"error": "키워드를 입력해주세요."}), 400
        target = {"keyword": keyword}
        target_label = keyword
        target_description = ""
    else:
        return jsonify({"error": "target_type은 'paper' 또는 'keyword'여야 해요."}), 400

    provider_name = target_type if use_rag else "none"

    try:
        result = curriculum_service.generate_curriculum(
            target_label=target_label,
            target_description=target_description,
            target=target,
            provider_name=provider_name,
            interest=interest,
        )
    except (pdf_service.PdfDownloadError, pdf_service.PdfTextExtractionError) as exc:
        return jsonify({"error": str(exc)}), 502
    except llm_service.QuotaExhaustedError as exc:
        return jsonify({"error": str(exc)}), 429
    except Exception:
        return jsonify({"error": "커리큘럼 생성에 실패했어요. 잠시 후 다시 시도해주세요."}), 502

    # 커리큘럼은 더 이상 일회성이 아니다 — 생성되는 즉시 저장해서 "내 커리큘럼" 탭에서
    # 다시 찾아볼 수 있고, 노드별 학습 완료 상태/AI 추가 설명을 나중에도 이어서 쓸 수 있다.
    curriculum_id = curriculum_store.save_curriculum(target_label, target_type, result, _get_owner_id())
    result["id"] = curriculum_id
    for node in result["nodes"]:
        node["completed"] = False
        node["ai_explanation"] = None
        node["quiz"] = None

    return jsonify(result)


@app.get("/api/curricula")
def list_curricula():
    return jsonify(curriculum_store.list_curricula(_get_owner_id()))


@app.get("/api/curriculum/<curriculum_id>")
def get_curriculum(curriculum_id):
    record = curriculum_store.get_curriculum(curriculum_id, _get_owner_id())
    if not record:
        return jsonify({"error": "커리큘럼을 찾을 수 없어요."}), 404
    return jsonify(record)


@app.delete("/api/curriculum/<curriculum_id>")
def delete_curriculum(curriculum_id):
    curriculum_store.delete_curriculum(curriculum_id, _get_owner_id())
    return jsonify({"deleted": curriculum_id})


@app.post("/api/curriculum/<curriculum_id>/nodes/<node_id>/complete")
def set_node_completion(curriculum_id, node_id):
    """"학습 완료" 버튼. body: {completed: bool} (기본 true) — 다시 누르면 취소도 된다."""
    body = request.get_json(silent=True) or {}
    completed = bool(body.get("completed", True))

    record = curriculum_store.get_curriculum(curriculum_id, _get_owner_id())
    if not record:
        return jsonify({"error": "커리큘럼을 찾을 수 없어요."}), 404
    if not any(n["id"] == node_id for n in record["nodes"]):
        return jsonify({"error": "노드를 찾을 수 없어요."}), 404

    curriculum_store.set_node_completion(curriculum_id, node_id, completed)
    return jsonify({"node_id": node_id, "completed": completed})


def _gather_node_rag_context(node_title):
    """explain/quiz 둘 다 쓰는, 노드 제목을 키워드 삼은 가벼운 RAG 조회 — 커리큘럼
    생성 때보다 그 노드 하나에 더 집중한 근거를 모은다."""
    try:
        context_chunks = context_providers.keyword_provider({"keyword": node_title})
    except Exception:
        context_chunks = []
    grounding_text = " ".join(context_chunks) if context_chunks else node_title
    known_keywords = extract_keywords(grounding_text, limit=KNOWLEDGE_BASE_HITS_LIMIT)
    return context_chunks, known_keywords


@app.post("/api/curriculum/<curriculum_id>/nodes/<node_id>/explain")
def explain_curriculum_node(curriculum_id, node_id):
    """"더 자세히 설명해줘" 버튼. 이미 생성된 설명이 있으면 재생성하지 않고 그대로 돌려준다."""
    record = curriculum_store.get_curriculum(curriculum_id, _get_owner_id())
    if not record:
        return jsonify({"error": "커리큘럼을 찾을 수 없어요."}), 404

    node = next((n for n in record["nodes"] if n["id"] == node_id), None)
    if not node:
        return jsonify({"error": "노드를 찾을 수 없어요."}), 404

    if node.get("ai_explanation"):
        return jsonify({"explanation": node["ai_explanation"]})

    context_chunks, known_keywords = _gather_node_rag_context(node["title"])

    try:
        explanation = llm_service.explain_concept(
            target_label=record["target_label"],
            node_title=node["title"],
            node_description=node["description"],
            context_chunks=context_chunks,
            known_keywords=known_keywords,
        )
    except llm_service.QuotaExhaustedError as exc:
        return jsonify({"error": str(exc)}), 429
    except Exception:
        return jsonify({"error": "AI 설명 생성에 실패했어요. 잠시 후 다시 시도해주세요."}), 502

    curriculum_store.save_node_explanation(curriculum_id, node_id, explanation)
    return jsonify({"explanation": explanation})


@app.post("/api/curriculum/<curriculum_id>/nodes/<node_id>/quiz")
def quiz_curriculum_node(curriculum_id, node_id):
    """"이해도 확인 퀴즈" 버튼. 이미 생성된 퀴즈가 있으면 재생성하지 않고 그대로 돌려준다
    (문제가 계속 바뀌면 재시도 비교가 안 되기도 하고, quota도 아낀다)."""
    record = curriculum_store.get_curriculum(curriculum_id, _get_owner_id())
    if not record:
        return jsonify({"error": "커리큘럼을 찾을 수 없어요."}), 404

    node = next((n for n in record["nodes"] if n["id"] == node_id), None)
    if not node:
        return jsonify({"error": "노드를 찾을 수 없어요."}), 404

    if node.get("quiz"):
        return jsonify({"questions": node["quiz"]})

    context_chunks, known_keywords = _gather_node_rag_context(node["title"])

    try:
        questions = llm_service.generate_quiz(
            target_label=record["target_label"],
            node_title=node["title"],
            node_description=node["description"],
            context_chunks=context_chunks,
            known_keywords=known_keywords,
        )
    except llm_service.QuotaExhaustedError as exc:
        return jsonify({"error": str(exc)}), 429
    except Exception:
        return jsonify({"error": "퀴즈 생성에 실패했어요. 잠시 후 다시 시도해주세요."}), 502

    curriculum_store.save_node_quiz(curriculum_id, node_id, questions)
    return jsonify({"questions": questions})


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
