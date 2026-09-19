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
# 예시 커리큘럼이 하나도 없으면(첫 실행, 또는 영구 볼륨 없이 재배포로 DB가
# 초기화된 경우) 커밋해둔 정적 데이터(example_curricula.py)로 채운다 — Gemini/
# arXiv를 다시 호출하지 않아 quota 소모도, 네트워크 의존도 없다.
curriculum_store.seed_example_curricula_if_missing()

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
    # "ai_ml"(기본값)이면 기존 실측 검증된 AI/ML 전용 프롬프트 문구를 그대로 쓰고,
    # 그 외 값은 llm_service가 AI 특정 용어를 빼고 관심사에서 분야를 유추하는
    # 일반화된 프롬프트를 쓴다 (llm_service.py 참고).
    domain = str(body.get("domain", "")).strip() or "ai_ml"

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
            refined_keywords = llm_service.refine_search_keywords(interest, remaining_slots, domain)
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
    # keyword_catalog는 전부 AI/ML 용어라서 다른 분야에서는 아예 쓰지 않는다 — "Momentum"
    # (물리학의 운동량) 같은 흔한 단어가 카탈로그의 AI/ML 용어("Momentum" = SGD
    # 모멘텀)와 우연히 겹쳐서 엉뚱한 tier/색이 붙는 걸 막기 위함.
    use_catalog = domain == "ai_ml"
    known_keywords = extract_keywords(" ".join(candidate_text_parts), limit=KNOWLEDGE_BASE_HITS_LIMIT) if use_catalog else []

    try:
        curated = llm_service.curate_papers(
            interest, selected_keywords, candidate_summaries, count, known_keywords, domain
        )
    except llm_service.QuotaExhaustedError as exc:
        return jsonify({"error": str(exc)}), 429
    except Exception:
        return jsonify({"error": "AI가 논문을 추천하는 데 실패했어요. 잠시 후 다시 시도해주세요."}), 502

    papers = []
    for item in curated:
        paper = candidates_by_id.get(item.get("id"))
        if not paper:
            continue
        # item["keywords"]/["prerequisites"]는 이제 {name, tier} 객체다 — tier는
        # LLM이 직접 매긴 판정으로, resolve_keyword가 카탈로그/별칭 어디에서도 못
        # 찾은 새 용어에 한해서만 이 값을 대신 쓴다(카탈로그에 있으면 그쪽이 우선).
        # AI/ML이 아닌 분야는 use_catalog=False라 카탈로그를 아예 안 보고 LLM 판정만 쓴다.
        paper = {
            **paper,
            "reason": item.get("reason", ""),
            "keywords": [
                resolve_keyword(kw.get("name", ""), kw.get("tier"), use_catalog) for kw in item.get("keywords", [])
            ],
            "prerequisites": [
                resolve_keyword(kw.get("name", ""), kw.get("tier"), use_catalog)
                for kw in item.get("prerequisites", [])
            ],
        }
        papers.append(paper)

    return jsonify({"search_keywords": search_keywords, "papers": papers})


def _generate_curriculum_result(target_type, target, target_label, target_description, use_rag, domain, interest):
    """`/api/curriculum`과 `/api/curriculum/<id>/regenerate`가 공유하는 실제 생성
    로직. provider 결정 -> curriculum_service 호출 -> 응답에 필요한 필드 부착까지
    한 곳에서 처리한다(에러 처리는 각자 라우트에서 하므로 예외는 그대로 던진다).

    target_type이 "paper"면 pdf_url/target_description을 결과에도 같이 실어서
    저장한다 — curriculum_store는 payload를 JSON 그대로 저장하는 구조라 스키마
    변경 없이 그대로 들어간다. 이게 없으면 나중에 저장된 커리큘럼을 다시(자연어
    조건을 추가해서) 만들려고 할 때 원본 PDF를 어디서 가져와야 할지 알 수 없다."""
    provider_name = curriculum_service.provider_name_for(target_type, use_rag, domain)
    result = curriculum_service.generate_curriculum(
        target_label=target_label,
        target_description=target_description,
        target=target,
        provider_name=provider_name,
        interest=interest,
        domain=domain,
    )
    result["target_type"] = target_type
    result["domain"] = domain
    if target_type == "paper":
        result["pdf_url"] = target["pdf_url"]
        result["target_description"] = target_description
    for node in result["nodes"]:
        node["completed"] = False
        node["ai_explanation"] = None
        node["quiz"] = None
    return result


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
    # "ai_ml"(기본값)이면 기존과 동일하게 D2L/HF Course/Spinning Up 교재까지 RAG에
    # 쓴다. 그 외 값(예: "other")은 이 교재들이 AI/ML 전용이라 섞을 수 없으니 논문
    # (arXiv) 근거만 쓴다 — curriculum_service.provider_name_for() 참고.
    domain = str(body.get("domain", "")).strip() or "ai_ml"
    # 논문 카드에서 미리보기 삼아 눌러볼 때마다 "내 커리큘럼"에 쌓이는 게 불편하다는
    # 피드백을 받아서, 기본은 저장하되(독립된 "커리큘럼" 탭은 만들 의도가 분명하므로)
    # 호출부가 명시적으로 save=false를 주면 저장 없이 미리보기만 반환한다.
    save = bool(body.get("save", True))

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

    try:
        result = _generate_curriculum_result(
            target_type, target, target_label, target_description, use_rag, domain, interest
        )
    except (pdf_service.PdfDownloadError, pdf_service.PdfTextExtractionError) as exc:
        return jsonify({"error": str(exc)}), 502
    except llm_service.QuotaExhaustedError as exc:
        return jsonify({"error": str(exc)}), 429
    except Exception:
        return jsonify({"error": "커리큘럼 생성에 실패했어요. 잠시 후 다시 시도해주세요."}), 502

    if save:
        # "내 커리큘럼" 탭에서 다시 찾아볼 수 있고, 노드별 학습 완료 상태/AI 추가
        # 설명을 나중에도 이어서 쓸 수 있게 즉시 저장한다. domain도 같이 저장해야
        # 나중에 /explain, /quiz가 이 커리큘럼이 교재 RAG를 써도 되는 도메인인지 안다.
        curriculum_id = curriculum_store.save_curriculum(target_label, target_type, domain, result, _get_owner_id())
        result["id"] = curriculum_id
    else:
        # 저장 안 함 — 완료 표시/AI 설명/퀴즈처럼 id가 있어야 되는 기능은 못 쓰고,
        # 그래프/설명만 미리 보여준다. 마음에 들면 /api/curriculum/save로 따로 저장한다.
        result["id"] = None

    return jsonify(result)


@app.post("/api/curriculum/save")
def save_curriculum():
    """미리보기로만 만들어둔(POST /api/curriculum을 save=false로 호출한) 커리큘럼을
    사용자가 실제로 "내 커리큘럼에 추가"할 때 쓴다. 다시 생성(Gemini/RAG 호출)하지
    않고, 프론트가 이미 들고 있는 결과를 그대로 저장만 한다."""
    body = request.get_json(silent=True) or {}
    target_label = str(body.get("target_label", "")).strip()
    target_type = str(body.get("target_type", "")).strip()
    domain = str(body.get("domain", "")).strip() or "ai_ml"
    nodes = body.get("nodes")
    edges = body.get("edges")

    if not target_label or target_type not in ("paper", "keyword") or not nodes:
        return jsonify({"error": "저장할 커리큘럼 데이터가 올바르지 않아요."}), 400

    result = {
        "target_label": target_label,
        "used_rag": bool(body.get("used_rag", False)),
        "nodes": nodes,
        "edges": edges or [],
    }
    # 논문 기반 미리보기(save:false)를 나중에 "저장"할 때도 pdf_url/target_description을
    # 같이 넘겨받아 저장한다 — 이게 없으면 나중에 이 커리큘럼을 자연어 조건을 추가해서
    # 다시 만들려고 할 때(재생성) 원본 PDF를 어디서 가져와야 할지 알 수 없다.
    if target_type == "paper":
        result["pdf_url"] = str(body.get("pdf_url", "")).strip()
        result["target_description"] = str(body.get("target_description", "")).strip()
    curriculum_id = curriculum_store.save_curriculum(target_label, target_type, domain, result, _get_owner_id())
    result["id"] = curriculum_id
    result["target_type"] = target_type
    result["domain"] = domain
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


@app.get("/api/curricula/examples")
def list_example_curricula():
    """유명 AI/ML 논문으로 미리 만들어둔, 모든 사용자에게 공유되는 읽기 전용 예시
    커리큘럼 목록. owner_id 쿠키와 무관하게 항상 같은 목록을 반환한다."""
    return jsonify(curriculum_store.list_example_curricula())


@app.get("/api/curriculum/examples/<curriculum_id>")
def get_example_curriculum(curriculum_id):
    record = curriculum_store.get_example_curriculum(curriculum_id)
    if not record:
        return jsonify({"error": "예시 커리큘럼을 찾을 수 없어요."}), 404
    return jsonify(record)


@app.post("/api/curriculum/<curriculum_id>/regenerate")
def regenerate_curriculum(curriculum_id):
    """저장된 커리큘럼에 자연어 조건을 추가해서 완전히 새 커리큘럼으로 다시 만든다.
    원본은 건드리지 않고 그대로 두고 새 id로 저장한다 — 재생성 결과가 마음에 안
    들어도 원본으로 돌아갈 수 있어야 하기 때문(이 앱의 다른 "재생성"들도 항상 새
    항목을 만드는 것과 같은 원칙)."""
    body = request.get_json(silent=True) or {}
    interest = str(body.get("interest", "")).strip()

    record = curriculum_store.get_curriculum(curriculum_id, _get_owner_id())
    if not record:
        return jsonify({"error": "커리큘럼을 찾을 수 없어요."}), 404

    target_type = record["target_type"]
    domain = record["domain"]
    target_label = record["target_label"]
    use_rag = bool(record.get("used_rag", True))

    if target_type == "paper":
        pdf_url = record.get("pdf_url", "")
        if not pdf_url:
            # pdf_url을 저장하기 전(이 기능이 생기기 전)에 만들어진 논문 커리큘럼은
            # 재생성에 필요한 원본 PDF 링크가 없다 — 새로 논문 카드에서 만들어야 한다.
            return jsonify({"error": "이 커리큘럼은 원본 PDF 링크가 저장되기 전에 만들어져서 다시 만들 수 없어요. 논문 카드에서 새로 만들어주세요."}), 400
        target = {"pdf_url": pdf_url, "title": target_label}
        target_description = record.get("target_description", "")
    else:
        target = {"keyword": target_label}
        target_description = ""

    try:
        result = _generate_curriculum_result(
            target_type, target, target_label, target_description, use_rag, domain, interest
        )
    except (pdf_service.PdfDownloadError, pdf_service.PdfTextExtractionError) as exc:
        return jsonify({"error": str(exc)}), 502
    except llm_service.QuotaExhaustedError as exc:
        return jsonify({"error": str(exc)}), 429
    except Exception:
        return jsonify({"error": "커리큘럼 생성에 실패했어요. 잠시 후 다시 시도해주세요."}), 502

    # 원본과 구분할 수 있는 표지 — curriculum_store는 payload를 JSON 그대로 저장하니
    # 스키마 변경 없이 이 두 필드만 얹으면 목록/상세 응답에 자동으로 실려 나간다.
    result["regenerated_from"] = curriculum_id
    result["regenerated_note"] = interest

    new_id = curriculum_store.save_curriculum(target_label, target_type, domain, result, _get_owner_id())
    result["id"] = new_id
    return jsonify(result)


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


def _gather_node_rag_context(node_title, domain):
    """explain/quiz 둘 다 쓰는, 노드 제목을 키워드 삼은 가벼운 RAG 조회 — 커리큘럼
    생성 때보다 그 노드 하나에 더 집중한 근거를 모은다. domain은 그 커리큘럼이
    저장될 때의 domain(record["domain"]) — AI/ML이 아니면 교재 RAG는 건너뛴다."""
    provider_name = curriculum_service.provider_name_for("keyword", True, domain)
    provider = context_providers.PROVIDERS.get(provider_name, context_providers.no_rag_provider)
    try:
        context_chunks = provider({"keyword": node_title})
    except Exception:
        context_chunks = []
    grounding_text = " ".join(context_chunks) if context_chunks else node_title
    # keyword_catalog는 전부 AI/ML 용어라서 다른 분야에서는 아예 쓰지 않는다(recommend/
    # curriculum 흐름과 같은 이유 — 흔한 단어가 AI/ML 용어와 우연히 겹치는 걸 방지).
    known_keywords = extract_keywords(grounding_text, limit=KNOWLEDGE_BASE_HITS_LIMIT) if domain == "ai_ml" else []
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

    context_chunks, known_keywords = _gather_node_rag_context(node["title"], record["domain"])

    try:
        explanation = llm_service.explain_concept(
            target_label=record["target_label"],
            node_title=node["title"],
            node_description=node["description"],
            context_chunks=context_chunks,
            known_keywords=known_keywords,
            domain=record["domain"],
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

    context_chunks, known_keywords = _gather_node_rag_context(node["title"], record["domain"])

    try:
        questions = llm_service.generate_quiz(
            target_label=record["target_label"],
            node_title=node["title"],
            node_description=node["description"],
            context_chunks=context_chunks,
            known_keywords=known_keywords,
            domain=record["domain"],
        )
    except llm_service.QuotaExhaustedError as exc:
        return jsonify({"error": str(exc)}), 429
    except Exception:
        return jsonify({"error": "퀴즈 생성에 실패했어요. 잠시 후 다시 시도해주세요."}), 502

    curriculum_store.save_node_quiz(curriculum_id, node_id, questions)
    return jsonify({"questions": questions})


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
