from flask import Flask, jsonify, request

import arxiv_service
import curriculum_service
import llm_service
import pdf_service
from keyword_catalog import TIERS, extract_keywords, resolve_keyword

app = Flask(__name__, static_folder="frontend", static_url_path="")

MIN_COUNT = 3
MAX_COUNT = 15
DEFAULT_COUNT = 8
CANDIDATE_POOL_SIZE = 40
KNOWLEDGE_BASE_HITS_LIMIT = 30


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

    try:
        search_keywords = llm_service.refine_search_keywords(interest, selected_keywords)
    except llm_service.QuotaExhaustedError as exc:
        return jsonify({"error": str(exc)}), 429
    except Exception:
        return jsonify({"error": "AI가 검색 키워드를 만드는 데 실패했어요. 잠시 후 다시 시도해주세요."}), 502

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

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
