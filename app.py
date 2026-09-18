from flask import Flask, jsonify, request

import arxiv_service
import llm_service
import pdf_service
from keyword_catalog import TIERS, extract_keywords, resolve_keyword

app = Flask(__name__, static_folder="frontend", static_url_path="")

MIN_COUNT = 3
MAX_COUNT = 15
DEFAULT_COUNT = 8
CANDIDATE_POOL_SIZE = 40
KNOWLEDGE_BASE_HITS_LIMIT = 30

# 학습 로드맵: PDF를 청크로 쪼갠 뒤, 아래 다섯 관점의 질의로 각각 가장 관련도 높은
# 청크를 뽑아 합친다 (논문 전체를 다 넣지 않고, RAG로 핵심만 골라 프롬프트에 넣기 위함).
ROADMAP_ASPECT_QUERIES = [
    "motivation and the problem this paper tries to solve",
    "background knowledge and related work the reader should already know",
    "core method, architecture, or algorithm proposed in this paper",
    "mathematical formulation, equations, and key definitions",
    "experiments, results, and limitations",
]
ROADMAP_CHUNKS_PER_QUERY = 4
ROADMAP_MAX_CHUNKS_TOTAL = 16


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


@app.post("/api/roadmap")
def roadmap():
    body = request.get_json(silent=True) or {}
    title = str(body.get("title", "")).strip()
    summary = str(body.get("summary", "")).strip()
    pdf_url = str(body.get("pdf_url", "")).strip()
    interest = str(body.get("interest", "")).strip()

    if not pdf_url:
        return jsonify({"error": "PDF 링크가 없어요."}), 400

    try:
        pdf_stream = pdf_service.download_pdf(pdf_url)
        full_text = pdf_service.extract_text(pdf_stream)
    except (pdf_service.PdfDownloadError, pdf_service.PdfTextExtractionError) as exc:
        return jsonify({"error": str(exc)}), 502

    chunks = pdf_service.chunk_text(full_text)
    if not chunks:
        return jsonify({"error": "PDF 내용을 분석할 수 없었어요."}), 502

    # RAG: 논문 전체(수십 페이지일 수 있음)를 통째로 프롬프트에 넣는 대신, 다섯 관점
    # 질의와 임베딩 유사도로 실제 이해에 필요한 부분만 골라낸다.
    try:
        chunk_embeddings = llm_service.embed_texts(chunks)
        query_embeddings = llm_service.embed_texts(ROADMAP_ASPECT_QUERIES)
    except llm_service.QuotaExhaustedError as exc:
        return jsonify({"error": str(exc)}), 429
    except Exception:
        return jsonify({"error": "논문 내용을 분석하는 데 실패했어요. 잠시 후 다시 시도해주세요."}), 502

    selected_indices = set()
    for query_embedding in query_embeddings:
        selected_indices.update(
            llm_service.top_similar_indices(query_embedding, chunk_embeddings, ROADMAP_CHUNKS_PER_QUERY)
        )
    ordered_indices = sorted(selected_indices)[:ROADMAP_MAX_CHUNKS_TOTAL]
    relevant_chunks = [chunks[i] for i in ordered_indices]

    known_keywords = extract_keywords(" ".join(relevant_chunks), limit=KNOWLEDGE_BASE_HITS_LIMIT)

    try:
        steps = llm_service.generate_roadmap(title, summary, relevant_chunks, interest, known_keywords)
    except llm_service.QuotaExhaustedError as exc:
        return jsonify({"error": str(exc)}), 429
    except Exception:
        return jsonify({"error": "학습 로드맵 생성에 실패했어요. 잠시 후 다시 시도해주세요."}), 502

    resolved_steps = [
        {
            "title": step.get("title", ""),
            "description": step.get("description", ""),
            "concepts": [resolve_keyword(name) for name in step.get("concepts", [])],
        }
        for step in steps
    ]

    return jsonify({"title": title, "steps": resolved_steps})


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
