from flask import Flask, jsonify, request

import arxiv_service
import llm_service
from keyword_catalog import TIERS, resolve_keyword

app = Flask(__name__, static_folder="frontend", static_url_path="")

MIN_COUNT = 3
MAX_COUNT = 15
DEFAULT_COUNT = 8
CANDIDATE_POOL_SIZE = 40


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
    for result in candidate_results:
        paper = arxiv_service.to_dict(result)
        candidates_by_id[paper["id"]] = paper
        candidate_summaries.append({"id": paper["id"], "title": paper["title"], "summary": paper["summary"]})

    try:
        curated = llm_service.curate_papers(interest, selected_keywords, candidate_summaries, count)
    except Exception:
        return jsonify({"error": "AI가 논문을 추천하는 데 실패했어요. 잠시 후 다시 시도해주세요."}), 502

    papers = []
    for item in curated:
        paper = candidates_by_id.get(item.get("id"))
        if not paper:
            continue
        paper = {**paper, "keywords": [resolve_keyword(name) for name in item.get("keywords", [])]}
        papers.append(paper)

    return jsonify({"search_keywords": search_keywords, "papers": papers})


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
