"""arXiv 논문 검색 연동.

주의: 이 파일 이름을 arxiv.py 로 지으면 pip로 설치한 arxiv 패키지를
가려버려서 import가 깨진다. 반드시 arxiv_service.py 로 유지할 것.
"""

import arxiv

MIN_RESULTS = 1
MAX_RESULTS = 50
DEFAULT_RESULTS = 10

# 관심사를 비워둔 채 검색해도 AI 관련 논문이 나오도록 하는 기본 쿼리.
DEFAULT_QUERY = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV"

_client = arxiv.Client()


def clamp_count(raw_count):
    try:
        count = int(raw_count)
    except (TypeError, ValueError):
        return DEFAULT_RESULTS
    return max(MIN_RESULTS, min(MAX_RESULTS, count))


def search_candidates(keywords, pool_size=40):
    """여러 키워드를 OR로 묶어 관련도 순으로 후보 논문 풀을 가져온다.

    최종 순위는 LLM이 매기므로 여기서는 정렬을 신경 쓰지 않고 관련도(Relevance)
    순으로만 가져온다. (참고: sort_by=SubmittedDate와 검색어를 같이 쓰면 검색어를
    사실상 무시하는 arXiv API 버그가 있어서, 날짜 정렬이 필요한 곳에서는 반드시
    Relevance로 가져온 뒤 이 코드에서 직접 정렬해야 한다.)
    """
    terms = [kw.strip().replace('"', "") for kw in keywords if kw.strip()]
    if not terms:
        query = DEFAULT_QUERY
    else:
        query = " OR ".join(f'all:"{term}"' for term in terms)

    search = arxiv.Search(
        query=query,
        max_results=pool_size,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = list(_client.results(search))

    # 같은 논문이 여러 키워드에 걸려 중복될 수 있어 id 기준으로 정리한다.
    seen = {}
    for result in results:
        seen.setdefault(result.get_short_id(), result)
    return list(seen.values())


def to_dict(result):
    return {
        "id": result.get_short_id(),
        "title": " ".join(result.title.split()),
        "authors": [author.name for author in result.authors],
        "year": result.published.year,
        "published": result.published.isoformat(),
        "summary": " ".join(result.summary.split()),
        "pdf_url": result.pdf_url,
        "abs_url": result.entry_id,
        "categories": result.categories,
    }
