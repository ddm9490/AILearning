"""커리큘럼 생성에 근거로 쓸 "그라운딩 컨텍스트"를 제공하는 pluggable 전략들.

curriculum_service.generate_curriculum()은 이 모듈의 PROVIDERS 딕셔너리에서 이름으로
provider 하나를 골라 쓸 뿐, 그 provider가 PDF를 읽는지 arXiv 초록을 모으는지 아무것도
안 하는지는 모른다. RAG를 켜고 끄거나 그라운딩 소스를 바꾸는 건 여기서 provider 하나만
갈아끼우면 되고, curriculum_service/llm_service는 손댈 필요가 없다.

Provider의 시그니처는 전부 동일하다: (target: dict) -> list[str] (그라운딩용 발췌
텍스트들, 없으면 빈 리스트 — 빈 리스트를 반환하면 LLM이 RAG 없이 자기 지식만으로
커리큘럼을 만든다).
"""

import arxiv_service
import embedding_service
import pdf_service
import textbook_index

# 다섯 관점으로 관련 있는 발췌를 찾는다 (동기/배경지식/핵심방법론/수식·정의/응용·한계).
# 논문 PDF든 arXiv 초록 모음이든 같은 관점으로 검색한다.
ASPECT_QUERIES = [
    "motivation and the problem this concept addresses",
    "background knowledge and prerequisites the learner should already know",
    "core method, mechanism, or architecture",
    "mathematical formulation, equations, and key definitions",
    "applications, variants, and current limitations",
]
CHUNKS_PER_QUERY = 4
MAX_CHUNKS_TOTAL = 16
KEYWORD_CANDIDATE_PAPERS = 6

# 교재(textbook_index)는 여러 주제를 다루는 큰 코퍼스라서, 위 ASPECT_QUERIES를 그대로
# 쓰면 "이 라벨과 무관하게 아무 수학 얘기"를 끌어올 위험이 있다. 그래서 교재 검색은
# 먼저 라벨 자체로 좁힌 다음(아래 쿼리들), 좁혀진 후보만 ASPECT_QUERIES로 다시 거른다.
TEXTBOOK_CHUNKS_PER_QUERY = 3
TEXTBOOK_MAX_CHUNKS = 8


def _textbook_queries(label):
    return [
        label,
        f"introduction to {label}",
        f"mathematical definition of {label}",
    ]


def _textbook_chunks(label):
    return textbook_index.search(_textbook_queries(label), per_query=TEXTBOOK_CHUNKS_PER_QUERY, max_total=TEXTBOOK_MAX_CHUNKS)


def no_rag_provider(target):
    """RAG를 아예 끈다 — LLM이 사전 지식만으로 커리큘럼을 만들게 한다."""
    return []


def paper_pdf_provider(target):
    """target: {"pdf_url": str, "title": str(선택)}. 논문 PDF를 받아 청크 임베딩
    유사도로 관련 발췌를 고르고, title이 있으면 D2L 교재에서도 배경 설명을 찾아 섞는다."""
    pdf_url = target.get("pdf_url", "").strip()
    if not pdf_url:
        return []

    pdf_stream = pdf_service.download_pdf(pdf_url)
    text = pdf_service.extract_text(pdf_stream)
    chunks = pdf_service.chunk_text(text)

    title = target.get("title", "").strip()
    textbook_chunks = _textbook_chunks(title) if title else []

    return _retrieve_relevant(chunks + textbook_chunks)


def keyword_provider(target):
    """target: {"keyword": str}. arXiv에서 그 키워드의 대표 논문 몇 개를 찾아 초록을
    발췌 텍스트로 쓰고(PDF 전체를 받지 않아 훨씬 가벼움), D2L 교재에서 같은 키워드로
    찾은 배경 설명 발췌도 함께 섞는다 — 논문은 "최신 연구가 뭘 했는지", 교재는
    "개념을 처음부터 어떻게 설명하는지"를 보완해준다."""
    keyword = target.get("keyword", "").strip()
    if not keyword:
        return []

    results = arxiv_service.search_candidates([keyword], pool_size=KEYWORD_CANDIDATE_PAPERS)
    arxiv_chunks = [f"{r.title}\n{' '.join(r.summary.split())}" for r in results]
    textbook_chunks = _textbook_chunks(keyword)

    return _retrieve_relevant(arxiv_chunks + textbook_chunks)


def _retrieve_relevant(chunks):
    if not chunks:
        return []

    chunk_embeddings = embedding_service.embed_texts(chunks)
    query_embeddings = embedding_service.embed_texts(ASPECT_QUERIES)

    selected_indices = set()
    for query_embedding in query_embeddings:
        selected_indices.update(
            embedding_service.top_similar_indices(query_embedding, chunk_embeddings, CHUNKS_PER_QUERY)
        )
    ordered_indices = sorted(selected_indices)[:MAX_CHUNKS_TOTAL]
    return [chunks[i] for i in ordered_indices]


# target_type(app.py) -> provider. "none"은 RAG를 끄고 싶을 때 명시적으로 고를 수 있는 값.
PROVIDERS = {
    "paper": paper_pdf_provider,
    "keyword": keyword_provider,
    "none": no_rag_provider,
}
