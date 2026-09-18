"""로컬 임베딩 (fastembed, ONNX 기반 — torch 불필요).

Gemini 임베딩(gemini-embedding-001)은 무료 티어 quota가 금방 소진되는 걸 이번
프로젝트에서 실측으로 여러 번 확인했다(사용자 쪽에서도 완전히 소진됨을 확인함).
학습 로드맵의 RAG 검색(청크 vs 질의 유사도)은 quota 걱정 없이 계속 써야 하므로
로컬 임베딩으로 옮긴다. 생성(요약/큐레이션/로드맵 작성)은 계속 Gemini를 쓴다 —
이 파일은 임베딩(검색)만 담당하고 llm_service.py는 생성만 담당하도록 분리했다.

주의: BAAI/bge-small-en-v1.5는 입력을 512 토큰에서 자른다(에러 없이 조용히
truncate됨). pdf_service.CHUNK_WORDS를 이 한도에 맞춰 낮춰뒀으니, 청크 크기를
다시 키울 일이 있으면 이 모델의 토큰 한도도 같이 고려할 것.
"""

import math

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384차원, 67MB, 영어, 512 토큰 한도

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def embed_texts(texts):
    """텍스트 여러 개를 로컬에서 임베딩한다. quota/네트워크 걱정이 없다."""
    if not texts:
        return []
    return [vec.tolist() for vec in _get_model().embed(texts)]


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def top_similar_indices(query_embedding, candidate_embeddings, top_k):
    """query_embedding과 가장 비슷한 candidate_embeddings의 인덱스 top_k개 (유사도 내림차순)."""
    scored = [(_cosine_similarity(query_embedding, emb), i) for i, emb in enumerate(candidate_embeddings)]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [i for _, i in scored[:top_k]]
