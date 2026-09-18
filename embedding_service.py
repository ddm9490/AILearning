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
        # threads를 명시적으로 고정한다. 기본값(None)이면 onnxruntime이
        # os.cpu_count()만큼 스레드 풀을 잡는데, 컨테이너 배포 환경에서는 이게
        # cgroup으로 제한된 실제 할당량이 아니라 호스트 머신의 전체 코어 수를
        # 반환하는 경우가 흔해서(예: 0.5 vCPU 컨테이너인데 16코어로 인식), 필요
        # 이상으로 큰 스레드 풀(스레드마다 스택+버퍼)을 만들어 메모리를 더 먹는다.
        _model = TextEmbedding(model_name=MODEL_NAME, threads=1)
    return _model


# fastembed의 embed() 기본 batch_size는 256이라, 논문 한 편(최대 100~150 청크)을
# 한 번에 넘기면 onnxruntime이 그 배치 크기에 맞춰 메모리 arena를 한 번에 키운다.
# 게다가 pdf_service.CHUNK_WORDS=300으로 자른 실제 논문 청크는 토큰 수가 모델 한도
# (512)에 거의 붙어 있다(실측: GPT-3 논문 138청크 평균 438토큰, 최대 512). 어텐션
# 메모리는 시퀀스 길이 제곱에 가깝게 늘어나므로, 512에 가까운 긴 시퀀스를 큰 배치로
# 한꺼번에 넣으면 메모리가 특히 더 폭발적으로 늘어난다 — 실측(threads=1, 같은 138청크,
# 프로세스별로 독립 측정): batch_size=32 -> 1,463MB, batch_size=8 -> 521MB,
# batch_size=4 -> 413MB, batch_size=2 -> 350MB, batch_size=1 -> 305MB. 이 arena는
# onnxruntime이 프로세스 생존 기간 내내 재사용 목적으로 들고 있어서(=한 번 커지면
# 해제 안 됨) 이후 요청에서도 메모리가 그 수준으로 유지된다. 512MB 컨테이너에 배포할
# 때는 워커 개수보다 이게 훨씬 더 결정적인 원인이었다. Flask+google-genai+pymupdf+
# fastembed+D2L 인덱스까지 다 로드한 상태에서 이 논문을 batch_size=4로 임베딩하면
# 전체 RSS가 446MB까지 나와서(512MB 대비 여유 66MB뿐) 안전 마진을 더 두려고 2로
# 낮췄다 — 청크 수십~백 개 수준에서는 배치를 더 낮춰도 체감 속도 차이가 거의 없다.
EMBED_BATCH_SIZE = 2


def embed_texts(texts):
    """텍스트 여러 개를 로컬에서 임베딩한다. quota/네트워크 걱정이 없다."""
    if not texts:
        return []
    return [vec.tolist() for vec in _get_model().embed(texts, batch_size=EMBED_BATCH_SIZE)]


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
