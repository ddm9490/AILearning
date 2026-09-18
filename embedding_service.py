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


# fastembed의 embed() 기본 batch_size는 256이라, 한 번의 onnx 실행이 (batch_size x
# 최대 시퀀스 길이) 크기로 메모리 arena를 잡는다. pdf_service.CHUNK_WORDS=300으로 자른
# 실제 논문 청크는 토큰 수가 모델 한도(512)에 거의 붙어 있어서(실측: GPT-3 논문 청크
# 평균 438토큰) 큰 batch_size는 arena를 특히 크게 키운다 — 실측(threads=1, 같은 138개
# 청크, 프로세스별 독립 측정): batch_size=32 -> 1,463MB, batch_size=8 -> 521MB,
# batch_size=4 -> 413MB, batch_size=2 -> 350MB. 이 arena는 onnxruntime이 프로세스
# 생존 기간 내내 재사용 목적으로 들고 있어서(한 번 커지면 해제 안 됨) 이후 요청에서도
# 메모리가 그 수준으로 유지된다.
#
# **주의**: 메모리는 batch_size가 지배하지만, 소요 시간은 "총 청크 개수"가 지배한다
# (청크 수만큼 순차적으로 forward pass가 도니까). 처음엔 batch_size만 2로 낮췄는데,
# 138개 청크를 통째로 batch_size=2로 임베딩하면 로컬 풀코어에서도 59초가 걸려서
# gunicorn 기본 타임아웃(30초)조차 못 버텼다. 진짜 해법은 batch_size를 더 낮추는 게
# 아니라 `context_providers.PAPER_MAX_CHUNKS_TO_EMBED`로 애초에 임베딩할 청크 총량
# 자체에 상한선을 두는 것이었다(논문이 아무리 길어도 40개로 캡). 총량이 40(+D2L 최대
# 8)개로 묶이고 나면 batch_size는 다시 메모리 쪽 레버로만 쓰면 된다 — 실제
# `context_providers.paper_pdf_provider()`로 GPT-3 논문(PDF 다운로드부터 D2L 배경
# 발췌까지 전부 포함) 전체를 실측: batch_size=4는 491MB/23.5초, batch_size=2는
# 389MB/25.6초로 메모리만 100MB 넘게 줄고 시간은 청크 총량이 이미 캡돼 있어서 거의
# 그대로였다. 그래서 batch_size=2를 최종값으로 뒀다 — 캡 이전(138청크)과 달리 이제는
# batch_size를 낮춰도 시간 손해가 거의 없다.
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
