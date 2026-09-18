"""D2L(Dive into Deep Learning) 교재를 로컬 RAG 소스로 쓰기 위한 인덱서.

교재는 매 요청마다 다시 받아서 임베딩하기엔 너무 크고(전체 191개 md 파일) 내용이
거의 안 바뀌는 정적 코퍼스다. 그래서 build_index()로 딱 한 번(오프라인, 개발 중에)
청킹 + 임베딩해서 로컬 파일에 저장해두고, 런타임에는 load_index()로 메모리에
올려서 재사용한다.

벡터 DB(Chroma 등)는 일부러 안 썼다 — 이 규모(교재 한 권, 수백~천 개 청크)에서는
벡터 DB가 풀어주는 문제(수만~수백만 벡터의 빠른 근사 검색)가 애초에 없다. 진짜
필요한 건 "매 요청마다 다시 계산하지 않기"였고, 그건 로컬 파일 캐시 + 이미 있는
embedding_service의 순수 파이썬 코사인 유사도로 충분하다.

라이선스: d2l.ai는 CC BY-SA 4.0 — 저작자 표시(d2l.ai) + 동일조건변경허락 조건 하에
가공(청킹/임베딩)과 재사용이 명시적으로 허용된다.
"""

import pickle
import re
import subprocess
import tempfile
import time
from pathlib import Path

import embedding_service

REPO_URL = "https://github.com/d2l-ai/d2l-en.git"
INDEX_PATH = Path(__file__).parent / "data" / "d2l_index.pkl"
SOURCE_LABEL = "Dive into Deep Learning (d2l.ai, CC BY-SA 4.0)"

CHUNK_WORDS = 300
CHUNK_OVERLAP_WORDS = 50

# 본문이 아닌 챕터는 건너뛴다 (참고문헌 목록, 표기법 표, 설치/개발 도구 안내).
EXCLUDED_CHAPTER_DIRS = {
    "chapter_references",
    "chapter_notation",
    "chapter_installation",
    "chapter_appendix-tools-for-deep-learning",
}

# d2l 소스는 MyST 마크다운이라 코드 셀(```{.python .input}...```)이 프레임워크별로
# (mxnet/pytorch/tensorflow/jax) 4번씩 반복된다 — RAG 근거로는 의미 없고 노이즈만
# 키워서 통째로 제거한다. :label:/:numref: 같은 상호참조 디렉티브와 이미지도 제거.
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
DIRECTIVE_RE = re.compile(r":[a-zA-Z]+:`[^`]*`")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

_index_cache = None


def _clean_markdown(text):
    text = CODE_BLOCK_RE.sub(" ", text)
    text = DIRECTIVE_RE.sub(" ", text)
    text = IMAGE_RE.sub(" ", text)
    return text


def _chunk_words(words):
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_WORDS
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - CHUNK_OVERLAP_WORDS
    return chunks


def _iter_book_chunks(repo_dir):
    repo_dir = Path(repo_dir)
    chapter_dirs = sorted(
        p for p in repo_dir.glob("chapter_*") if p.is_dir() and p.name not in EXCLUDED_CHAPTER_DIRS
    )
    for chapter_dir in chapter_dirs:
        for md_file in sorted(chapter_dir.glob("*.md")):
            raw = md_file.read_text(encoding="utf-8", errors="ignore")
            words = _clean_markdown(raw).split()
            yield from _chunk_words(words)


def build_index():
    """D2L 저장소를 clone해서 청킹 + 로컬 임베딩 후 INDEX_PATH에 저장한다 (오프라인, 1회성)."""
    start = time.time()
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"{REPO_URL} clone 중...")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, tmp_dir], check=True, capture_output=True)
        chunks = list(_iter_book_chunks(tmp_dir))

    print(f"{len(chunks)}개 청크 추출 완료 ({time.time() - start:.0f}s). 로컬 임베딩 시작...")
    embeddings = embedding_service.embed_texts(chunks)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"source": SOURCE_LABEL, "chunks": chunks, "embeddings": embeddings}, f)

    print(f"저장 완료: {INDEX_PATH} ({len(chunks)}개 청크, 총 {time.time() - start:.0f}s)")


def load_index():
    global _index_cache
    if _index_cache is None:
        if not INDEX_PATH.exists():
            _index_cache = {"source": SOURCE_LABEL, "chunks": [], "embeddings": []}
        else:
            with open(INDEX_PATH, "rb") as f:
                _index_cache = pickle.load(f)
    return _index_cache


def search(query_texts, per_query=3, max_total=8):
    """query_texts(키워드/문구 목록)와 가장 관련 있는 교재 발췌를 최대 max_total개 찾는다.
    인덱스가 아직 빌드되지 않았으면(개발 환경에서 build_index()를 안 돌렸으면) 조용히 빈
    리스트를 반환한다 — 텍스트북 그라운딩 없이도 나머지 파이프라인은 그대로 동작해야 한다.
    """
    index = load_index()
    chunks = index["chunks"]
    embeddings = index["embeddings"]
    if not chunks or not query_texts:
        return []

    query_embeddings = embedding_service.embed_texts(query_texts)
    selected = set()
    for query_embedding in query_embeddings:
        selected.update(embedding_service.top_similar_indices(query_embedding, embeddings, per_query))
    ordered = sorted(selected)[:max_total]
    return [chunks[i] for i in ordered]


if __name__ == "__main__":
    build_index()
