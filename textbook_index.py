"""오픈 라이선스 학습 자료들을 로컬 RAG 소스로 쓰기 위한 인덱서 (D2L 교재 +
Hugging Face NLP Course + OpenAI Spinning Up in Deep RL).

교재/코스는 매 요청마다 다시 받아서 임베딩하기엔 너무 크고(D2L 191개 md 파일 등)
내용이 거의 안 바뀌는 정적 코퍼스다. 그래서 build_index()로 소스별로 딱 한 번
(오프라인, 개발 중에) 청킹 + 임베딩해서 로컬 파일에 저장해두고, 런타임에는
load_index()로 메모리에 올려서 재사용한다.

벡터 DB(Chroma 등)는 일부러 안 썼다 — 이 규모(소스 3개 합쳐도 수천 개 청크)에서는
벡터 DB가 풀어주는 문제(수만~수백만 벡터의 빠른 근사 검색)가 애초에 없다. 진짜
필요한 건 "매 요청마다 다시 계산하지 않기"였고, 그건 로컬 파일 캐시 + 이미 있는
embedding_service의 순수 파이썬 코사인 유사도로 충분하다.

소스별 라이선스 (전부 직접 LICENSE 파일을 확인함):
- D2L(d2l.ai): CC BY-SA 4.0 — 저작자 표시 + 동일조건변경허락 조건 하에 가공/재사용 허용.
- Hugging Face NLP Course: Apache License 2.0 — 저작권/변경사항 표시 조건 하에 자유롭게
  재사용/재배포 허용.
- OpenAI Spinning Up in Deep RL: MIT License — "documentation files"까지 명시적으로
  포함한다고 라이선스 본문에 적혀 있어서(소프트웨어뿐 아니라 문서도 대상) 셋 중 가장 명확함.
"""

import pickle
import re
import subprocess
import tempfile
import time
from pathlib import Path

import embedding_service

CHUNK_WORDS = 300
CHUNK_OVERLAP_WORDS = 50
# embedding_service.EMBED_BATCH_SIZE(2)는 512MB 배포 서버가 매 요청마다 감당해야 하는
# 메모리 한도에 맞춘 값이다. 이 스크립트는 로컬에서 한 번만 돌리는 오프라인 빌드라
# 그 제약을 받을 이유가 없어서 훨씬 크게 잡아 속도를 우선한다.
BUILD_EMBED_BATCH_SIZE = 32

_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

# D2L: MyST 마크다운 특유의 문법(코드 셀이 mxnet/pytorch/tensorflow/jax용으로 4번씩
# 반복돼서 RAG 근거로는 의미 없이 노이즈만 키움, :label:/:numref: 상호참조 디렉티브).
_D2L_DIRECTIVE_RE = re.compile(r":[a-zA-Z]+:`[^`]*`")
_D2L_EXCLUDED_CHAPTER_DIRS = {
    "chapter_references",
    "chapter_notation",
    "chapter_installation",
    "chapter_appendix-tools-for-deep-learning",
}

# HF Course: .mdx는 마크다운 + JSX라서 <CourseFloatingBanner .../>, <Tip> 같은 컴포넌트
# 태그가 섞여 있다. 태그만 벗겨내고 안쪽 텍스트(<Tip>의 실제 팁 내용 등)는 살린다.
_JSX_TAG_RE = re.compile(r"<[^>]+>")

# Spinning Up: Sphinx reStructuredText라 `.. math::`/`.. admonition::` 같은 디렉티브
# 줄과 `:math:`...`` 같은 인라인 role이 섞여 있다. 디렉티브 "줄"만 제거하고(그 아래
# 들여쓰기된 실제 수식/본문은 남긴다 — 수식·정의 근거로 오히려 유용함), role은
# 감싸는 문법만 벗겨서 내용만 남긴다.
_RST_DIRECTIVE_LINE_RE = re.compile(r"^\.\. .*$", re.MULTILINE)
_RST_ROLE_RE = re.compile(r":[a-z]+:`([^`]*)`")
_RST_UNDERLINE_RE = re.compile(r"^[=\-~^\"']{3,}\s*$", re.MULTILINE)
_SPINNINGUP_EXCLUDED_FILES = {
    # 벤치마크 그래프/표 페이지 — 산문이 아니라 RAG 근거로 의미 없음.
    "bench.rst", "bench_ddpg.rst", "bench_ppo.rst", "bench_sac.rst", "bench_td3.rst", "bench_vpg.rst",
    # 연습문제 안내/정답 코드 — 개념 설명이 아니라 과제/구현 코드 위주.
    "exercises.rst", "exercise2_1_soln.rst", "exercise2_2_soln.rst", "extra_tf_pg_implementation.rst",
}


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


def _chunk_text(text):
    return _chunk_words(text.split())


def _clean_d2l(raw):
    text = _CODE_BLOCK_RE.sub(" ", raw)
    text = _D2L_DIRECTIVE_RE.sub(" ", text)
    text = _IMAGE_RE.sub(" ", text)
    return text


def _iter_d2l_chunks(repo_dir):
    repo_dir = Path(repo_dir)
    chapter_dirs = sorted(
        p for p in repo_dir.glob("chapter_*") if p.is_dir() and p.name not in _D2L_EXCLUDED_CHAPTER_DIRS
    )
    for chapter_dir in chapter_dirs:
        for md_file in sorted(chapter_dir.glob("*.md")):
            raw = md_file.read_text(encoding="utf-8", errors="ignore")
            yield from _chunk_text(_clean_d2l(raw))


def _clean_hf_course(raw):
    text = _CODE_BLOCK_RE.sub(" ", raw)
    text = _IMAGE_RE.sub(" ", text)
    text = _JSX_TAG_RE.sub(" ", text)
    return text


def _iter_hf_course_chunks(repo_dir):
    # 영어 챕터만 쓴다 — 코스가 여러 언어로 번역돼 있는데, 다른 언어를 섞으면 한국어
    # 설명을 만들 때 오히려 방해가 된다(그리고 이 프로젝트의 다른 모든 근거 자료도 영어
    # 원문 기준이라 일관성 유지).
    en_dir = Path(repo_dir) / "chapters" / "en"
    for mdx_file in sorted(en_dir.glob("chapter*/*.mdx")):
        raw = mdx_file.read_text(encoding="utf-8", errors="ignore")
        yield from _chunk_text(_clean_hf_course(raw))


def _clean_rst(raw):
    text = _RST_DIRECTIVE_LINE_RE.sub(" ", raw)
    text = _RST_ROLE_RE.sub(r"\1", text)
    text = _RST_UNDERLINE_RE.sub(" ", text)
    return text


def _iter_spinningup_chunks(repo_dir):
    docs_dir = Path(repo_dir) / "docs"
    for rst_file in sorted(docs_dir.glob("**/*.rst")):
        if rst_file.name in _SPINNINGUP_EXCLUDED_FILES:
            continue
        raw = rst_file.read_text(encoding="utf-8", errors="ignore")
        yield from _chunk_text(_clean_rst(raw))


# 소스 하나 추가하고 싶으면 이 리스트에 항목 하나 더하면 끝 — build_index()/load_index()/
# search()는 전부 이 리스트를 기준으로 동작한다. tag는 프롬프트에 붙는 짧은 출처
# 표시(context_providers.py의 "[arXiv 논문 ...]" 같은 태그와 같은 자리에 쓰임), license는
# 이 파일 맨 위 docstring에 이미 확인해둔 라이선스 요약.
SOURCES = [
    {
        "name": "d2l",
        "tag": "D2L 교재",
        "repo_url": "https://github.com/d2l-ai/d2l-en.git",
        "branch": "master",
        "index_path": Path(__file__).parent / "data" / "d2l_index.pkl",
        "iter_chunks": _iter_d2l_chunks,
    },
    {
        "name": "hf_course",
        "tag": "HuggingFace NLP Course",
        "repo_url": "https://github.com/huggingface/course.git",
        "branch": "main",
        "index_path": Path(__file__).parent / "data" / "hf_course_index.pkl",
        "iter_chunks": _iter_hf_course_chunks,
    },
    {
        "name": "spinningup",
        "tag": "OpenAI Spinning Up",
        "repo_url": "https://github.com/openai/spinningup.git",
        "branch": "master",
        "index_path": Path(__file__).parent / "data" / "spinningup_index.pkl",
        "iter_chunks": _iter_spinningup_chunks,
    },
]

_index_cache = None


def build_index(source_name=None):
    """소스 하나(source_name) 또는 전체(None)를 clone해서 청킹 + 로컬 임베딩 후
    각자의 index_path에 저장한다 (오프라인, 1회성)."""
    targets = [s for s in SOURCES if source_name is None or s["name"] == source_name]
    if not targets:
        raise ValueError(f"알 수 없는 소스: {source_name} (선택지: {[s['name'] for s in SOURCES]})")

    for source in targets:
        start = time.time()
        print(f"[{source['name']}] {source['repo_url']} clone 중...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", source["branch"], source["repo_url"], tmp_dir],
                check=True,
                capture_output=True,
            )
            chunks = list(source["iter_chunks"](tmp_dir))

        print(f"[{source['name']}] {len(chunks)}개 청크 추출 완료 ({time.time() - start:.0f}s). 로컬 임베딩 시작...")
        # 이건 배포 서버가 요청마다 돌리는 경로가 아니라 로컬에서 한 번만 돌리는
        # 오프라인 빌드라, embedding_service의 기본값(EMBED_BATCH_SIZE=2, 512MB 배포
        # 환경에 맞춘 안전값)을 따를 이유가 없다 — 여기서는 속도를 우선해 크게 잡는다.
        embeddings = embedding_service.embed_texts(chunks, batch_size=BUILD_EMBED_BATCH_SIZE)

        source["index_path"].parent.mkdir(parents=True, exist_ok=True)
        with open(source["index_path"], "wb") as f:
            pickle.dump({"tag": source["tag"], "chunks": chunks, "embeddings": embeddings}, f)

        print(f"[{source['name']}] 저장 완료: {source['index_path']} ({len(chunks)}개 청크, 총 {time.time() - start:.0f}s)")


def load_index():
    """소스별 인덱스를 전부 읽어서 하나의 리스트([{tag, chunks, embeddings}, ...])로
    캐싱한다. 아직 build_index()를 안 돌린 소스는 조용히 빈 항목으로 채운다 — 소스
    하나가 없어도 나머지 파이프라인은 그대로 동작해야 한다."""
    global _index_cache
    if _index_cache is None:
        loaded = []
        for source in SOURCES:
            if source["index_path"].exists():
                with open(source["index_path"], "rb") as f:
                    loaded.append(pickle.load(f))
            else:
                loaded.append({"tag": source["tag"], "chunks": [], "embeddings": []})
        _index_cache = loaded
    return _index_cache


def search(query_texts, per_query=3, max_total=8):
    """query_texts(키워드/문구 목록)와 가장 관련 있는 발췌를, 소스 구분 없이 전체
    코퍼스를 통틀어 최대 max_total개 찾는다. 각 발췌 앞에는 어느 소스에서 왔는지
    짧은 태그가 붙는다(예: "[D2L 교재] ...", "[OpenAI Spinning Up] ..."). 인덱스가
    하나도 없으면(개발 환경에서 build_index()를 안 돌렸으면) 조용히 빈 리스트를
    반환한다 — 텍스트북 그라운딩 없이도 나머지 파이프라인은 그대로 동작해야 한다.
    """
    if not query_texts:
        return []

    all_texts = []  # 태그 붙인 최종 텍스트 (반환값과 1:1 대응)
    all_embeddings = []
    for index in load_index():
        for chunk, embedding in zip(index["chunks"], index["embeddings"]):
            all_texts.append(f"[{index['tag']}] {chunk}")
            all_embeddings.append(embedding)

    if not all_texts:
        return []

    query_embeddings = embedding_service.embed_texts(query_texts)
    selected = set()
    for query_embedding in query_embeddings:
        selected.update(embedding_service.top_similar_indices(query_embedding, all_embeddings, per_query))
    ordered = sorted(selected)[:max_total]
    return [all_texts[i] for i in ordered]


if __name__ == "__main__":
    import sys

    build_index(sys.argv[1] if len(sys.argv) > 1 else None)
