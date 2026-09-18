"""arXiv 논문 PDF 다운로드 + 텍스트 추출 + 청크 분할.

학습 로드맵 생성(RAG)의 전처리 단계. PDF 원문을 통째로 프롬프트에 넣는 대신
청크로 쪼개서, 임베딩 유사도로 실제 관련 있는 부분만 골라 쓰기 위한 준비 작업이다.

pypdf 대신 pymupdf를 쓴다: 두 라이브러리를 실측 비교해보니 pypdf는 수식/아래첨자
주변에서 공백을 자주 빠뜨린다(예: "dimension dk" -> "dimensiondk", "by √dk" ->
"by√dk"). arXiv 논문은 LaTeX으로 조판된 수식이 많아서 이 차이가 실제로 발생한다.
단, pymupdf는 AGPL-3.0(또는 상용 라이선스) — 로컬/개인용으로는 문제없지만, 이 앱을
공개 서비스로 배포할 계획이 있다면 라이선스를 다시 검토할 것.
"""

import io
import re

import pymupdf
import requests

MAX_PDF_BYTES = 25 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 60
# 로컬 임베딩 모델(embedding_service.MODEL_NAME, 512 토큰 한도)에 맞춘 값. 실측해보니
# 논문 본문 영어 텍스트는 단어당 평균 ~1.5~1.6 토큰으로 토큰화된다(300단어 -> 483토큰,
# 350단어 -> 549토큰으로 이미 512를 넘김). 900단어였던 예전 값은 Gemini 임베딩 기준으로
# 정한 거라 로컬 모델에서는 청크 뒷부분이 조용히 잘려나가는 채로 임베딩됐을 것이다.
CHUNK_WORDS = 300
CHUNK_OVERLAP_WORDS = 50
MIN_EXTRACTED_CHARS = 500

# References/Bibliography는 순수 인용 목록이라 로드맵에 도움이 안 되면서 청크 슬롯만
# 차지한다 (실측: GPT-3 논문(2005.14165)에서 References 섹션이 전체 텍스트의 약
# 11%, 51개 중 5~6개 청크를 차지함). Acknowledgments는 일부러 제외했다 — 실측해보니
# 바로 뒤에 "Appendix: Details of Common Crawl Filtering" 같은 실제 방법론 내용이
# 이어지는 경우가 있어서, 거기서 자르면 유용한 내용을 같이 날려버린다. 독립된 줄로
# 된 헤딩(앞에 "1." 같은 번호가 붙을 수도 있음)만 매칭해서, 본문 중간에 "이 결과를
# 참고문헌으로 쓴다" 같은 문장이 우연히 걸리지 않게 한다.
TRAILING_SECTION_HEADING = re.compile(
    r"^\s*(?:[0-9]+\.?\s+)?(references|bibliography)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# 논문 앞부분(초록 등)에서 우연히 매칭되는 걸 막기 위해, 본문 뒤쪽 일정 비율 안에서
# 나온 매치만 실제 섹션 헤딩으로 인정한다.
TRAILING_SECTION_SEARCH_START_RATIO = 0.4


class PdfDownloadError(RuntimeError):
    """PDF를 받아오지 못했을 때 (네트워크 오류, 404, 용량 초과 등)."""


class PdfTextExtractionError(RuntimeError):
    """PDF는 받았지만 텍스트를 추출하지 못했을 때 (스캔 이미지 PDF 등)."""


def download_pdf(pdf_url):
    try:
        headers = {
            # 브라우저인 것처럼 보이거나, 본인의 서비스/이메일을 명시
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(pdf_url, headers=headers, timeout=DOWNLOAD_TIMEOUT_SECONDS, stream=True)
        response.raise_for_status()
        
    except requests.RequestException as exc:
        raise PdfDownloadError(f"PDF를 받아오지 못했어요: {exc}") from exc

    buffer = io.BytesIO()
    total = 0
    for chunk in response.iter_content(chunk_size=65536):
        total += len(chunk)
        if total > MAX_PDF_BYTES:
            raise PdfDownloadError("PDF 용량이 너무 커요 (25MB 초과).")
        buffer.write(chunk)
    buffer.seek(0)
    return buffer


def extract_text(pdf_stream):
    try:
        with pymupdf.open(stream=pdf_stream.read(), filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
    except Exception as exc:
        raise PdfTextExtractionError(f"PDF에서 텍스트를 추출하지 못했어요: {exc}") from exc

    text = _strip_trailing_sections(text)

    if len(text.strip()) < MIN_EXTRACTED_CHARS:
        raise PdfTextExtractionError(
            "PDF에서 읽을 수 있는 텍스트가 너무 적어요 (스캔 이미지 PDF일 수 있어요)."
        )
    return text


def _strip_trailing_sections(text):
    search_start = int(len(text) * TRAILING_SECTION_SEARCH_START_RATIO)
    for match in TRAILING_SECTION_HEADING.finditer(text):
        if match.start() >= search_start:
            return text[: match.start()]
    return text


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap_words=CHUNK_OVERLAP_WORDS):
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks
