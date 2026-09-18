"""arXiv 논문 PDF 다운로드 + 텍스트 추출 + 청크 분할.

학습 로드맵 생성(RAG)의 전처리 단계. PDF 원문을 통째로 프롬프트에 넣는 대신
청크로 쪼개서, 임베딩 유사도로 실제 관련 있는 부분만 골라 쓰기 위한 준비 작업이다.
"""

import io

import requests
from pypdf import PdfReader

MAX_PDF_BYTES = 25 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 30
CHUNK_WORDS = 900
CHUNK_OVERLAP_WORDS = 150
MIN_EXTRACTED_CHARS = 500


class PdfDownloadError(RuntimeError):
    """PDF를 받아오지 못했을 때 (네트워크 오류, 404, 용량 초과 등)."""


class PdfTextExtractionError(RuntimeError):
    """PDF는 받았지만 텍스트를 추출하지 못했을 때 (스캔 이미지 PDF 등)."""


def download_pdf(pdf_url):
    try:
        response = requests.get(pdf_url, timeout=DOWNLOAD_TIMEOUT_SECONDS, stream=True)
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
        reader = PdfReader(pdf_stream)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
    except Exception as exc:
        raise PdfTextExtractionError(f"PDF에서 텍스트를 추출하지 못했어요: {exc}") from exc

    if len(text.strip()) < MIN_EXTRACTED_CHARS:
        raise PdfTextExtractionError(
            "PDF에서 읽을 수 있는 텍스트가 너무 적어요 (스캔 이미지 PDF일 수 있어요)."
        )
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
