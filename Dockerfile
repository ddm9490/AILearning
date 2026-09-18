# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13.14

FROM python:${PYTHON_VERSION}-slim

LABEL fly_launch_runtime="flask"

WORKDIR /code

COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY . .

EXPOSE 8080

# flask run은 개발 서버라 프로덕션에 안 맞아서 gunicorn으로 띄운다.
# --workers 1: fastembed(로컬 임베딩) 모델이 프로세스마다 따로 로드되는데
# (~270MB, embedding_service._get_model()), 워커를 늘리면 그만큼 메모리가
# 배수로 늘어난다 — 동시 처리량은 워커가 아니라 --threads로 확보한다.
# --worker-class gthread --threads 4: 스레드는 프로세스를 새로 안 띄우니
# 모델을 다시 로드하지 않으면서, Gemini API 호출처럼 네트워크 대기가 긴
# 요청들을 동시에 처리할 수 있게 해준다.
# --timeout 120: 커리큘럼 생성은 PDF 다운로드+RAG+Gemini 호출까지 겹쳐서
# 실측 10~40초대가 걸린다 — gunicorn 기본 타임아웃(30초)보다 넉넉하게 잡아서
# 정상적으로 처리 중인 요청이 중간에 강제 종료되지 않게 한다.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", \
     "--worker-class", "gthread", "--threads", "4", "--timeout", "120", "app:app"]
