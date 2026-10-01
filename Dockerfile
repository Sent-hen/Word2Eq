# syntax=docker/dockerfile:1.7
# CPU inference image. Build after training:
#   docker build --build-arg ARTIFACT=runs/base/artifact -t word2eq:latest .
FROM python:3.12-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY word2eq ./word2eq
RUN pip install --no-cache-dir --prefix=/install ".[serve]"

FROM python:3.12-slim
ARG ARTIFACT=runs/base/artifact
ENV PYTHONUNBUFFERED=1 \
    WORD2EQ_ARTIFACT=/models/current \
    WORD2EQ_THREADS=2 \
    OMP_NUM_THREADS=2
COPY --from=build /install /usr/local
COPY ${ARTIFACT} /models/current
RUN useradd --system --uid 10001 app
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=2s --start-period=30s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2).status == 200 else 1)"
ENTRYPOINT ["word2eq", "serve", "--host", "0.0.0.0", "--port", "8000"]
