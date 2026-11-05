FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/hf_cache
WORKDIR /app

# small CPU-only torch first, then the rest
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY api/requirements.txt api/requirements.txt
RUN pip install -r api/requirements.txt

# download the two AI models into the image so start-up is fast
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-v2-m3')"

COPY api api
COPY data data
EXPOSE 8000
CMD ["uvicorn", "main:app", "--app-dir", "api", "--host", "0.0.0.0", "--port", "8000"]