# Build context is the repo root (see docker-compose.yml)
FROM python:3.12-slim

WORKDIR /srv
RUN pip install --no-cache-dir "fastapi==0.138.0" "uvicorn[standard]==0.49.0"

COPY mock_services /srv/mock_services

RUN useradd --create-home --uid 1000 mockorder && chown -R mockorder:mockorder /srv
USER mockorder

EXPOSE 8001

HEALTHCHECK --interval=5s --timeout=5s --retries=10 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health', timeout=3)"

CMD ["uvicorn", "mock_services.order_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
