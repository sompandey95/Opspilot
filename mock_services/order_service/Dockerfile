# Build context is the repo root (see docker-compose.yml)
FROM python:3.12-slim

WORKDIR /srv
RUN pip install --no-cache-dir fastapi "uvicorn[standard]"

COPY mock_services /srv/mock_services

EXPOSE 8001
CMD ["uvicorn", "mock_services.order_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
