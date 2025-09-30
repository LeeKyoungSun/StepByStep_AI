FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 시스템 의존성(필요시 추가)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 요구사항 설치
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 앱 복사
COPY . /app
ENV PYTHONPATH=/app

EXPOSE 8000

# 프로덕션에선 --workers/--timeout 등 조정 가능
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]