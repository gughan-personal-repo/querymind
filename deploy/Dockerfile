FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY sql ./sql

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src.entrypoint"]
