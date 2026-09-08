FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline/ pipeline/
COPY app.py main.py config.yaml ./

EXPOSE 8000

# Serve the API by default; `docker run ... python main.py <command>` runs the CLI.
# The feedback commands (predict-race --save, score-race) write to data/ + models/,
# so mount those :rw (see docker-compose.yml) or run the CLI on the host.
CMD ["gunicorn", "app:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--workers", "2", "--bind", "0.0.0.0:8000"]
