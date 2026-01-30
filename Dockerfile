# Minimal Dockerfile for pycaret-mcp-server
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*
# Install lightweight dependencies; for full PyCaret use conda-based image or extend
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir fastapi uvicorn[standard]

# Note: PyCaret has heavy dependencies and is best installed via conda; leave it to user to extend

EXPOSE 8080
CMD ["uvicorn", "pycaret_mcp.server:app", "--host", "0.0.0.0", "--port", "8080"]
