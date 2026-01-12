FROM python:3.11-slim

WORKDIR /app

# Install git and build tools
RUN apt-get update && \
    apt-get install -y git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install via pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code into the container
COPY . .

# Environment variable
ENV PORT=8001

ENV PYTHONPATH=src

# Expose port
EXPOSE 8001

# Startup command using PORT variable
CMD ["sh", "-c", "uvicorn src.app.adapters.web.api:app --host 0.0.0.0 --port ${PORT}"]
