
# Use Python 3.12, matching our local development environment.
FROM python:3.12-slim

# Set the application's working directory.
WORKDIR /app

# Prevent Python from buffering logs.
ENV PYTHONUNBUFFERED=1

# Prevent Python from creating unnecessary .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Install dependencies before copying application code.
# This improves Docker build caching.
COPY requirements.txt .

# Install the exact Python dependencies we tested locally.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application, frontend, and configuration files.
COPY app/ ./app/
COPY static/ ./static/
COPY main.py .

# Bundle our existing Chroma vector database.
COPY chroma_index/ ./chroma_index/

# Configure the database's absolute path.
ENV CHROMA_PATH=/app/chroma_index

# Cloud Run uses the PORT environment variable.
ENV PORT=8080

# Document the port our application listens on.
EXPOSE 8080

# Start FastAPI using the port supplied by Cloud Run.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
