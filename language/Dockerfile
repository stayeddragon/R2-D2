FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose default FastAPI port
EXPOSE 8000

# Set an unprivileged user (optional)
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser

# Default command runs the API under Uvicorn.  Celery workers are
# started via a separate service in docker-compose.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]