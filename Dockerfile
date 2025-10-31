FROM python:3.11-slim
WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code and (later) model artifacts
COPY api/ api/
# We'll add / copy artifacts/ in a later step when the model file is ready
ENV MODEL_PATH=/app/artifacts/model.joblib

EXPOSE 8080
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]
