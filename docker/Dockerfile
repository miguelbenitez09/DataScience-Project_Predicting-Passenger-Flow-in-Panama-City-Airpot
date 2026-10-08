# docker/Dockerfile.api
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi "pydantic>=2" "uvicorn[standard]" joblib
COPY . .
EXPOSE 8000

# Start the FastAPI app with uvicorn (serve static web/ when present)
CMD ["uvicorn", "src.inference.serve:app", "--host", "0.0.0.0", "--port", "8000"]

