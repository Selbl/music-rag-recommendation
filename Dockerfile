FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Models cached at runtime to keep image light
COPY . .

ENV DATA_DIR=/app/data
ENV LOG_DB=/app/logs/app.db
RUN mkdir -p /app/data /app/logs

EXPOSE 8501
CMD ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]
