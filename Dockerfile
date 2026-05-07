FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libasound2 \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN mkdir -p /app/vst

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "server.py"]