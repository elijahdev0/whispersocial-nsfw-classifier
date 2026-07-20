FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NSFW_MODEL_ID=Marqo/nsfw-image-detection-384 \
    NSFW_THRESHOLD=0.5

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.1+cpu \
    torchvision==0.22.1+cpu

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

RUN python -c "import timm; timm.create_model('hf_hub:Marqo/nsfw-image-detection-384', pretrained=True)" \
    && echo "Model cached"

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
