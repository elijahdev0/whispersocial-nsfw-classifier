import logging
import os
from io import BytesIO

import timm
import torch
from fastapi import FastAPI, HTTPException, Request
from PIL import Image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_ID = os.getenv("NSFW_MODEL_ID", "Marqo/nsfw-image-detection-384")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = float(os.getenv("NSFW_THRESHOLD", "0.5"))

app = FastAPI(title="nsfw-classifier")


@app.on_event("startup")
def load_model():
    log.info("Loading model %s on %s...", MODEL_ID, DEVICE)
    app.state.model = timm.create_model(
        f"hf_hub:{MODEL_ID}", pretrained=True
    ).to(DEVICE).eval()
    cfg = timm.data.resolve_model_data_config(app.state.model)
    app.state.transform = timm.data.create_transform(**cfg, is_training=False)
    app.state.labels = app.state.model.pretrained_cfg.get("label_names", ["sfw", "nsfw"])
    log.info("Model ready. Labels: %s", app.state.labels)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_ID}


@app.post("/predict")
async def predict(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(400, "Empty body")

    try:
        img = Image.open(BytesIO(body)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    with torch.no_grad():
        tensor = app.state.transform(img).unsqueeze(0).to(DEVICE)
        probs = app.state.model(tensor).softmax(dim=-1)[0].cpu()

    nsfw_idx = 1 if "nsfw" in str(app.state.labels[1]).lower() else 0
    sfw_idx = 1 - nsfw_idx

    return {
        "sfw": round(probs[sfw_idx].item(), 6),
        "nsfw": round(probs[nsfw_idx].item(), 6),
        "is_nsfw": probs[nsfw_idx].item() >= THRESHOLD,
    }
