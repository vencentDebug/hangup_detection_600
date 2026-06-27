"""
serve.py - 挂断检测 FastAPI 推理服务
启动: uvicorn serve:app --host 0.0.0.0 --port 8000
"""

import os
import time
import logging
import numpy as np
import torch
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import BertTokenizer, BertForSequenceClassification

# ──────────────────────────────────────────────
# 日志配置
# ──────────────────────────────────────────────
LOG_DIR = os.getenv("LOG_DIR", "./logs")
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{LOG_DIR}/serve.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("hangup_service")

# ──────────────────────────────────────────────
# 全局模型（单例）
# ──────────────────────────────────────────────
MODEL_DIR  = os.getenv("MODEL_DIR", "./output/fine_tuned_hangup")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "64"))
THRESHOLD  = float(os.getenv("THRESHOLD", "0.5"))   # 可调整判断阈值

_tokenizer = None
_model     = None
_device    = None
_load_time = None


def get_model():
    global _tokenizer, _model, _device, _load_time
    if _model is None:
        logger.info(f"加载模型: {MODEL_DIR}")
        t0 = time.time()
        _tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        _model     = BertForSequenceClassification.from_pretrained(MODEL_DIR)
        _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device)
        _model.eval()
        _load_time = time.time() - t0
        logger.info(f"模型加载完成，设备: {_device}，耗时: {_load_time:.2f}s")
    return _tokenizer, _model, _device


# ──────────────────────────────────────────────
# Lifespan（预热）
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()                          # 启动时预加载
    yield


# ──────────────────────────────────────────────
# FastAPI 应用
# ──────────────────────────────────────────────
app = FastAPI(
    title="挂断检测服务",
    description="基于 BERT-mini 的电话挂断意图检测",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# 请求 / 响应模型
# ──────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., example="好的，再见", description="待检测文本")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="覆盖默认阈值")


class PredictItem(BaseModel):
    text: str
    threshold: Optional[float] = None


class BatchPredictRequest(BaseModel):
    items: List[PredictItem] = Field(..., max_length=64)


class PredictResponse(BaseModel):
    text: str
    is_hangup: bool
    label: str
    hangup_prob: float
    non_hangup_prob: float
    threshold: float
    latency_ms: float


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total: int
    latency_ms: float


# ──────────────────────────────────────────────
# 推理核心
# ──────────────────────────────────────────────
def _infer(texts: List[str]) -> List[dict]:
    tokenizer, model, device = get_model()
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
    return probs.tolist()


# ──────────────────────────────────────────────
# 路由
# ──────────────────────────────────────────────
@app.get("/health")
def health():
    tokenizer, model, device = get_model()
    return {
        "status": "ok",
        "model_dir": MODEL_DIR,
        "device": str(device),
        "load_time_s": round(_load_time, 3) if _load_time else None,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request):
    t0 = time.time()
    try:
        probs = _infer([req.text])
    except Exception as e:
        logger.error(f"推理异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    threshold = req.threshold if req.threshold is not None else THRESHOLD
    p_hangup  = probs[0][1]
    is_hangup = p_hangup >= threshold
    latency   = (time.time() - t0) * 1000

    logger.info(
        f"predict | text={req.text!r} | is_hangup={is_hangup} "
        f"| prob={p_hangup:.4f} | {latency:.1f}ms | ip={request.client.host}"
    )

    return PredictResponse(
        text=req.text,
        is_hangup=is_hangup,
        label="挂断" if is_hangup else "未挂断",
        hangup_prob=round(p_hangup, 4),
        non_hangup_prob=round(probs[0][0], 4),
        threshold=threshold,
        latency_ms=round(latency, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest, request: Request):
    t0 = time.time()
    texts = [item.text for item in req.items]
    try:
        probs_list = _infer(texts)
    except Exception as e:
        logger.error(f"批量推理异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    results = []
    for item, probs in zip(req.items, probs_list):
        threshold = item.threshold if item.threshold is not None else THRESHOLD
        p_hangup  = probs[1]
        is_hangup = p_hangup >= threshold
        results.append(PredictResponse(
            text=item.text,
            is_hangup=is_hangup,
            label="挂断" if is_hangup else "未挂断",
            hangup_prob=round(p_hangup, 4),
            non_hangup_prob=round(probs[0], 4),
            threshold=threshold,
            latency_ms=0,
        ))

    total_latency = (time.time() - t0) * 1000
    logger.info(f"batch_predict | n={len(texts)} | {total_latency:.1f}ms | ip={request.client.host}")

    return BatchPredictResponse(
        results=results,
        total=len(results),
        latency_ms=round(total_latency, 2),
    )


@app.get("/")
def root():
    return {"message": "挂断检测服务已启动", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False, workers=1)
