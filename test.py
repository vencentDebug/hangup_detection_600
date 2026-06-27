"""
test.py - 挂断检测模型测试 & 评估脚本
用法:
    python test.py --model_dir ./output/fine_tuned_hangup --data_path <jsonl>
    python test.py --model_dir ./output/fine_tuned_hangup --text "好的，再见"
"""

import argparse
import json
import logging
import numpy as np
import torch
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

LABEL_MAP = {0: "未挂断", 1: "挂断"}
MAX_LENGTH = 64


# ──────────────────────────────────────────────
# 推理工具
# ──────────────────────────────────────────────
def load_model(model_dir: str):
    logger.info(f"加载模型: {model_dir}")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def predict_batch(texts, tokenizer, model, device, batch_size=32):
    all_probs, all_preds = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
    return all_preds, all_probs


# ──────────────────────────────────────────────
# 单条文本预测
# ──────────────────────────────────────────────
def predict_single(text: str, tokenizer, model, device):
    preds, probs = predict_batch([text], tokenizer, model, device)
    label = LABEL_MAP[preds[0]]
    confidence = max(probs[0])
    print(f"\n文本: 「{text}」")
    print(f"预测: {label}  (置信度: {confidence:.4f})")
    print(f"  未挂断概率: {probs[0][0]:.4f}")
    print(f"  挂断概率:   {probs[0][1]:.4f}")
    return label, confidence


# ──────────────────────────────────────────────
# 数据集评估
# ──────────────────────────────────────────────
def evaluate_dataset(data_path: str, tokenizer, model, device):
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    texts  = [r["text"] for r in records]
    labels = [1 if r["is_hangup"] else 0 for r in records]

    logger.info(f"评估样本数: {len(texts)}")
    preds, probs = predict_batch(texts, tokenizer, model, device)

    print("\n" + "="*60)
    print("分类报告")
    print("="*60)
    print(classification_report(labels, preds, target_names=["未挂断", "挂断"]))

    print("混淆矩阵 (行=真实, 列=预测):")
    cm = confusion_matrix(labels, preds)
    print(f"             未挂断  挂断")
    print(f"  真实未挂断   {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"  真实挂断     {cm[1][0]:4d}  {cm[1][1]:4d}")

    try:
        auc = roc_auc_score(labels, [p[1] for p in probs])
        print(f"\nROC-AUC: {auc:.4f}")
    except Exception:
        pass

    # 错误样本分析
    print("\n── 预测错误样本 (最多显示20条) ──")
    errors = [
        (texts[i], LABEL_MAP[labels[i]], LABEL_MAP[preds[i]], max(probs[i]))
        for i in range(len(texts)) if labels[i] != preds[i]
    ]
    print(f"错误总数: {len(errors)} / {len(texts)}")
    for text, true, pred, conf in errors[:20]:
        print(f"  [{true}→{pred}] conf={conf:.3f}  {text}")


# ──────────────────────────────────────────────
# 导出 ONNX（可选）
# ──────────────────────────────────────────────
def export_onnx(model_dir: str, output_path: str = "hangup_model.onnx"):
    tokenizer, model, device = load_model(model_dir)
    dummy = tokenizer(
        ["测试导出"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    dummy = {k: v.to(device) for k, v in dummy.items()}
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"], dummy.get("token_type_ids")),
        output_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch"},
            "attention_mask": {0: "batch"},
            "token_type_ids": {0: "batch"},
            "logits":         {0: "batch"},
        },
        opset_version=14,
    )
    logger.info(f"ONNX 模型已导出: {output_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="挂断检测模型测试工具")
    parser.add_argument("--model_dir", default="./output/fine_tuned_hangup", help="模型目录")
    parser.add_argument("--data_path", default=None, help="评估数据集路径 (.jsonl)")
    parser.add_argument("--text",      default=None, help="单条文本预测")
    parser.add_argument("--export_onnx", action="store_true", help="导出 ONNX 模型")
    parser.add_argument("--onnx_path", default="hangup_model.onnx", help="ONNX 输出路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tokenizer, model, device = load_model(args.model_dir)

    if args.export_onnx:
        export_onnx(args.model_dir, args.onnx_path)
    elif args.text:
        predict_single(args.text, tokenizer, model, device)
    elif args.data_path:
        evaluate_dataset(args.data_path, tokenizer, model, device)
    else:
        # 交互模式
        print("交互模式 (输入 quit 退出)")
        while True:
            text = input("\n请输入文本: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if text:
                predict_single(text, tokenizer, model, device)
