"""
train.py - 挂断检测模型训练脚本
数据集: hangup_detection_600.jsonl
模型: boltuix/bert-mini (本地路径)
标签: is_hangup -> True(1) / False(0)
"""

import os
import json
import logging
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# 路径配置（按实际环境修改）
MODEL_PATH   = "/home/ai-repository/models/dienstag/rbt3"
DATA_PATH    = "/home/ai-repository/github/2026/202604/NLU/hangup_detection_600/data/hangup_detection_600.jsonl"
OUTPUT_DIR   = "./output/fine_tuned_hangup"
LOGGING_DIR  = "./output/logs"
MAX_LENGTH   = 64
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
SEED         = 42

# ──────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────
def load_jsonl(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"加载数据 {len(records)} 条，路径: {path}")
    return records


class HangupDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ──────────────────────────────────────────────
# 评估指标
# ──────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall":    recall_score(labels, preds, average="binary", zero_division=0),
    }


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1. 加载数据
    records = load_jsonl(DATA_PATH)
    texts  = [r["text"] for r in records]
    labels = [1 if r["is_hangup"] else 0 for r in records]

    pos = sum(labels)
    logger.info(f"正样本(挂断): {pos} | 负样本(未挂断): {len(labels)-pos}")

    # 2. 划分训练/验证集 (8:2)
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    logger.info(f"训练集: {len(tr_texts)} | 验证集: {len(val_texts)}")

    # 3. 加载分词器与模型
    logger.info(f"加载模型: {MODEL_PATH}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)

    # 4. 构建 Dataset
    train_dataset = HangupDataset(tr_texts,  tr_labels,  tokenizer, MAX_LENGTH)
    eval_dataset  = HangupDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    # 5. 训练参数
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=LOGGING_DIR,
        logging_steps=20,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",            # 如需 wandb 改为 "wandb"
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 7. 训练
    logger.info("开始训练...")
    trainer.train()

    # 8. 保存最优模型
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"模型已保存至: {OUTPUT_DIR}")

    # 9. 验证集详细报告
    preds_output = trainer.predict(eval_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)
    logger.info("\n" + classification_report(val_labels, preds, target_names=["未挂断", "挂断"]))
    logger.info("混淆矩阵:\n" + str(confusion_matrix(val_labels, preds)))


if __name__ == "__main__":
    main()
