# ══════════════════════════════════════════════
# 挂断检测项目 Makefile
# ══════════════════════════════════════════════

.PHONY: install train test serve build up down logs clean

DATA ?= /home/ai-repository/github/2026/202604/NLU/hangup_detection_600/data/hangup_detection_600.jsonl
MODEL_DIR ?= ./output/fine_tuned_hangup

## 安装依赖
install:
	pip install -r requirements.txt

## 训练模型
train:
	python train.py

## 评估数据集
test:
	python test.py --model_dir $(MODEL_DIR) --data_path $(DATA)

## 单条测试
predict:
	python test.py --model_dir $(MODEL_DIR) --text "好的，再见"

## 导出 ONNX
onnx:
	python test.py --model_dir $(MODEL_DIR) --export_onnx

## 本地启动服务
serve:
	MODEL_DIR=$(MODEL_DIR) uvicorn serve:app --host 0.0.0.0 --port 8000 --reload

## Docker 构建
build:
	docker build -t hangup-detection:latest .

## 完整栈启动
up:
	docker-compose up -d --build

## 停止所有服务
down:
	docker-compose down

## 查看推理服务日志
logs:
	docker-compose logs -f hangup-api

## 清理输出目录
clean:
	rm -rf ./output/bert_mini_results ./output/logs/__pycache__
