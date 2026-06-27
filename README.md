# Hangup Detection 挂断检测

基于 BERT 的文本分类项目，用于检测对话中的挂断意图。

## 项目结构

- train.py：训练脚本
- output/fine_tuned_hangup/：训练好的模型（未包含在仓库，见下方下载）

## 模型下载

模型权重托管在 ModelScope（魔搭），克隆代码后请按以下方式下载，放到 output/fine_tuned_hangup/ 目录：

    pip install modelscope
    modelscope download --model linrener/hangup --local_dir ./output/fine_tuned_hangup

模型仓库地址：https://www.modelscope.cn/models/linrener/hangup

## 使用

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained('./output/fine_tuned_hangup')
    model = AutoModelForSequenceClassification.from_pretrained('./output/fine_tuned_hangup')
```bash
mkdir -p speakers

docker compose build

docker compose down

docker compose up -d

docker compose logs -f hangup-detection
```
