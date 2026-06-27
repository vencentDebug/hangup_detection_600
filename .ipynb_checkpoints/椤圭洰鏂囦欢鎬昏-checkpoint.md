我来查看项目文件结构，然后为你生成完整的训练、测试、部署方案。现在我来生成完整的项目文件：以下是完整项目结构和使用说明：

---

## 项目文件总览

```
.
├── train.py              # 训练脚本
├── test.py               # 测试 & 评估脚本
├── serve.py              # FastAPI 推理服务
├── requirements.txt      # Python 依赖
├── Dockerfile            # 推理服务镜像
├── docker-compose.yml    # 完整部署栈
├── Makefile              # 快捷命令
└── deploy/
    ├── nginx.conf        # 反向代理 + 限流
    ├── loki-config.yaml  # 日志聚合
    ├── promtail-config.yaml  # 日志采集
    └── prometheus.yml    # 指标采集
```

---

## 快速使用

**1. 训练**
```bash
make install
make train
# 输出模型到 ./output/fine_tuned_hangup/
```

**2. 测试评估**
```bash
make test              # 整个数据集评估（精确率/召回率/F1/混淆矩阵）
make predict           # 单条文本预测
make onnx              # 导出 ONNX 格式
```

**3. 本地服务**
```bash
make serve
# POST http://localhost:8000/predict
# Body: {"text": "好的，再见"}
```

**4. Docker 完整部署**
```bash
make up
# 推理服务:  http://localhost:8000
# Grafana:  http://localhost:3000  (admin/admin123)
# Loki 日志: http://localhost:3100
```

---

## 关键设计说明

| 模块 | 要点 |
|------|------|
| **train.py** | 8:2 划分，EarlyStopping(patience=3)，最优模型按 F1 保存，输出分类报告+混淆矩阵 |
| **test.py** | 支持单条/批量/交互/数据集评估，可选 ONNX 导出，错误样本分析 |
| **serve.py** | `/predict` 单条 + `/predict/batch` 批量，支持动态阈值覆盖，单例模型预热 |
| **日志栈** | Promtail 采集 → Loki 聚合 → Grafana 面板，保留 7 天 |