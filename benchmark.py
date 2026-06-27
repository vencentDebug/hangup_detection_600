"""
benchmark.py - 挂断检测模型并发 & 性能压测脚本
用法:
    # 基础延迟测试
    python benchmark.py --model_dir ./output/fine_tuned_hangup

    # 并发压测
    python benchmark.py --model_dir ./output/fine_tuned_hangup --mode concurrent --workers 8 --requests 200

    # 吞吐量测试（batch size sweep）
    python benchmark.py --model_dir ./output/fine_tuned_hangup --mode throughput

    # 全量测试（含 JSONL 数据集）
    python benchmark.py --model_dir ./output/fine_tuned_hangup --mode all --data_path test.jsonl
"""

import argparse
import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MAX_LENGTH = 64

# ── 默认测试语料（无数据集时使用）──────────────────────────
DEFAULT_TEXTS = [
    "好的，再见",
    "挂了吧",
    "你好，请问有什么可以帮您？",
    "不需要了，拜拜",
    "明天再说吧",
    "请稍等，我帮您查一下",
    "88",
    "好的没问题",
    "886",
    "信号不好你再说一遍",
    "我先挂了啊",
    "稍微等一下",
    "好，就这样，再见",
    "不用了谢谢",
    "哦哦",
    "行，就酱，拜",
    "请问还有其他问题吗",
    "没有了，谢谢你",
    "嗯嗯，知道了",
    "那先这样",
]


# ──────────────────────────────────────────────────────────
# 模型加载
# ──────────────────────────────────────────────────────────
def load_model(model_dir: str):
    logger.info(f"加载模型: {model_dir}")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"运行设备: {device}")
    return tokenizer, model, device


# ──────────────────────────────────────────────────────────
# 核心推理（单次，计时精确）
# ──────────────────────────────────────────────────────────
def infer_single(text: str, tokenizer, model, device) -> float:
    """返回推理耗时（秒）"""
    inputs = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(**inputs)
    return time.perf_counter() - t0


def infer_batch(texts: list, tokenizer, model, device) -> float:
    """批量推理，返回耗时（秒）"""
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(**inputs)
    return time.perf_counter() - t0


# ──────────────────────────────────────────────────────────
# 测试 1：单条请求延迟分布
# ──────────────────────────────────────────────────────────
def test_latency(tokenizer, model, device, texts: list, warmup: int = 5, n: int = 100):
    print("\n" + "=" * 60)
    print("📊  单条请求延迟测试")
    print("=" * 60)

    # 预热
    logger.info(f"预热 {warmup} 次...")
    for i in range(warmup):
        infer_single(texts[i % len(texts)], tokenizer, model, device)

    # 正式测试
    latencies = []
    for i in range(n):
        t = infer_single(texts[i % len(texts)], tokenizer, model, device)
        latencies.append(t * 1000)  # ms

    _print_latency_stats(latencies, label="单条推理")


def _print_latency_stats(latencies_ms: list, label: str = ""):
    p50  = statistics.median(latencies_ms)
    p90  = np.percentile(latencies_ms, 90)
    p95  = np.percentile(latencies_ms, 95)
    p99  = np.percentile(latencies_ms, 99)
    mean = statistics.mean(latencies_ms)
    _min = min(latencies_ms)
    _max = max(latencies_ms)

    print(f"\n  {label}  (样本数={len(latencies_ms)})")
    print(f"  {'指标':<10} {'延迟(ms)':>10}")
    print(f"  {'-'*22}")
    print(f"  {'平均':<10} {mean:>10.2f}")
    print(f"  {'最小':<10} {_min:>10.2f}")
    print(f"  {'P50':<10} {p50:>10.2f}")
    print(f"  {'P90':<10} {p90:>10.2f}")
    print(f"  {'P95':<10} {p95:>10.2f}")
    print(f"  {'P99':<10} {p99:>10.2f}")
    print(f"  {'最大':<10} {_max:>10.2f}")
    print(f"  QPS (理论) ≈ {1000/mean:.1f} req/s")


# ──────────────────────────────────────────────────────────
# 测试 2：并发压测
# ──────────────────────────────────────────────────────────
def test_concurrent(
    tokenizer,
    model,
    device,
    texts: list,
    workers: int = 4,
    total_requests: int = 100,
    warmup: int = 5,
):
    print("\n" + "=" * 60)
    print(f"⚡  并发压测  workers={workers}  total_requests={total_requests}")
    print("=" * 60)

    # 预热
    for i in range(warmup):
        infer_single(texts[i % len(texts)], tokenizer, model, device)

    def _task(idx):
        text = texts[idx % len(texts)]
        t0 = time.perf_counter()
        infer_single(text, tokenizer, model, device)
        return (time.perf_counter() - t0) * 1000

    wall_start = time.perf_counter()
    latencies = []
    errors = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_task, i): i for i in range(total_requests)}
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
            except Exception as e:
                errors += 1
                logger.warning(f"请求失败: {e}")

    wall_elapsed = time.perf_counter() - wall_start
    actual_qps = total_requests / wall_elapsed

    _print_latency_stats(latencies, label=f"并发推理 (workers={workers})")
    print(f"\n  总耗时:     {wall_elapsed:.2f} s")
    print(f"  实际 QPS:   {actual_qps:.1f} req/s")
    print(f"  错误请求:   {errors} / {total_requests}")

    return {
        "workers": workers,
        "total_requests": total_requests,
        "wall_elapsed_s": round(wall_elapsed, 3),
        "actual_qps": round(actual_qps, 1),
        "p50_ms": round(np.percentile(latencies, 50), 2),
        "p95_ms": round(np.percentile(latencies, 95), 2),
        "p99_ms": round(np.percentile(latencies, 99), 2),
        "errors": errors,
    }


# ──────────────────────────────────────────────────────────
# 测试 3：Batch size 吞吐量扫描
# ──────────────────────────────────────────────────────────
def test_throughput(tokenizer, model, device, texts: list, repeats: int = 20):
    print("\n" + "=" * 60)
    print("🚀  Batch Size 吞吐量扫描")
    print("=" * 60)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    print(f"\n  {'Batch':>6}  {'总耗时(ms)':>12}  {'单条耗时(ms)':>14}  {'吞吐(样本/s)':>14}")
    print(f"  {'-'*52}")

    results = []
    for bs in batch_sizes:
        batch = (texts * ((bs // len(texts)) + 2))[:bs]

        # 预热
        for _ in range(3):
            infer_batch(batch, tokenizer, model, device)

        times = []
        for _ in range(repeats):
            t = infer_batch(batch, tokenizer, model, device)
            times.append(t * 1000)

        mean_total = statistics.mean(times)
        mean_per   = mean_total / bs
        throughput = bs / (mean_total / 1000)

        print(f"  {bs:>6}  {mean_total:>12.2f}  {mean_per:>14.2f}  {throughput:>14.1f}")
        results.append({"batch_size": bs, "throughput_per_s": round(throughput, 1)})

    return results


# ──────────────────────────────────────────────────────────
# 测试 4：多 worker 并发扫描（汇总表）
# ──────────────────────────────────────────────────────────
def test_concurrent_sweep(
    tokenizer,
    model,
    device,
    texts: list,
    worker_list=None,
    requests_per_worker: int = 20,
):
    if worker_list is None:
        worker_list = [1, 2, 4, 8, 16]

    print("\n" + "=" * 60)
    print("📈  并发 Worker 扫描（汇总）")
    print("=" * 60)
    print(f"\n  {'Workers':>8}  {'总请求':>8}  {'耗时(s)':>8}  {'QPS':>8}  {'P50(ms)':>9}  {'P95(ms)':>9}  {'P99(ms)':>9}  {'错误':>6}")
    print(f"  {'-'*72}")

    summary = []
    for w in worker_list:
        total = w * requests_per_worker
        r = test_concurrent(
            tokenizer, model, device, texts,
            workers=w, total_requests=total, warmup=3,
        )
        print(
            f"  {r['workers']:>8}  {r['total_requests']:>8}  "
            f"{r['wall_elapsed_s']:>8.2f}  {r['actual_qps']:>8.1f}  "
            f"{r['p50_ms']:>9.1f}  {r['p95_ms']:>9.1f}  {r['p99_ms']:>9.1f}  "
            f"{r['errors']:>6}"
        )
        summary.append(r)

    return summary


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="挂断检测模型并发 & 性能压测")
    parser.add_argument("--model_dir",  default="./output/fine_tuned_hangup", help="模型目录")
    parser.add_argument("--data_path",  default=None, help="测试语料 (.jsonl)，含 text 字段")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["latency", "concurrent", "throughput", "sweep", "all"],
        help="测试模式",
    )
    parser.add_argument("--workers",   type=int, default=8,   help="并发 worker 数（concurrent 模式）")
    parser.add_argument("--requests",  type=int, default=200, help="并发模式总请求数")
    parser.add_argument("--warmup",    type=int, default=10,  help="预热次数")
    parser.add_argument("--n_latency", type=int, default=200, help="延迟测试采样次数")
    parser.add_argument("--repeats",   type=int, default=30,  help="Batch 吞吐测试重复次数")
    parser.add_argument(
        "--sweep_workers",
        type=str,
        default="1,2,4,8,16",
        help="sweep 模式的 worker 列表，逗号分隔",
    )
    parser.add_argument("--requests_per_worker", type=int, default=25, help="sweep 模式每 worker 请求数")
    return parser.parse_args()


def load_texts_from_jsonl(path: str) -> list:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if "text" in obj:
                    texts.append(obj["text"])
    logger.info(f"从数据集加载文本: {len(texts)} 条")
    return texts


def main():
    args = parse_args()

    # 加载语料
    if args.data_path and Path(args.data_path).exists():
        texts = load_texts_from_jsonl(args.data_path)
    else:
        texts = DEFAULT_TEXTS
        logger.info(f"使用内置测试语料 ({len(texts)} 条)")

    # 加载模型
    tokenizer, model, device = load_model(args.model_dir)

    # 执行测试
    mode = args.mode
    sweep_workers = [int(x) for x in args.sweep_workers.split(",")]

    print(f"\n{'='*60}")
    print(f"  挂断检测模型 性能压测报告")
    print(f"  设备: {device}  |  模型: {args.model_dir}")
    print(f"{'='*60}")

    if mode in ("latency", "all"):
        test_latency(tokenizer, model, device, texts,
                     warmup=args.warmup, n=args.n_latency)

    if mode in ("throughput", "all"):
        test_throughput(tokenizer, model, device, texts, repeats=args.repeats)

    if mode == "concurrent":
        test_concurrent(tokenizer, model, device, texts,
                        workers=args.workers, total_requests=args.requests,
                        warmup=args.warmup)

    if mode in ("sweep", "all"):
        test_concurrent_sweep(
            tokenizer, model, device, texts,
            worker_list=sweep_workers,
            requests_per_worker=args.requests_per_worker,
        )

    print("\n✅ 压测完成")


if __name__ == "__main__":
    main()
