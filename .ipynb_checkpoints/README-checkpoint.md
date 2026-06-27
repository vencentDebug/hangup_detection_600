
### 2. 创建 speakers 目录并启动

```bash
mkdir -p speakers

docker compose build

docker compose down

docker compose up -d

docker compose logs -f hangup-detection
```
