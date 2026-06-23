# 视频理解 PoC — 生产镜像
# 关键点：基于 slim 镜像，额外装系统级 ffmpeg（真正的 ffmpeg，抽帧更稳）。
FROM python:3.12-slim

# 装 ffmpeg（系统级）。这一步就是 Step 4 “容器化” 的核心价值：
# 把 ffmpeg 这种二进制依赖打进镜像，部署到哪都自带，不依赖宿主环境。
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先单独拷依赖清单并安装，利用 Docker 层缓存（依赖不变时重建很快）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再拷应用代码（改代码只会重建这层，不会重装依赖）
COPY app ./app
COPY templates ./templates

# App Service for Containers 通过 PORT 注入端口；本地默认 8000
ENV PORT=8000
EXPOSE 8000

# 单 worker：本 PoC 的任务状态存在进程内存里，多 worker 会导致
# “A 进程建的 job，B 进程查不到”。生产要多副本时改用 Redis/DB 再加 worker。
CMD ["sh", "-c", "gunicorn app.main:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT} --workers 1 --timeout 180"]
