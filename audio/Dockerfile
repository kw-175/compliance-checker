FROM python:3.11-slim

# 容器内工作目录。
WORKDIR /app

# 安装 ffmpeg 以支持音频归一化与脱敏渲染。
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖清单，利于 Docker 层缓存。
COPY audio/requirements.txt /app/audio/requirements.txt
RUN pip install --no-cache-dir -r /app/audio/requirements.txt

# 复制项目源码。
COPY . /app

EXPOSE 19001

# 启动 FastAPI 服务。
CMD ["uvicorn", "audio.server:app", "--host", "0.0.0.0", "--port", "19001"]
