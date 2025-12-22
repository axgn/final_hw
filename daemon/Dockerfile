# 轻量级 Python 基础镜像
FROM python:3.11-slim

# 避免 Python 输出缓冲
ENV PYTHONUNBUFFERED=1

# 工作目录
WORKDIR /app

# 安装基础依赖
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 层缓存
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制守护进程代码
COPY train_daemon.py ./

# 默认端口（与 Deployment 中保持一致）
EXPOSE 8000

# 入口命令
CMD ["python", "-u", "train_daemon.py"]
