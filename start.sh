#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/root/final_hw"
cd "${ROOT_DIR}" || exit 1

echo "[1/6] 提交并等待联邦训练 Job 完成..."
./train.sh 1

echo "[2/6] 重新构建后端 Docker和守护进程镜像 my_repository/blog-backend:latest和my_repository/train-daemon:latest..."
docker build -t my_repository/blog-backend:latest -f backend/Dockerfile backend
docker build -t my_repository/train-daemon:latest -f daemon/Dockerfile daemon

echo "[3/6] 构建前端并将 dist 复制到项目根目录..."
./buildvue.sh

echo "[4/6] 启动 Redis / MySQL / 推理服务 / 后端 / Nginx..."
kubectl delete configmap mysql-init -n bloginfer --ignore-not-found
kubectl create configmap mysql-init --from-file=init.sql=configs/init.sql -n bloginfer

kubectl apply -f configs/redis.yaml
kubectl apply -f configs/blog.yaml
kubectl apply -f configs/mysql.yaml
kubectl apply -f configs/inference.yaml
kubectl apply -f configs/train-daemon.yaml
kubectl apply -f configs/nginx.yaml

echo "[5/6] 安装并配置 Prometheus + Grafana 监控栈..."
./prometheus_install.sh
kubectl apply -f configs/monitor.yaml

echo "[6/6] 为 Nginx 启用水平伸缩 HPA..."
kubectl apply -f configs/nginx-hpa.yaml

echo "全部模块已启动。当前 nginx Pod 数量持续监控中（Ctrl+C 退出监控）..."
watch -n 5 "kubectl get hpa nginx-hpa -n bloginfer && echo '---' && kubectl get pods -n bloginfer -l app=nginx"
