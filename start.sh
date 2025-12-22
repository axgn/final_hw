#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/root/final_hw"
cd "${ROOT_DIR}" || exit 1

echo "[1/7] 提交并等待联邦训练 Job 完成..."
kubectl apply -f configs/train.yaml
kubectl wait --for=condition=complete --timeout=600s job/train -n bloginfer

echo "[2/7] 触发参数聚合 Job，并等待完成..."
kubectl apply -f configs/Integrajob.yaml
kubectl wait --for=condition=complete --timeout=600s job/model-aggregator -n bloginfer

echo "[3/7] 重新构建后端 Docker 镜像 my_repository/blog-backend:latest..."
./build_backend.sh || echo "build_backend.sh 未找到，跳过本地构建（请确保镜像已存在于镜像仓库）"

echo "[4/7] 构建前端并将 dist 复制到项目根目录..."
./buildvue.sh

echo "[5/7] 启动 Redis / MySQL / 推理服务 / 后端 / Nginx..."
kubectl delete configmap mysql-init -n bloginfer --ignore-not-found
kubectl create configmap mysql-init --from-file=init.sql=configs/init.sql -n bloginfer

kubectl apply -f configs/redis.yaml
kubectl apply -f configs/blog.yaml
kubectl apply -f configs/mysql.yaml
kubectl apply -f configs/inference.yaml
kubectl apply -f configs/nginx.yaml

echo "[6/7] 安装并配置 Prometheus + Grafana 监控栈..."
./prometheus_install.sh
kubectl apply -f configs/monitor.yaml

echo "[7/7] 为 Nginx 启用水平伸缩 HPA..."
kubectl apply -f configs/nginx-hpa.yaml

echo "全部模块已启动。当前 nginx Pod 数量持续监控中（Ctrl+C 退出监控）..."
watch -n 5 "kubectl get hpa nginx-hpa -n bloginfer && echo '---' && kubectl get pods -n bloginfer -l app=nginx"
