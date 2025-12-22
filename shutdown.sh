#!/usr/bin/env bash

kubectl delete -f configs/redis.yaml
kubectl delete -f configs/blog.yaml
kubectl delete -f configs/mysql.yaml
kubectl delete -f configs/inference.yaml
kubectl delete -f configs/nginx.yaml
rm -rf ../sql_data

# 3. 关机前清理训练历史模型文件（仅保留聚合后的全局模型）
rm -f /root/final_hw/models/client_*.pt 2>/dev/null || true
