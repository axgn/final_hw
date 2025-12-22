#!/usr/bin/env bash

kubectl delete -f configs/redis.yaml
kubectl delete -f configs/blog.yaml
kubectl delete -f configs/mysql.yaml
kubectl delete -f configs/inference.yaml
kubectl delete -f configs/nginx.yaml
kubectl delete -f configs/train-daemon.yaml
rm -rf ../sql_data

rm -f /root/final_hw/models/client_*.pt 2>/dev/null || true

