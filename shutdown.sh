#!/usr/bin/env bash

kubectl delete -f configs/redis.yaml
kubectl delete -f configs/blog.yaml
kubectl delete -f configs/mysql.yaml
kubectl delete -f configs/inference.yaml
kubectl delete -f configs/nginx.yaml
kubectl delete -f configs/train-daemon.yaml
kubectl delete -f configs/monitor.yaml
kubectl delete -f configs/nginx-hpa.yaml
kubectl delete -f configs/train.yaml
kubectl delete -f configs/integrajob.yaml
kubectl delete configmap mysql-init -n bloginfer --ignore-not-found
kubectl delete -f configs/pv_pvc.yaml
kubectl delete pvc data-mysql-0 -n bloginfer

rm -rf ../sql_data

rm -f /root/final_hw/models/client_*.pt 2>/dev/null || true
sudo nerdctl -n k8s.io rmi -f my_repository/blog-backend:latest
sudo nerdctl -n k8s.io rmi -f my_repository/train-daemon:latest
