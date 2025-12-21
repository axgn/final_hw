#!/usr/bin/env bash

kubectl delete configmap mysql-init -n bloginfer

kubectl apply -f configs/redis.yaml
kubectl apply -f configs/blog.yaml
kubectl apply -f configs/mysql.yaml
kubectl apply -f configs/inference.yaml
kubectl apply -f configs/nginx.yaml
