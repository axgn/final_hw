#!/usr/bin/env bash

kubectl create namespace bloginfer

kubectl apply -f configs/pv_pvc.yaml

kubectl apply -f configs/train.yaml

kubectl apply -f configs/integrajob.yaml

sudo buildkitd --oci-worker=false --containerd-worker=true


kubectl create configmap mysql-init --from-file=init.sql=configs/init.sql -n bloginfer

kubectl apply -f configs/redis.yaml
kubectl apply -f configs/blog.yaml
kubectl apply -f configs/mysql.yaml
kubectl apply -f configs/inference.yaml
kubectl apply -f configs/nginx.yaml

kubectl get svc redis blog-backend-service
sudo nerdctl -n k8s.io rmi -f my_repository/blog-backend:latest
kubectl describe pod blog-backend-deployment-795765df44-d6vgx -n bloginfer

kubectl logs deploy/blog-backend-deployment -n bloginfer

kubectl delete configmap mysql-init -n bloginfer

kubectl exec -it mysql-0 -n bloginfer -- bash

rm -rf ../sql_data

wrk -t4 -c200 -d60s http://8.216.21.524/
