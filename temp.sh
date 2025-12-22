#!/usr/bin/env bash

sudo buildkitd --oci-worker=false --containerd-worker=true

wrk -t4 -c200 -d60s http://8.216.0.230:30080

kubectl get pods -n bloginfer

kubectl logs deploy/train-daemon -n bloginfer

kubectl exec -it mysql-0 -n bloginfer -- bash

show databases;
use blog;
show tables;
select * from users;



kubectl get svc redis blog-backend-service
sudo nerdctl -n k8s.io rmi -f my_repository/blog-backend:latest
kubectl describe pod blog-backend-deployment-795765df44-d6vgx -n bloginfer

kubectl logs deploy/blog-backend-deployment -n bloginfer

kubectl delete configmap mysql-init -n bloginfer

kubectl exec -it mysql-0 -n bloginfer -- bash

rm -rf ../sql_data

wrk -t4 -c200 -d60s http://8.216.21.524/
