kubectl delete -f configs/redis.yaml
kubectl delete -f configs/blog.yaml
kubectl delete -f configs/mysql.yaml
kubectl delete -f configs/inference.yaml
kubectl delete -f configs/nginx.yaml

kubectl delete configmap mysql-init -n bloginfer    
