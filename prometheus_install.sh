#!/usr/bin/env bash

curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 777 get_helm.sh
./get_helm.sh
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install monitor prometheus-community/kube-prometheus-stack \
  -n monitor --create-namespace

helm upgrade monitor prometheus-community/kube-prometheus-stack \
  -n monitor -f configs/values.yaml
