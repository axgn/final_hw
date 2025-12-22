#!/usr/bin/env bash

set -euo pipefail

NAMESPACE="bloginfer"

ROUNDS="${1:-1}"

echo "即将执行 ${ROUNDS} 轮 联邦训练 + 模型聚合 (namespace=${NAMESPACE})"

for (( round=1; round<=ROUNDS; round++ )); do
	echo "========== Round ${round} / ${ROUNDS} =========="

	kubectl delete job/train -n "$NAMESPACE" --ignore-not-found
	kubectl delete job/model-aggregator -n "$NAMESPACE" --ignore-not-found

	kubectl apply -f configs/train.yaml

	kubectl wait --for=condition=complete --timeout=3600s job/train -n "$NAMESPACE"

	kubectl apply -f configs/Integrajob.yaml

	kubectl wait --for=condition=complete --timeout=3600s job/model-aggregator -n "$NAMESPACE"

	echo "正在重启 inference 部署以加载最新模型..."
	kubectl rollout restart deployment/inference -n "$NAMESPACE"

	echo "第 ${round} 轮训练与增量聚合已完成。"
done

echo "全部 ${ROUNDS} 轮训练与聚合已完成。"
