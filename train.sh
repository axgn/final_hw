#!/usr/bin/env bash

set -euo pipefail

NAMESPACE="bloginfer"

# 训练轮数，默认 1 轮，可通过第一个参数指定
ROUNDS="${1:-1}"

echo "即将执行 ${ROUNDS} 轮 联邦训练 + 模型聚合 (namespace=${NAMESPACE})"

for (( round=1; round<=ROUNDS; round++ )); do
	echo "========== Round ${round} / ${ROUNDS} =========="

	# 确保上一次的 Job 资源被清理，避免命名冲突
	kubectl delete job/train -n "$NAMESPACE" --ignore-not-found
	kubectl delete job/model-aggregator -n "$NAMESPACE" --ignore-not-found

	# 启动一轮联邦训练（多个客户端 Pod）
	kubectl apply -f configs/train.yaml

	# 等待训练 Job 全部完成（completions 个 Pod 都跑完）
	kubectl wait --for=condition=complete --timeout=3600s job/train -n "$NAMESPACE"

	# 启动一次增量式模型聚合 Job
	kubectl apply -f configs/Integrajob.yaml

	# 等待聚合完成
	kubectl wait --for=condition=complete --timeout=3600s job/model-aggregator -n "$NAMESPACE"

	echo "第 ${round} 轮训练与增量聚合已完成。"
done

echo "全部 ${ROUNDS} 轮训练与聚合已完成。"
