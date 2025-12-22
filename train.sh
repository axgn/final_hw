#!/usr/bin/env bash

set -euo pipefail

NAMESPACE="bloginfer"

ROUNDS="${1:-1}"

stream_job_logs() {
	local job_name="$1"

	echo "等待 job/${job_name} 的 Pod 创建..."
	local pod_name=""
	for i in {1..60}; do
		pod_name=$(kubectl get pods -n "${NAMESPACE}" -l job-name="${job_name}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
		if [[ -n "${pod_name}" ]]; then
			break
		fi
		sleep 2
	done

	if [[ -z "${pod_name}" ]]; then
		echo "在预期时间内没有找到 job/${job_name} 的 Pod，直接等待 job 完成。"
		kubectl wait --for=condition=complete --timeout=3600s "job/${job_name}" -n "${NAMESPACE}"
		return
	fi

	echo "找到 job/${job_name} 的 Pod: ${pod_name}，等待至少一个 Pod Ready 再开始日志跟踪..."
	# 等待至少一个 Pod Ready，避免 ContainerCreating 阶段立刻报错
	kubectl wait --for=condition=ready pod -n "${NAMESPACE}" -l job-name="${job_name}" --timeout=300s || true

	echo "开始跟踪 job/${job_name} 的所有 Pod 日志 (label: job-name=${job_name})..."
	# 使用 label 选择器跟踪整个 Job 下所有 Pod 的日志，直到 Job 结束
	kubectl logs -f -n "${NAMESPACE}" -l job-name="${job_name}" --all-containers=true || true

	echo "Pod 日志结束，继续等待 job/${job_name} 完成..."
	kubectl wait --for=condition=complete --timeout=3600s "job/${job_name}" -n "${NAMESPACE}"
}

echo "即将执行 ${ROUNDS} 轮 联邦训练 + 模型聚合 (namespace=${NAMESPACE})"

for (( round=1; round<=ROUNDS; round++ )); do
	echo "========== Round ${round} / ${ROUNDS} =========="

	kubectl delete job/train -n "$NAMESPACE" --ignore-not-found
	kubectl delete job/model-aggregator -n "$NAMESPACE" --ignore-not-found

	kubectl apply -f configs/train.yaml
	stream_job_logs "train"

	kubectl apply -f configs/Integrajob.yaml
	stream_job_logs "model-aggregator"

	echo "正在重启 inference 部署以加载最新模型..."
	kubectl rollout restart deployment/inference -n "$NAMESPACE"

	echo "第 ${round} 轮训练与增量聚合已完成。"
done

echo "全部 ${ROUNDS} 轮训练与聚合已完成。"
