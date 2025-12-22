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

	echo "找到 job/${job_name} 的 Pod: ${pod_name}，等待容器启动..."
	# 再等待一段时间，直到 Pod 进入 Running/已启动状态，再开始跟踪日志
	for i in {1..60}; do
		phase=$(kubectl get pod "${pod_name}" -n "${NAMESPACE}" -o jsonpath='{.status.phase}' 2>/dev/null || true)
		if [[ "${phase}" == "Running" || "${phase}" == "Succeeded" || "${phase}" == "Failed" ]]; then
			break
		fi
		sleep 2
	done

	echo "开始跟踪 job/${job_name} 的一个 Pod 日志: ${pod_name} (phase=${phase})"
	kubectl logs -f "${pod_name}" -n "${NAMESPACE}" || true

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
