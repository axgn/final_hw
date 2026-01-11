import logging
import os
import threading
import time
from typing import Optional

from flask import Flask, jsonify
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
import yaml


# =========================
# 配置
# =========================
NAMESPACE = os.getenv("NAMESPACE", "bloginfer")
MODEL_DIR = os.getenv("MODEL_DIR", "/blog/models")
MIN_TRAIN_INTERVAL_SECONDS = int(os.getenv("MIN_TRAIN_INTERVAL_SECONDS", "600"))
TRAIN_POLL_INTERVAL_SECONDS = int(os.getenv("TRAIN_POLL_INTERVAL_SECONDS", "5"))
CLEAN_INTERVAL_SECONDS = int(os.getenv("CLEAN_INTERVAL_SECONDS", "3600"))
MAX_CLIENT_MODEL_AGE_SECONDS = int(os.getenv("MAX_CLIENT_MODEL_AGE_SECONDS", "86400"))

TRAIN_JOB_YAML = os.getenv("TRAIN_JOB_YAML", "/blog/configs/train.yaml")
AGG_JOB_YAML = os.getenv("AGG_JOB_YAML", "/blog/configs/Integrajob.yaml")
INFERENCE_DEPLOYMENT_NAME = os.getenv("INFERENCE_DEPLOYMENT_NAME", "inference")


# =========================
# 全局状态（简单的触发队列 + 去抖）
# =========================
_pending_triggers = 0
_is_training = False
_last_train_ts = 0.0
_state_lock = threading.Lock()

_batch_v1: Optional[client.BatchV1Api] = None
_apps_v1: Optional[client.AppsV1Api] = None


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )


def load_k8s_config() -> None:
    global _batch_v1, _apps_v1
    try:
        config.load_incluster_config()
        logging.info("Loaded in-cluster Kubernetes config")
    except config.config_exception.ConfigException:
        config.load_kube_config()
        logging.info("Loaded local kubeconfig")

    _batch_v1 = client.BatchV1Api()
    _apps_v1 = client.AppsV1Api()


def _delete_job_if_exists(name: str) -> None:
    assert _batch_v1 is not None
    try:
        _batch_v1.delete_namespaced_job(
            name=name,
            namespace=NAMESPACE,
            body=client.V1DeleteOptions(
                propagation_policy="Background",
                grace_period_seconds=0,
            ),
        )
        logging.info("Deleted existing Job %s", name)
    except ApiException as e:
        if e.status != 404:
            logging.warning("Failed to delete Job %s: %s", name, e)


def _load_job_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Job yaml not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        job = yaml.safe_load(f)

    if not isinstance(job, dict) or job.get("kind") != "Job":
        raise ValueError(f"Yaml {path} does not describe a Job")

    metadata = job.setdefault("metadata", {})
    metadata["namespace"] = NAMESPACE

    # 移除可能导致冲突的字段（静态文件里一般不会有，这里只是防御）
    for k in ["uid", "resourceVersion", "creationTimestamp"]:
        metadata.pop(k, None)

    job.pop("status", None)
    return job


def _create_job_from_yaml(path: str) -> str:
    assert _batch_v1 is not None
    job_def = _load_job_yaml(path)
    name = job_def.get("metadata", {}).get("name")
    if not name:
        raise ValueError(f"Job in {path} has no metadata.name")

    try:
        _batch_v1.create_namespaced_job(namespace=NAMESPACE, body=job_def)
        logging.info("Created Job %s from %s", name, path)
    except ApiException as e:
        logging.error("Failed to create Job from %s: %s", path, e)
        raise

    return name


def _wait_for_job_complete(name: str, timeout_seconds: int = 3600) -> None:
    assert _batch_v1 is not None
    w = watch.Watch()
    start = time.time()

    logging.info("Waiting for Job %s to complete", name)

    try:
        for event in w.stream(
            _batch_v1.list_namespaced_job,
            namespace=NAMESPACE,
            timeout_seconds=timeout_seconds,
        ):
            job: client.V1Job = event["object"]
            if job.metadata.name != name:
                continue

            if job.status.succeeded and job.status.succeeded >= 1:
                logging.info("Job %s succeeded", name)
                w.stop()
                return

            if job.status.failed and job.spec.backoff_limit is not None:
                if job.status.failed >= job.spec.backoff_limit:
                    w.stop()
                    raise RuntimeError(f"Job {name} failed after backoff limit")

            if time.time() - start >= timeout_seconds:
                w.stop()
                raise TimeoutError(f"Timeout waiting for Job {name}")
    finally:
        w.stop()


def _restart_inference_deployment() -> None:
    assert _apps_v1 is not None
    now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    body = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "train-daemon/restartedAt": now_str
                    }
                }
            }
        }
    }

    try:
        _apps_v1.patch_namespaced_deployment(
            name=INFERENCE_DEPLOYMENT_NAME,
            namespace=NAMESPACE,
            body=body,
        )
        logging.info("Patched deployment %s to trigger rollout", INFERENCE_DEPLOYMENT_NAME)
    except ApiException as e:
        logging.error("Failed to restart deployment %s: %s", INFERENCE_DEPLOYMENT_NAME, e)


def run_training_round() -> None:
    """执行一轮完整训练：train Job -> 聚合 Job -> 重启 inference。"""

    logging.info("Starting one training round")

    # 先清理旧 Job
    for name in ["train", "model-aggregator"]:
        _delete_job_if_exists(name)

    # 启动 train Job
    train_job_name = _create_job_from_yaml(TRAIN_JOB_YAML)
    _wait_for_job_complete(train_job_name)

    # 启动聚合 Job
    agg_job_name = _create_job_from_yaml(AGG_JOB_YAML)
    _wait_for_job_complete(agg_job_name)

    # 重启 inference 部署以加载最新模型
    _restart_inference_deployment()

    logging.info("One training round finished")


# =========================
# 清理历史客户端模型文件
# =========================

def clean_old_client_models() -> None:
    if not os.path.isdir(MODEL_DIR):
        logging.info("Model dir %s not found, skip cleanup", MODEL_DIR)
        return

    now = time.time()
    removed = 0

    for root, _dirs, files in os.walk(MODEL_DIR):
        for fname in files:
            if not (fname.startswith("client_") and (fname.endswith(".pt") or fname.endswith(".pth"))):
                continue

            full_path = os.path.join(root, fname)
            try:
                mtime = os.path.getmtime(full_path)
            except OSError:
                continue

            if now - mtime >= MAX_CLIENT_MODEL_AGE_SECONDS:
                try:
                    os.remove(full_path)
                    removed += 1
                    logging.info("Removed old client model: %s", full_path)
                except OSError as e:
                    logging.warning("Failed to remove %s: %s", full_path, e)

    if removed:
        logging.info("Cleanup finished, removed %d old client models", removed)


# =========================
# 后台线程：训练触发器 & 定期清理
# =========================

def train_worker_loop() -> None:
    global _pending_triggers, _is_training, _last_train_ts

    while True:
        time.sleep(TRAIN_POLL_INTERVAL_SECONDS)

        with _state_lock:
            pending = _pending_triggers
            is_training = _is_training
            last_ts = _last_train_ts

        if pending <= 0 or is_training:
            continue

        now = time.time()
        if now - last_ts < MIN_TRAIN_INTERVAL_SECONDS:
            # 还在冷却期，将触发保留在队列中
            continue

        # 条件满足，消费一次触发，开始训练
        with _state_lock:
            if _is_training or _pending_triggers <= 0:
                continue
            _pending_triggers = 0
            _is_training = True

        try:
            run_training_round()
        except Exception as e:
            logging.error("Training round failed: %s", e)
        finally:
            with _state_lock:
                _is_training = False
                _last_train_ts = time.time()


def cleanup_worker_loop() -> None:
    while True:
        time.sleep(CLEAN_INTERVAL_SECONDS)
        try:
            clean_old_client_models()
        except Exception as e:
            logging.error("Cleanup failed: %s", e)


# =========================
# Flask 应用
# =========================

app = Flask(__name__)


@app.route("/healthz", methods=["GET"])
def healthz():
    with _state_lock:
        data = {
            "pending_triggers": _pending_triggers,
            "is_training": _is_training,
            "last_train_ts": _last_train_ts,
        }
    return jsonify({"status": "ok", "data": data})


@app.route("/train/trigger", methods=["POST"])
def trigger_train():
    """对外暴露的训练触发接口。

    - 每次调用只是在内存中累加一次触发计数。
    - 后台线程会根据最小间隔合并触发，避免评论洪水导致无限制训练。
    """

    with _state_lock:
        global _pending_triggers
        _pending_triggers += 1
        pending = _pending_triggers
        is_training = _is_training

    return jsonify({
        "status": "queued",
        "pending_triggers": pending,
        "is_training": is_training,
        "min_interval_seconds": MIN_TRAIN_INTERVAL_SECONDS,
    })


def main() -> None:
    init_logging()
    load_k8s_config()

    # 启动后台线程
    t1 = threading.Thread(target=train_worker_loop, name="train-worker", daemon=True)
    t1.start()

    t2 = threading.Thread(target=cleanup_worker_loop, name="cleanup-worker", daemon=True)
    t2.start()

    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
