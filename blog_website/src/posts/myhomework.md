# 云计算和大数据概论课程作业

## 基于 Kubernetes 的云原生个人博客系统与联邦学习模型训练/推理平台（实现全流程的自动化）

---

## 一、项目整体概述

本项目构建了一个运行于 Kubernetes 集群之上的**高度自动化的多工作负载云原生系统**，核心目标不仅是功能实现本身，更强调 **从模型训练、参数聚合、模型部署到在线推理与业务服务的全流程自动化**。

系统以 Kubernetes 作为统一调度与控制平面，将模型训练、联邦参数聚合、在线推理服务与 Web 应用服务全部纳入容器编排体系，实现：

* 训练任务的自动调度与生命周期管理
* 模型参数聚合流程的自动触发与执行
* 推理服务的自动部署、自动恢复与自动扩展
* 博客业务服务与模型服务之间的自动解耦与协同运行

在算法层面，项目采用 **PyTorch 框架**，基于 **LSTM 神经网络结构** 实现中文文本情感分类任务，并引入 **联邦学习（Federated Learning）** 训练范式，训练结束后自动开启相关聚合任务，通过多训练节点协同优化模型参数，同时使用轮次聚合和基于样本权重的的平均，实现对训练过程的优化，在不共享原始数据的前提下完成模型训练，为分布式、自动化训练流程提供算法基础。

在系统层面，项目采用 **前后端分离架构 + 微服务化设计**，并通过 Kubernetes 对各组件进行容器编排、状态管理与弹性伸缩，使整个系统能够在最少人工干预的情况下完成部署、运行与演化。

整体的自动化脚本：
```sh
ROOT_DIR="/root/final_hw"
cd "${ROOT_DIR}" || exit 1

echo "[1/9] 创建pv和pvc..."

kubectl apply -f configs/pv_pvc.yaml

echo "[2/9] 构建后端 Docker和守护进程镜像 my_repository/blog-backend:latest和my_repository/train-daemon:latest..."
sudo nerdctl -n k8s.io build -t my_repository/blog-backend:latest -f backend/Dockerfile backend
sudo nerdctl -n k8s.io build -t my_repository/train-daemon:latest -f daemon/Dockerfile daemon

echo "[3/9] 启动 Redis / MySQL的数据库服务 ..."

kubectl delete configmap mysql-init -n bloginfer --ignore-not-found
kubectl create configmap mysql-init --from-file=init.sql=configs/init.sql -n bloginfer
kubectl apply -f configs/redis.yaml
kubectl apply -f configs/mysql.yaml

echo "[4/9] 启动守护进程 ..."
kubectl apply -f configs/train-daemon.yaml

echo "[5/9] 提交并等待联邦训练 Job 完成..."
./train.sh 5


echo "[6/9] 构建前端并将 dist 复制到项目根目录..."
./buildvue.sh

echo "[7/9] 启动 Redis / MySQL / 推理服务 / 后端 / 守护进程 / Nginx..."

kubectl apply -f configs/inference.yaml
kubectl apply -f configs/blog.yaml
kubectl apply -f configs/nginx.yaml

echo "[8/9] 安装并配置 Prometheus + Grafana 监控栈..."
./prometheus_install.sh
kubectl apply -f configs/monitor.yaml

echo "[9/9] 为 Nginx 启用水平伸缩 HPA..."
kubectl apply -f configs/nginx-hpa.yaml

echo "全部模块已启动。当前 nginx Pod 数量持续监控中（Ctrl+C 退出监控）..."
watch -n 5 "kubectl get hpa nginx-hpa -n bloginfer && echo '---' && kubectl get pods -n bloginfer -l app=nginx"
```

训练部分的自动化脚本：
```sh
#!/usr/bin/env bash
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
	kubectl wait --for=condition=ready pod -n "${NAMESPACE}" -l job-name="${job_name}" --timeout=300s || true

	echo "开始跟踪 job/${job_name} 的所有 Pod 日志 (label: job-name=${job_name})..."
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

	echo "第 ${round} 轮训练与增量聚合已完成。"
done

echo "全部 ${ROUNDS} 轮训练与聚合已完成。"

```

---

## 二、系统架构与 Pod 级划分

这是系统的架构图：
![alt text](homework/image.png)

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: blog-pv
  labels:
    app: blog
spec:
  capacity:
    storage: 7Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  hostPath:
    path: /root/final_hw
    type: DirectoryOrCreate
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: mysql-pv
  labels:
    app: mysql
spec:
  capacity:
    storage: 5Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  hostPath:
    path: /root/sql_data
    type: DirectoryOrCreate
```

```yaml
---
apiVersion: v1
kind: Service
metadata:
  name: mysql
  namespace: bloginfer
spec:
  type: ClusterIP
  ports:
    - port: 3306
      targetPort: 3306
  selector:
    app: mysql
```

系统在 Kubernetes 中被拆分为多类功能明确的 Pod，分别对应不同的工作负载类型：

### 1. 模型训练 Pod（Training Workers）

* 作为 **Batch Job 类型工作负载** 存在
* 各 Pod 持有彼此独立的内存数据子集，内存数据在节点之间保持隔离（Data Isolation）
* 使用 PyTorch 完成本地模型训练，仅对外暴露每一个批次的参数信息

使用job，而非deployment
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: train
  namespace: bloginfer
spec:
  completions: 2
  parallelism: 2
  backoffLimit: 1
  template:
    metadata:
      labels:
        app: train
```

### 2. 参数聚合 Pod（Federated Aggregator）

* 作为联邦学习中的中心协调节点
* 负责接收各训练 Pod 上传的梯度
* 采用 **Gradient Averaging（梯度平均）** 而非参数平均进行模型更新
* 在非 IID 数据分布条件下，该方式具有更稳定的收敛特性

### 3. 模型推理 Pod（Inference Service）

* 基于 FastAPI 构建
* 属于 **无状态服务（Stateless Service）**
* 可通过增加副本数量进行横向扩展
* 负责对外提供实时情感分析推理接口

### 4. 博客后端 Pod（Application Backend）

* 使用 **C++ 与 Drogon 框架** 实现
* 负责用户管理、文章管理、评论系统等核心业务逻辑
* 通过 RESTful API 与前端和数据库交互

```Dockerfile
FROM drogonframework/drogon:latest AS builder

ENV MYSQL_HOST=mysql
ENV REDIS_HOST=redis

WORKDIR /app

COPY . /app

RUN mkdir -p build \
	&& cd build \
	&& cmake .. -DCMAKE_BUILD_TYPE=Release \
	&& cmake --build . --config Release


FROM drogonframework/drogon:latest

WORKDIR /app

COPY --from=builder /app/build/blog_backend /app/blog_backend

COPY --from=builder /app/build/config.json /app/config.json
COPY sql /app/sql

RUN mkdir -p /app/logs /app/uploads

EXPOSE 8080

WORKDIR /app

CMD ["./blog_backend"]
```


```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: blog-backend-deployment
  namespace: bloginfer
  labels:
    app: blog-backend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: blog-backend
  template:
    metadata:
      labels:
        app: blog-backend
    spec:
      containers:
        - name: blog-backend
          image: my_repository/blog-backend:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8080
          env:
            - name: LISTEN_PORT
              value: "8080"
```

### 5. Nginx Pod（Frontend & Gateway）

* 用于前端静态资源托管
* 负责请求反向代理与路由转发
* 使用NodePort对外暴露端口

```sh
location /api/ {
  proxy_pass http://blog-backend-service.bloginfer.svc.cluster.local:8080;
  proxy_set_header Host $host;
  proxy_set_header X-Real-IP $remote_addr;
  proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
  proxy_set_header X-Forwarded-Proto $scheme;
}
```

```json
{
  "listeners": [
    {
      "address": "0.0.0.0",
      "port": 8080
    }
  ],
  "log": {
    "log_path": "./logs",
    "log_level": "INFO",
    "logfile_base_name": "blog_backend"
  },
  "db_clients": [
    {
      "name": "default",
      "rdbms": "mysql",
      "host": "@MYSQL_IP@",
      "port": 3306,
      "dbname": "blog",
      "user": "root",
      "password": "123456",
      "is_fast": false,
      "connection_number": 5
    }
  ],
  "redis_clients": [
    {
      "name": "default",
      "host": "@REDIS_IP@",
      "port": 6379,
      "db": 0
    }
  ]
}
```


```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
  namespace: bloginfer
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
        - name: nginx
          image: nginx:1.25
          volumeMounts:
            - name: nginx-conf
              mountPath: /etc/nginx/nginx.conf
              subPath: nginx.conf
            - name: dist-storage
              mountPath: /usr/share/nginx/html
              subPath: dist
              readOnly: true
```

### 6. 数据库与缓存 Pod

* **MySQL Pod**：

  * 有状态服务
  * 通过 PV / PVC 绑定持久化存储
  * 确保 Pod 重建后的数据一致性

* **Redis Pod**：

  * In-Memory Key-Value 存储
  * 用于 Session 管理与高频访问数据缓存
  * 减轻后端服务对关系型数据库的访问压力

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mysql-secret
  namespace: bloginfer
type: Opaque
data:
  MYSQL_ROOT_PASSWORD: MTIzNDU2

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
  namespace: bloginfer
spec:
  replicas: 1
  serviceName: mysql
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
        - name: mysql
          image: mysql:9.2
          env:
            - name: MYSQL_ROOT_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: mysql-secret
                  key: MYSQL_ROOT_PASSWORD
          ports:
            - containerPort: 3306
          volumeMounts:
            - name: data
              mountPath: /var/lib/mysql
            - name: mysql-init
              mountPath: /docker-entrypoint-initdb.d
              readOnly: true
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "1"
              memory: "2Gi"
        - name: mysqld-exporter
          image: prom/mysqld-exporter:latest
          command: ["sh", "-c"]
          args:
            - |
              until mysqladmin ping -h 127.0.0.1 -u root -p$MYSQL_ROOT_PASSWORD; do
              sleep 2
              done
              /bin/mysqld_exporter
          env:
            - name: DATA_SOURCE_NAME
              value: "root:123456@(localhost:3306)/"
```

### 7. 训练守护进程 Pod（Training Daemon）

* 以 **长生命周期 Pod（或 Deployment）** 形式常驻运行
* 暴露简单的 HTTP 接口，用于接收“启动训练任务”的触发请求
* 接收到请求后，通过调用 Kubernetes API 或执行脚本，动态创建训练 Job/Pod，完成一次完整的模型训练流程
* 与模型存储目录挂载在同一个PVC，用于写入最新模型文件
* 内置定时任务（例如基于定时循环 + Shell/Python 脚本），定期扫描模型目录，只保留最近的若干个模型版本，自动清理更早的历史模型文件
* 通过这种方式，将“训练触发 + 历史模型清理”封装到统一的守护进程中，减少人工干预

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: train-daemon-sa
  namespace: bloginfer
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: train-daemon-role
  namespace: bloginfer
rules:
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "list", "watch", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: train-daemon-rb
  namespace: bloginfer
subjects:
  - kind: ServiceAccount
    name: train-daemon-sa
    namespace: bloginfer
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: train-daemon-role
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: train-daemon
  namespace: bloginfer
  labels:
    app: train-daemon
spec:
  replicas: 1
  selector:
    matchLabels:
      app: train-daemon
  template:
    metadata:
      labels:
        app: train-daemon
    spec:
      serviceAccountName: train-daemon-sa
      containers:
        - name: train-daemon
          image: my_repository/train-daemon:latest
          imagePullPolicy: IfNotPresent
          env:
            - name: NAMESPACE
              value: "bloginfer"
            - name: MODEL_DIR
              value: "/blog/models"
            - name: MIN_TRAIN_INTERVAL_SECONDS
              value: "600"
            - name: TRAIN_POLL_INTERVAL_SECONDS
              value: "5"
            - name: CLEAN_INTERVAL_SECONDS
              value: "3600"
            - name: MAX_CLIENT_MODEL_AGE_SECONDS
              value: "86400"
            - name: TRAIN_JOB_YAML
              value: "/blog/configs/train.yaml"
            - name: AGG_JOB_YAML
              value: "/blog/configs/Integrajob.yaml"
            - name: INFERENCE_DEPLOYMENT_NAME
              value: "inference"
            - name: PORT
              value: "8000"
```

---

## 三、联邦学习训练流程设计（自动化训练流水线）

模型训练采用典型的 **Federated Data-Parallel Training** 流程，并被设计为一条可重复、可自动执行的训练流水线：

1. Kubernetes 自动调度多个训练 Pod 作为 Batch Job 启动
2. 各训练 Pod 在本地数据集上独立完成模型训练
3. 每一轮训练结束后，训练 Pod 自动将梯度信息上传至参数聚合 Pod
4. 聚合 Pod 自动执行梯度平均（Gradient Averaging）并更新全局模型参数
5. 更新后的模型参数被自动分发回各训练 Pod，进入下一轮训练或结束训练任务

整个过程无需人工手动干预训练节点或参数同步，训练任务的启动、执行、结束均由 Kubernetes 与程序逻辑协同完成。

该自动化训练流程模拟了真实跨组织、跨节点的分布式协同训练场景，在不共享原始数据的前提下完成模型优化，同时体现了云计算环境中“任务驱动 + 自动调度”的核心思想。

```Python
base_params = None
base_vocab = None
base_weight = 0.0
global_ckpt_path = os.path.join(model_dir, "sentiment_zh_final.pt")
if os.path.exists(global_ckpt_path):
    try:
        global_ckpt = torch.load(global_ckpt_path, map_location="cpu")
        if isinstance(global_ckpt, dict) and "model" in global_ckpt:
            base_params = global_ckpt["model"]
            base_vocab = global_ckpt.get("vocab")
            base_weight = float(global_ckpt.get("total_weight", 0.0) or 0.0)
            print(f"已加载上一轮全局模型，累计权重 total_weight={base_weight}")
    except Exception as e:
        print(f"加载上一轮全局模型失败，将仅基于本轮客户端重新聚合: {e}")

if not param_file_paths and base_params is None:
    print("错误：未在 models 目录下找到任何以 client_ 开头的模型文件，且无已有全局模型。", flush=True)
    exit(1)

final_aggregated_params, shared_vocab, total_weight = federated_average_aggregate(
    param_file_paths, model_dir, base_params=base_params, base_vocab=base_vocab, base_weight=base_weight
)
```

```Python

model = SimpleSentiment(len(vocab))

if global_ckpt is not None:
    try:
        if isinstance(global_ckpt, dict) and "model" in global_ckpt:
            state = global_ckpt["model"]
        else:
            state = global_ckpt
        model.load_state_dict(state, strict=False)
        print("loaded global model state for continued training")
    except Exception as e:
        print(f"failed to load global model state: {e}")

extra_texts = load_comments_from_db(limit=5000)
extra_labels = []
if extra_texts:
    try:
        extra_labels = pseudo_label_comments(extra_texts, model, vocab)
        print(f"pseudo labeled {len(extra_labels)} comments from db")
    except Exception as e:
        print(f"failed to pseudo label comments: {e}")
        extra_texts = []
        extra_labels = []

if extra_texts and len(extra_texts) == len(extra_labels):
    merged_texts = list(train_texts) + list(extra_texts)
    merged_labels = list(train_labels) + list(extra_labels)
    train_ds = SentimentDataset(merged_texts, merged_labels, vocab)
    print(
        f"total train samples: labeled={len(train_texts)}, from_db={len(extra_texts)}, total={len(merged_texts)}"
    )



torch.save(
    {
        "model": model.state_dict(),
        "vocab": vocab,
        "num_samples": len(train_texts),
    },
    save_path,
)
```

---

## 四、模型推理与在线服务（自动部署与弹性运行）

训练完成的模型被自动导出并封装为 FastAPI 推理服务镜像，随后由 Kubernetes 负责部署与运行，对外提供在线预测能力。

推理服务在集群中具备如下自动化特性：

* 推理服务与训练流程完全解耦，模型更新后可自动替换服务版本
* 推理 Pod 属于无状态服务，可通过副本数调整实现自动横向扩展
* 当 Pod 异常终止或节点不可用时，Kubernetes 会自动拉起新的 Pod 以恢复服务

通过将推理过程纳入统一的编排与调度体系，系统实现了模型从“训练完成”到“对外服务”的自动过渡。

```Python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.infer = SentimentInferencer(
        ckpt_path="models/sentiment_zh_final.pt",
        device="cpu",
    )
    yield

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    infer: SentimentInferencer = app.state.infer
    label, prob = infer.predict_label(req.text, threshold=0.5)
    return PredictResponse(label=label, prob=prob)

```

---

## 五、博客系统与数据管理

博客系统作为模型能力的应用载体，完整运行在云端：

* 前端基于 Vue.js 构建，通过 Nginx 对外提供访问
* 后端基于 Drogon 框架实现高性能 C++ Web 服务
* 业务数据持久化存储于 MySQL 中（相关评论和用户信息）
* 用户登录状态缓存在 Redis 中（也就是讲token进行缓存）

该架构在保证数据一致性的同时，有效提升了系统响应性能。

```c++
client->execSqlAsync(
"SELECT c.id, c.post_id, c.user_id, c.content, c.created_at, u.username, u.avatar_url "
"FROM comments c LEFT JOIN users u ON c.user_id = u.id "
"WHERE c.post_id = ? ORDER BY c.created_at ASC",
[callback](const drogon::orm::Result &r) {
    Json::Value list(Json::arrayValue);
    for (const auto &row : r)
    {
        Json::Value item;
        item["id"] = Json::Int64(row["id"].as<long long>());
        item["post_id"] = Json::Int64(row["post_id"].as<long long>());
        item["user_id"] = Json::Int64(row["user_id"].as<long long>());
        item["content"] = row["content"].as<std::string>();
        if (!row["created_at"].isNull())
        {
            item["created_at"] = row["created_at"].as<std::string>();
        }
        if (!row["username"].isNull())
        {
            item["username"] = row["username"].as<std::string>();
        }
        if (!row["avatar_url"].isNull())
        {
            item["avatar_url"] = row["avatar_url"].as<std::string>();
        }
        list.append(item);
    }

    Json::Value body;
    body["code"] = 0;
    body["message"] = "ok";
    body["data"] = list;
    auto resp = HttpResponse::newHttpJsonResponse(body);
    resp->setStatusCode(k200OK);
    callback(resp);
},
[callback](const std::exception_ptr &eptr) {
    Json::Value body;
    body["code"] = 1;
    body["message"] = "Database error";
    if (eptr)
    {
        try
        {
            std::rethrow_exception(eptr);
        }
        catch (const std::exception &e)
        {
            body["detail"] = e.what();
        }
        catch (...)
        {
            body["detail"] = "unknown error";
        }
    }
    auto resp = HttpResponse::newHttpJsonResponse(body);
    resp->setStatusCode(k500InternalServerError);
    callback(resp);
},
postId)
```

---

## 六、用户行为驱动的模型迭代机制（自动数据闭环）

系统在博客评论功能中引入模型能力，构建了一条 **自动化的 Online Inference + Human Feedback 数据闭环**：

1. 用户发表评论后，评论内容被自动发送至推理服务进行情感分析
2. 系统调用大模型接口对评论进行情感与语义自动标注，生成弱监督标签（目前使用伪标签）
3. 标注后的数据被自动结构化存储，作为新增训练样本
4. 在后续训练周期中，新数据被自动纳入联邦学习训练流程

通过该机制，系统实现了用户行为、在线推理与离线训练之间的自动联动，使模型能够在云端环境中持续迭代与演化。


```Python
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
```

---

## 七、实验结果与系统验证

* Kubernetes 集群中多 Pod 协同运行效果
* 推理服务的水平扩展能力验证
* Drogon 后端镜像构建与自动部署结果
* 自动化构建与推送流程展示
* 模型训练过程中的损失变化
* 模型测试集上的分类性能评估

---

## 八、总结

本项目以云计算与大数据相关技术为核心，综合运用了容器化部署、微服务架构、联邦学习训练、在线推理与弹性伸缩机制，构建了一个具备工程复杂度与学术背景的完整云端系统，较为全面地体现了云计算环境下模型与应用协同运行的典型模式。
