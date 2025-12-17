# Vue Markdown 博客

这是一个使用 Vue 3 + Vite 搭建的简单博客示例，支持渲染用 Markdown 编写的文章。

## 运行方式

在项目根目录（`vue-blog`）下执行：

```bash
npm install
npm run dev
```

然后在浏览器中打开终端输出的本地地址（通常是 `http://localhost:5173`）。

## 调用 FastAPI 情感分析接口

后端 FastAPI 服务在 `inference/infer.py` 中定义，提供接口：

- `POST /predict`，请求体：`{"text": "这家店真的很好，强烈推荐！"}`

前端在 `App.vue` 中新增了一个情感分析区域，通过浏览器直接调用该接口。

### 启动后端

在仓库根目录下执行：

```bash
cd inference
python3 infer.py
```

默认在 `http://localhost:9000` 启动 FastAPI 服务。

### 配置前端访问地址（可选）

前端默认使用 `http://localhost:9000` 作为后端地址，你也可以通过环境变量自定义：

```bash
export VITE_API_BASE_URL="http://localhost:9000"
npm run dev
```

启动后，在浏览器中打开页面，在底部的“情感分析（调用 FastAPI 接口）”区域输入中文句子并点击“开始分析”，即可看到后端返回的情感标签和置信度。

## 结构说明

- `src/App.vue`：页面布局和文章列表
- `src/components/MarkdownViewer.vue`：Markdown 渲染组件
- `src/posts/index.js`：示例文章数据（Markdown 字符串）

你可以在 `src/posts/index.js` 中添加或修改文章内容。
