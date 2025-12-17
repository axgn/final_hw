# Blog Backend (C++ Drogon)

本目录是你的 Vue 博客项目的 C++ Drogon REST 后端示例，使用 CMake 作为构建工具。

## 依赖

- CMake >= 3.10
- C++17 编译器（g++ / clang++）
- Drogon 框架（需在系统中已安装，例如通过包管理器或源码安装）

> 安装 Drogon 的方式会因系统而异，一般流程可参考官方文档：https://github.com/drogonframework/drogon

## 构建

在 `blog_website/backend` 目录下执行：

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

成功后会生成可执行文件 `blog_backend`。

## 运行

在 `build` 目录：

```bash
./blog_backend
```

服务会读取 `../config.json` 中的监听端口、MySQL 和 Redis 配置，默认监听 `http://0.0.0.0:8080`。

## 数据库和 Redis

1. 在 MySQL 中创建数据库并导入表结构：

```sql
CREATE DATABASE blog CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE blog;
SOURCE ../sql/schema.sql;
```

2. 根据你的实际 MySQL/Redis 地址和账号，修改 `backend/config.json` 中的 `db_clients` 和 `redis_clients` 配置。

3. 手动插入一条测试用户记录，例如：

```sql
INSERT INTO users(username, password) VALUES('test', '123456');
```

> 这里只是作业示例，密码未做加密处理，真实项目请使用密码哈希。

## 接口说明

### 文章

- `GET /api/posts` 返回示例文章列表，并在 Redis 中自增键 `views:/api/posts`
- `GET /api/posts/{id}` 返回指定 ID 的示例文章详情，并在 Redis 中自增键 `views:/api/posts/{id}`

### 用户登录

- `POST /api/login`

请求体 JSON：

```json
{
  "username": "test",
  "password": "123456"
}
```

返回示例：

```json
{
  "success": true,
  "userId": 1
}
```

访问该接口会在 Redis 中自增键 `views:/api/login`。

### 评论

- `POST /api/posts/{id}/comments` 添加评论

请求体 JSON：

```json
{
  "userId": 1,
  "content": "这是一个评论"
}
```

成功时返回：

```json
{
  "success": true
}
```

同时在 Redis 中自增键 `comments:add:{id}`。

- `GET /api/posts/{id}/comments` 获取某篇文章的评论列表

返回示例：

```json
[
  {
    "id": 1,
    "content": "这是一个评论",
    "username": "test",
    "created_at": "2025-12-15 12:00:00"
  }
]
```

访问该接口会在 Redis 中自增键 `comments:list:{id}`。
