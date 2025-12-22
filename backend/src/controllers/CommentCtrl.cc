#include "CommentCtrl.h"
#include <drogon/drogon.h>
#include <cstdlib>

using namespace drogon;

// 创建评论：POST /api/comments
// 请求体示例：{ "post_id": 123, "content": "xxx" }
void CommentCtrl::createComment(const HttpRequestPtr &req,
                                std::function<void(const HttpResponsePtr &)> &&callback)
{
    auto jsonPtr = req->getJsonObject();
    if (!jsonPtr)
    {
        Json::Value body;
        body["code"] = 1;
        body["message"] = "Invalid JSON body";
        auto resp = HttpResponse::newHttpJsonResponse(body);
        resp->setStatusCode(k400BadRequest);
        callback(resp);
        return;
    }

    const auto &json = *jsonPtr;
    if (!json.isMember("post_id") || !json.isMember("content"))
    {
        Json::Value body;
        body["code"] = 1;
        body["message"] = "post_id and content are required";
        auto resp = HttpResponse::newHttpJsonResponse(body);
        resp->setStatusCode(k400BadRequest);
        callback(resp);
        return;
    }

    long long postId = 0;
    try
    {
        postId = std::stoll(json["post_id"].asString());
    }
    catch (...)
    {
        Json::Value body;
        body["code"] = 1;
        body["message"] = "invalid post_id";
        auto resp = HttpResponse::newHttpJsonResponse(body);
        resp->setStatusCode(k400BadRequest);
        callback(resp);
        return;
    }

    const auto content = json["content"].asString();
    if (content.empty())
    {
        Json::Value body;
        body["code"] = 1;
        body["message"] = "content cannot be empty";
        auto resp = HttpResponse::newHttpJsonResponse(body);
        resp->setStatusCode(k400BadRequest);
        callback(resp);
        return;
    }

    const auto userIdStr = req->getAttributes()->get<std::string>("userId");
    const auto userId = std::stoll(userIdStr);

    auto client = app().getDbClient();
    client->execSqlAsync(
        "INSERT INTO comments (post_id, user_id, content) VALUES (?, ?, ?)",
        [callback, postId, userId, content](const drogon::orm::Result &r) {
            Json::Value data;
            data["post_id"] = Json::Int64(postId);
            data["user_id"] = Json::Int64(userId);
            data["content"] = content;

            Json::Value body;
            body["code"] = 0;
            body["message"] = "ok";
            body["data"] = data;
            auto resp = HttpResponse::newHttpJsonResponse(body);
            resp->setStatusCode(k200OK);
            callback(resp);

            // 异步触发训练守护进程，不阻塞当前请求
            try
            {
                const char *daemonUrlEnv = std::getenv("TRAIN_DAEMON_URL");
                std::string daemonUrl = daemonUrlEnv ? daemonUrlEnv : "http://train-daemon-service:8000";

                auto httpClient = drogon::HttpClient::newHttpClient(daemonUrl);
                auto trainReq = drogon::HttpRequest::newHttpRequest();
                trainReq->setMethod(drogon::Post);
                trainReq->setPath("/train/trigger");

                httpClient->sendRequest(
                    trainReq,
                    [](ReqResult result, const HttpResponsePtr &resp) {
                        if (result != ReqResult::Ok)
                        {
                            LOG_WARN << "Failed to trigger train daemon: result=" << static_cast<int>(result);
                        }
                    });
            }
            catch (const std::exception &e)
            {
                LOG_WARN << "Exception while triggering train daemon: " << e.what();
            }
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
        postId,
        userId,
        content);
}

// 获取某个帖子的评论列表：GET /api/comments?post_id=123
void CommentCtrl::listByPost(const HttpRequestPtr &req,
                             std::function<void(const HttpResponsePtr &)> &&callback)
{
    const auto postIdStr = req->getParameter("post_id");
    if (postIdStr.empty())
    {
        Json::Value body;
        body["code"] = 1;
        body["message"] = "post_id is required";
        auto resp = HttpResponse::newHttpJsonResponse(body);
        resp->setStatusCode(k400BadRequest);
        callback(resp);
        return;
    }

    long long postId = 0;
    try
    {
        postId = std::stoll(postIdStr);
    }
    catch (...)
    {
        Json::Value body;
        body["code"] = 1;
        body["message"] = "invalid post_id";
        auto resp = HttpResponse::newHttpJsonResponse(body);
        resp->setStatusCode(k400BadRequest);
        callback(resp);
        return;
    }

    auto client = app().getDbClient();
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
        postId);
}

// 删除评论：DELETE /api/comments/{id}，只能删除自己的
void CommentCtrl::deleteComment(const HttpRequestPtr &req,
                                std::function<void(const HttpResponsePtr &)> &&callback,
                                long long commentId)
{
    const auto userIdStr = req->getAttributes()->get<std::string>("userId");
    const auto userId = std::stoll(userIdStr);

    auto client = app().getDbClient();
    client->execSqlAsync(
        "DELETE FROM comments WHERE id = ? AND user_id = ?",
        [callback](const drogon::orm::Result &r) {
            Json::Value body;
            if (r.affectedRows() == 0)
            {
                body["code"] = 1;
                body["message"] = "Comment not found or no permission";
                auto resp = HttpResponse::newHttpJsonResponse(body);
                resp->setStatusCode(k404NotFound);
                callback(resp);
                return;
            }

            body["code"] = 0;
            body["message"] = "ok";
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
        commentId,
        userId);
}
