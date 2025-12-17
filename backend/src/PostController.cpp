#include "PostController.h"
#include <drogon/drogon.h>
#include <drogon/orm/DbClient.h>
#include <drogon/nosql/RedisClient.h>
#include <json/json.h>

using namespace drogon;
using namespace drogon::orm;

namespace {
// 简单的工具函数：从请求中解析 JSON，失败则返回 nullptr
std::shared_ptr<Json::Value> parseJsonBody(const HttpRequestPtr &req) {
    auto json = req->getJsonObject();
    if (!json) {
        return nullptr;
    }
    return json;
}

void sendBadRequest(const std::function<void (const HttpResponsePtr &)> &callback,
                    const std::string &message) {
    Json::Value err;
    err["success"] = false;
    err["message"] = message;
    auto resp = HttpResponse::newHttpJsonResponse(err);
    resp->setStatusCode(k400BadRequest);
    callback(resp);
}

void sendServerError(const std::function<void (const HttpResponsePtr &)> &callback,
                     const std::string &message) {
    Json::Value err;
    err["success"] = false;
    err["message"] = message;
    auto resp = HttpResponse::newHttpJsonResponse(err);
    resp->setStatusCode(k500InternalServerError);
    callback(resp);
}

void incrRedisCounter(const std::string &key) {
    try {
        auto redisClient = app().getFastRedisClient();
        if (redisClient) {
            redisClient->execCommandAsync(
                [](const drogon::nosql::RedisResult &) {
                    // ignore result
                },
                [](const std::exception &) {
                    // ignore redis errors for now
                },
                "INCR %s", key.c_str());
        }
    } catch (...) {
        // 如果没有配置 redis 或运行失败，忽略计数错误
    }
}
}

void PostController::getPosts(const HttpRequestPtr &req,
                              std::function<void (const HttpResponsePtr &)> &&callback) {
    incrRedisCounter("views:/api/posts");

    Json::Value posts(Json::arrayValue);

    Json::Value p1;
    p1["id"] = 1;
    p1["title"] = "First post from Drogon backend";
    p1["summary"] = "This is a sample post served by C++ Drogon";
    posts.append(p1);

    Json::Value p2;
    p2["id"] = 2;
    p2["title"] = "Second post";
    p2["summary"] = "Another sample post";
    posts.append(p2);

    auto resp = HttpResponse::newHttpJsonResponse(posts);
    callback(resp);
}

void PostController::getPostById(const HttpRequestPtr &req,
                                 std::function<void (const HttpResponsePtr &)> &&callback,
                                 int postId) {
    incrRedisCounter("views:/api/posts/" + std::to_string(postId));

    Json::Value post;
    post["id"] = postId;
    post["title"] = "Post #" + std::to_string(postId);
    post["content"] = "This is the detail content for post #" + std::to_string(postId);

    auto resp = HttpResponse::newHttpJsonResponse(post);
    callback(resp);
}

void PostController::login(const HttpRequestPtr &req,
                           std::function<void (const HttpResponsePtr &)> &&callback) {
    incrRedisCounter("views:/api/login");

    auto json = parseJsonBody(req);
    if (!json) {
        sendBadRequest(callback, "JSON body required");
        return;
    }

    if (!(*json).isMember("username") || !(*json)["username"].isString() ||
        !(*json).isMember("password") || !(*json)["password"].isString()) {
        sendBadRequest(callback, "username and password are required");
        return;
    }

    std::string username = (*json)["username"].asString();
    std::string password = (*json)["password"].asString();

    auto dbClient = app().getDbClient();
    dbClient->execSqlAsync(
        "SELECT id, password FROM users WHERE username = ?",
        [callback, password](const Result &r) {
            Json::Value respJson;
            if (r.size() == 0) {
                respJson["success"] = false;
                respJson["message"] = "Invalid username or password";
            } else {
                const auto &row = r[0];
                auto dbPassword = row["password"].as<std::string>();
                if (dbPassword == password) {
                    respJson["success"] = true;
                    respJson["userId"] = static_cast<Json::Int64>(row["id"].as<long long>());
                } else {
                    respJson["success"] = false;
                    respJson["message"] = "Invalid username or password";
                }
            }
            auto resp = HttpResponse::newHttpJsonResponse(respJson);
            callback(resp);
        },
        [callback](const DrogonDbException &e) {
            (void)e;
            sendServerError(callback, "Database error in login");
        },
        username);
}

void PostController::addComment(const HttpRequestPtr &req,
                                std::function<void (const HttpResponsePtr &)> &&callback,
                                int postId) {
    incrRedisCounter("comments:add:" + std::to_string(postId));

    auto json = parseJsonBody(req);
    if (!json) {
        sendBadRequest(callback, "JSON body required");
        return;
    }

    if (!(*json).isMember("userId") || !(*json)["userId"].isInt64() ||
        !(*json).isMember("content") || !(*json)["content"].isString()) {
        sendBadRequest(callback, "userId (int) and content (string) are required");
        return;
    }

    long long userId = (*json)["userId"].asInt64();
    std::string content = (*json)["content"].asString();

    auto dbClient = app().getDbClient();
    dbClient->execSqlAsync(
        "INSERT INTO comments(post_id, user_id, content) VALUES(?, ?, ?)",
        [callback](const Result &r) {
            (void)r;
            Json::Value respJson;
            respJson["success"] = true;
            auto resp = HttpResponse::newHttpJsonResponse(respJson);
            callback(resp);
        },
        [callback](const DrogonDbException &e) {
            (void)e;
            sendServerError(callback, "Database error when adding comment");
        },
        postId,
        userId,
        content);
}

void PostController::getComments(const HttpRequestPtr &req,
                                 std::function<void (const HttpResponsePtr &)> &&callback,
                                 int postId) {
    incrRedisCounter("comments:list:" + std::to_string(postId));

    auto dbClient = app().getDbClient();
    dbClient->execSqlAsync(
        "SELECT c.id, c.content, c.created_at, u.username "
        "FROM comments c JOIN users u ON c.user_id = u.id "
        "WHERE c.post_id = ? ORDER BY c.created_at DESC",
        [callback](const Result &r) {
            Json::Value arr(Json::arrayValue);
            for (const auto &row : r) {
                Json::Value item;
                item["id"] = static_cast<Json::Int64>(row["id"].as<long long>());
                item["content"] = row["content"].as<std::string>();
                item["username"] = row["username"].as<std::string>();
                item["created_at"] = row["created_at"].as<std::string>();
                arr.append(item);
            }
            auto resp = HttpResponse::newHttpJsonResponse(arr);
            callback(resp);
        },
        [callback](const DrogonDbException &e) {
            (void)e;
            sendServerError(callback, "Database error when fetching comments");
        },
        postId);
}
