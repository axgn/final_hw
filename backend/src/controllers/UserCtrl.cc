#include "UserCtrl.h"
#include <drogon/drogon.h>
#include <random>
#include <sstream>
#include <iomanip>

using namespace drogon;

namespace
{
    std::string generateToken()
    {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<unsigned long long> dis;

        std::ostringstream oss;
        for (int i = 0; i < 4; ++i)
        {
            oss << std::hex << std::setw(16) << std::setfill('0') << dis(gen);
        }
        return oss.str();
    }

    std::string generateSalt()
    {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<unsigned long long> dis;

        std::ostringstream oss;
        for (int i = 0; i < 2; ++i)
        {
            oss << std::hex << std::setw(16) << std::setfill('0') << dis(gen);
        }
        return oss.str();
    }
}

// 注册接口：POST /api/user/register
void UserCtrl::registerUser(const HttpRequestPtr &req, std::function<void(const HttpResponsePtr &)> &&callback)
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
    if (!json.isMember("username") || !json.isMember("password"))
    {
        Json::Value body;
        body["code"] = 1;
        body["message"] = "username and password are required";
        auto resp = HttpResponse::newHttpJsonResponse(body);
        resp->setStatusCode(k400BadRequest);
        callback(resp);
        return;
    }

    const auto username = json["username"].asString();
    const auto password = json["password"].asString();
    const auto email = json.isMember("email") ? json["email"].asString() : std::string();
    const auto salt = generateSalt();

    auto client = app().getDbClient();

    // 先检查用户名是否已存在
    client->execSqlAsync(
        "SELECT id FROM users WHERE username = ?",
        [callback, username, password, email, client, salt](const drogon::orm::Result &r) {
            if (r.size() > 0)
            {
                Json::Value body;
                body["code"] = 1;
                body["message"] = "Username already exists";
                auto resp = HttpResponse::newHttpJsonResponse(body);
                resp->setStatusCode(k400BadRequest);
                callback(resp);
                return;
            }

            // 不存在则插入新用户，使用 salt + SHA2 哈希保存密码
            client->execSqlAsync(
                "INSERT INTO users (username, password, salt, email) VALUES (?, SHA2(CONCAT(?, ?), 256), ?, ?)",
                [callback, username, email](const drogon::orm::Result &) {
                    Json::Value data;
                    data["username"] = username;
                    if (!email.empty())
                    {
                        data["email"] = email;
                    }

                    Json::Value body;
                    body["code"] = 0;
                    body["message"] = "ok";
                    body["data"] = data;
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
                username,
                password,
                salt,
                salt,
                email);
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
        username);
}

// 登录接口：POST /api/user/login
void UserCtrl::login(const HttpRequestPtr &req, std::function<void(const HttpResponsePtr &)> &&callback)
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
    if (!json.isMember("username") || !json.isMember("password"))
    {
        Json::Value body;
        body["code"] = 1;
        body["message"] = "username and password are required";
        auto resp = HttpResponse::newHttpJsonResponse(body);
        resp->setStatusCode(k400BadRequest);
        callback(resp);
        return;
    }

    const auto username = json["username"].asString();
    const auto password = json["password"].asString();

    auto client = app().getDbClient();
    client->execSqlAsync(
        "SELECT id, username, email, avatar_url FROM users WHERE username = ? AND password = SHA2(CONCAT(?, salt), 256)",
        [callback](const drogon::orm::Result &r) {
            if (r.size() == 0)
            {
                Json::Value body;
                body["code"] = 1;
                body["message"] = "Invalid username or password";
                auto resp = HttpResponse::newHttpJsonResponse(body);
                resp->setStatusCode(k401Unauthorized);
                callback(resp);
                return;
            }

            const auto &row = r[0];

            auto userId = row["id"].as<long long>();

            // 生成登录 token 并写入 Redis，过期时间 1 小时
            std::string token = generateToken();
            auto redisClient = app().getRedisClient();
            const std::string key = "session:" + token;
            const std::string userIdStr = std::to_string(userId);
            const int expireSeconds = 3600;

            redisClient->execCommandAsync(
                [](const nosql::RedisResult &)
                {
                    // ignore success
                },
                [](const std::exception &e)
                {
                    LOG_ERROR << "Redis SETEX failed: " << e.what();
                },
                "SETEX %s %d %s", key.c_str(), expireSeconds, userIdStr.c_str());

            Json::Value data;
            data["id"] = Json::Int64(userId);
            data["username"] = row["username"].as<std::string>();
            if (!row["email"].isNull())
            {
                data["email"] = row["email"].as<std::string>();
            }
            if (!row["avatar_url"].isNull())
            {
                data["avatar_url"] = row["avatar_url"].as<std::string>();
            }

            Json::Value body;
            body["code"] = 0;
            body["message"] = "ok";
            body["token"] = token;
            body["data"] = data;
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
        username,
        password);
}

// 需要鉴权的接口：GET /api/user/profile
void UserCtrl::profile(const HttpRequestPtr &req, std::function<void(const HttpResponsePtr &)> &&callback)
{
    // AuthFilter 已经校验 token 并在属性中写入 userId
    const auto userIdStr = req->getAttributes()->get<std::string>("userId");
    const auto userId = std::stoll(userIdStr);

    auto client = app().getDbClient();
    client->execSqlAsync(
        "SELECT id, username, email, avatar_url FROM users WHERE id = ?",
        [callback](const drogon::orm::Result &r) {
            if (r.size() == 0)
            {
                Json::Value body;
                body["code"] = 1;
                body["message"] = "User not found";
                auto resp = HttpResponse::newHttpJsonResponse(body);
                resp->setStatusCode(k404NotFound);
                callback(resp);
                return;
            }

            const auto &row = r[0];
            Json::Value data;
            data["id"] = Json::Int64(row["id"].as<long long>());
            data["username"] = row["username"].as<std::string>();
            if (!row["email"].isNull())
            {
                data["email"] = row["email"].as<std::string>();
            }
            if (!row["avatar_url"].isNull())
            {
                data["avatar_url"] = row["avatar_url"].as<std::string>();
            }

            Json::Value body;
            body["code"] = 0;
            body["message"] = "ok";
            body["data"] = data;
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
        userId);
}
