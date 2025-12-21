#include "AuthFilter.h"
#include <drogon/drogon.h>
#include <memory>

using namespace drogon;

void AuthFilter::doFilter(const HttpRequestPtr &req,
                          FilterCallback &&fcb,
                          FilterChainCallback &&fccb)
{
    auto fcbPtr = std::make_shared<FilterCallback>(std::move(fcb));
    auto fccbPtr = std::make_shared<FilterChainCallback>(std::move(fccb));

    const auto authHeader = req->getHeader("authorization");
    if (authHeader.empty())
    {
        Json::Value body;
        body["code"] = 1;
        body["message"] = "Unauthorized: missing token";
        auto resp = HttpResponse::newHttpJsonResponse(body);
        resp->setStatusCode(k401Unauthorized);
        (*fcbPtr)(resp);
        return;
    }

    std::string token = authHeader;
    const std::string bearerPrefix = "Bearer ";
    if (authHeader.size() > bearerPrefix.size() &&
        authHeader.compare(0, bearerPrefix.size(), bearerPrefix) == 0)
    {
        token = authHeader.substr(bearerPrefix.size());
    }

    if (token.empty())
    {
        Json::Value body;
        body["code"] = 1;
        body["message"] = "Unauthorized: empty token";
        auto resp = HttpResponse::newHttpJsonResponse(body);
        resp->setStatusCode(k401Unauthorized);
        (*fcbPtr)(resp);
        return;
    }

    auto client = app().getRedisClient();
    const std::string key = "session:" + token;

    client->execCommandAsync(
        [req, fccbPtr, fcbPtr](const nosql::RedisResult &r) {
            if (r.isNil())
            {
                Json::Value body;
                body["code"] = 1;
                body["message"] = "Unauthorized: invalid token";
                auto resp = HttpResponse::newHttpJsonResponse(body);
                resp->setStatusCode(k401Unauthorized);
                (*fcbPtr)(resp);
                return;
            }

            const auto userIdStr = r.asString();
            req->getAttributes()->insert("userId", userIdStr);

            (*fccbPtr)();
        },
        [fcbPtr](const std::exception &e) {
            Json::Value body;
            body["code"] = 1;
            body["message"] = "Redis error";
            body["detail"] = e.what();
            auto resp = HttpResponse::newHttpJsonResponse(body);
            resp->setStatusCode(k500InternalServerError);
            (*fcbPtr)(resp);
        },
        "GET %s", key.c_str());
}
