#pragma once

#include <drogon/HttpController.h>

using namespace drogon;

class UserCtrl : public drogon::HttpController<UserCtrl>
{
  public:
    METHOD_LIST_BEGIN
    // 注册
    ADD_METHOD_TO(UserCtrl::registerUser, "/api/user/register", drogon::Post);
    // 登录
    ADD_METHOD_TO(UserCtrl::login, "/api/user/login", drogon::Post);
    // 需要登录的接口示例：获取当前用户信息（挂载 AuthFilter 中间件）
    ADD_METHOD_TO(UserCtrl::profile, "/api/user/profile", drogon::Get, "AuthFilter");
    METHOD_LIST_END

    void registerUser(const HttpRequestPtr &req,
                      std::function<void(const HttpResponsePtr &)> &&callback);

    void login(const HttpRequestPtr &req,
               std::function<void(const HttpResponsePtr &)> &&callback);

    void profile(const HttpRequestPtr &req,
                 std::function<void(const HttpResponsePtr &)> &&callback);
};
