#pragma once

#include <drogon/HttpController.h>

class PostController : public drogon::HttpController<PostController> {
public:
    METHOD_LIST_BEGIN
    ADD_METHOD_TO(PostController::getPosts, "/api/posts", drogon::Get);
    ADD_METHOD_TO(PostController::getPostById, "/api/posts/{1}", drogon::Get);
    ADD_METHOD_TO(PostController::login, "/api/login", drogon::Post);
    ADD_METHOD_TO(PostController::addComment, "/api/posts/{1}/comments", drogon::Post);
    ADD_METHOD_TO(PostController::getComments, "/api/posts/{1}/comments", drogon::Get);
    METHOD_LIST_END

    void getPosts(const drogon::HttpRequestPtr &req,
                  std::function<void (const drogon::HttpResponsePtr &)> &&callback);

    void getPostById(const drogon::HttpRequestPtr &req,
                     std::function<void (const drogon::HttpResponsePtr &)> &&callback,
                     int postId);

    void login(const drogon::HttpRequestPtr &req,
               std::function<void (const drogon::HttpResponsePtr &)> &&callback);

    void addComment(const drogon::HttpRequestPtr &req,
                    std::function<void (const drogon::HttpResponsePtr &)> &&callback,
                    int postId);

    void getComments(const drogon::HttpRequestPtr &req,
                     std::function<void (const drogon::HttpResponsePtr &)> &&callback,
                     int postId);
};
