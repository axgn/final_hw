#pragma once

#include <drogon/HttpController.h>

using namespace drogon;

class CommentCtrl : public drogon::HttpController<CommentCtrl>
{
  public:
    METHOD_LIST_BEGIN
    // 创建评论，需要登录
    ADD_METHOD_TO(CommentCtrl::createComment, "/api/comments", drogon::Post, "AuthFilter");
    // 按帖子获取评论列表（公开）
    ADD_METHOD_TO(CommentCtrl::listByPost, "/api/comments", drogon::Get);
    // 删除自己的评论，需要登录
    ADD_METHOD_TO(CommentCtrl::deleteComment, "/api/comments/{1}", drogon::Delete, "AuthFilter");
    METHOD_LIST_END

    void createComment(const HttpRequestPtr &req,
                       std::function<void(const HttpResponsePtr &)> &&callback);

    void listByPost(const HttpRequestPtr &req,
                    std::function<void(const HttpResponsePtr &)> &&callback);

    void deleteComment(const HttpRequestPtr &req,
                       std::function<void(const HttpResponsePtr &)> &&callback,
                       long long commentId);
};
