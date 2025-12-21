#include <drogon/drogon.h>

int main() {
    // 优先从 config.json 加载配置（端口、日志、数据库等）
    drogon::app()
        .loadConfigFile("config.json")
        .run();

    return 0;
}
