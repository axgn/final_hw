#include <drogon/drogon.h>

int main() {
    // 优先从 config.json 加载配置（端口、日志、数据库等）
    drogon::app()
        .loadConfigFile("../config.json")
        // 如果是 Debug，则作为守护进程运行
    #ifdef DEBUG
        .enableRunAsDaemon()
    #endif
        .run();

    return 0;
}
