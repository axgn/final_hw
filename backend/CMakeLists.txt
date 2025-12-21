cmake_minimum_required(VERSION 3.10)
project(blog_backend CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define DEBUG macro when building in Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
endif()

# Read MySQL and Redis IPs from environment and generate config.json
set(MYSQL_IP $ENV{MYSQL_HOST})
if(NOT MYSQL_IP)
    set(MYSQL_IP "mysql")
endif()

set(REDIS_IP $ENV{REDIS_HOST})
if(NOT REDIS_IP)
    set(REDIS_IP "redis")
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config.json.in
    ${CMAKE_CURRENT_BINARY_DIR}/config.json
    @ONLY
)

find_package(Drogon REQUIRED)

add_executable(blog_backend
    src/main.cpp
    src/controllers/UserCtrl.cc
    src/controllers/CommentCtrl.cc
    src/AuthFilter.cc
)

target_include_directories(blog_backend PRIVATE include)

target_link_libraries(blog_backend PRIVATE Drogon::Drogon)
