cmake_minimum_required(VERSION 3.10)
project(blog_backend CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define DEBUG macro when building in Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
endif()

find_package(Drogon REQUIRED)

add_executable(blog_backend
    src/main.cpp
    src/PostController.cpp
)

target_include_directories(blog_backend PRIVATE include)

target_link_libraries(blog_backend PRIVATE Drogon::Drogon)
