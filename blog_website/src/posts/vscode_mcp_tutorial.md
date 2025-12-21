# 有关vscode的mcp教程

在这篇文章里，我会先介绍一下mcp(Model Context Protocol)，并说一下我对mcp的理解，然后我会讲一下，如何使用在vscode中使用mcp，最后，我也会尝试着实现一个简单的mcp server。
这里的mcp server的使用都是基于github copilot的agent模式的（ask模式不支持mcp），其他拓展我暂时还没有尝试过。学生可以通过github student pack免费使用github copilot。至于如何申请github student pack，可以参考这个博主写的[Github学生认证及学生包保姆级申请指南](https://zhuanlan.zhihu.com/p/578964972)，我后面看情况也可能会再写一篇，记住认证的时候一定要有2fa，不然认证不过。


## MCP server介绍

首先，MCP，全称 Model Context Protocol，是给大语言模型与应用程序服务进行通信设计的一种协议。它可以使得大语言模型可以实现与应用程序进行交互的功能。就比如使用搜索引擎进行搜索，查询天气，播放音乐等。

然后，要实现这样一个协议，通常需要实现两部分，一部分是mcp server，还有一部分是mcp client。mcp client现在主流的大语言模型客户端都会直接集成，所以不需要我们自己实现，而服务器端就是mcp可以为我们带来多种功能的关键，而我们要实现我们自己所想要的额外功能的话，就需要自己实现一个mcp server。下面是我在官网上找的示意图：
![mcp 示意图](vscode_mcp_tutorial/image-2.png)
每一个mcp client和mcp server 是一对一的连接关系。

## 如何使用在vscode中使用mcp

### 第一种方式

在最新的vscode中，可以直接在拓展界面直接看到自己安装的mcp server，
![拓展界面](vscode_mcp_tutorial/image.png)
最开始是什么都没有的，你可以通过Browse MCP Servers跳转到浏览器界面(在最新版(2025.9)的更新中可以直接在vscode内浏览mcp server，不需要跳转到浏览器界面，但是需要在设置里打开chat.mcp.gallery.enabled选项)，然后就可以选择你想要的mcp server进行安装了，然后这里的mcp server其实本身的配置内容是存储在配置文件中的，你可以在C:\Users\aaa1\AppData\Roaming\Code\User（如果你是在windows使用vscode，且全局安装，如果不是的话，你也可以打开settings.json的目录，两者就在统一个目录下面）目录下看到一个mcp.json的文件，这里就存储了你刚刚在浏览器界面安装的mcp server的配置，当你安装mcp server的时候实际上就是在servers的值里添加了一段mcp server名称与相应配置的键值对。你把相应的内容删除了，就相当于卸载了这个mcp server，拓展界面的mcp server也会消失，所以说安装mcp server也有第二种方式。

### 第二种方式

直接在mcp.json文件中添加相应的mcp server配置就可以了，但是要理解mcp.json的内容，就需要对mcp通信协议的内容有一定的理解，所以我会再详细地讲一下mcp通信协议的内容。在mcp的规定中，mcp server和client的通信有两种方式，一种是通过sse（基于http协议），另一种是通过stdio（标准输入输出）的方式。所以参数里的"type"参数就有两种选择，分别是http和stdio。然后mcp server的配置内容里还会有一些其他的参数，比如说url，command，args，env等，根据你开始时选择的type不同，这里的参数也会不同。如果说，type是http的话，比如说url参数就是用来配置mcp server的地址的。type是stdio的话，command参数就是用来配置mcp server的启动命令的，args参数就是用来配置mcp server的启动参数的，env参数就是用来配置mcp server的环境变量的。还有更详细的参数介绍，可以参考[github mcp的官方文档](https://code.visualstudio.com/docs/copilot/customization/mcp-servers?wt.md_id=AZ-MVP-5004796#_standard-io-stdio-servers)

### 第三种方式

这第三种方式实际上和第二种方式本质上是一样的，但是我们是通过vscode command palette来实现的。打开command palette，然后输入MCP，就可以看到相关的命令了，选择Add MCP Server，然后根据提示进行输入就可以了，添加的内容也是直接添加到mcp.json文件中的，这样我们就不需要自己去修改mcp.json文件了（也不用特意去记相关的参数）。有关mcp的其他命令也可以自己尝试一下。一般来说，我更习惯使用这种方式添加mcp server。因为第一种只能安装官方提供的mcp server，而第二种方式需要自己去记住mcp.json的键名，第三种方式则是通过命令的方式一步步进行添加，比较方便。通过命令面板，我们也可以看到有哪些具体的方式:
![主要方式](vscode_mcp_tutorial/image-1.png)
command方式就是直接填启动的命令就可以了，http就是填url。npm，pip，docker image这三个就是填入包名后，就会下载包或者拉取镜像，然后mcp.json中会自动添加相关的启动命令，如果是command方式的话，往往需要自己去下载包，这里推荐使用uv进行下载。uv是最新的python环境管理工具，可以更好地管理python环境，以及相关的包，在mcp的官方教程中也是使用改工具去管理环境的。
uv在macos/linux下的安装方式:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

在windows下的安装方式:

```sh
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

补充一下，不要打开太多的mcp server，在mcp server 太多的时候，模型对功能的识别可能不是很准确，无法正确的去调用相应的tool。

## 如何实现一个简单的mcp server

最后我们讲一下如何实现一个简单的mcp server。这里我会使用python来实现一个简单的mcp server，它的功能是获取一些当前的系统信息，当然你也可以使用其他语言来实现。官方提供了多个语言的sdk。

这里先使用uv来初始化我们的工作目录：

```bash
# 初始化uv工作目录
uv init mcp_system_info
cd mcp_system_info
# 创建虚拟环境
uv venv
source .venv/bin/activate
# 添加相关依赖
uv add "mcp[cli]" psutil
```

然后我们将下面的代码替换掉main.py的内容：

```python
from mcp.server.fastmcp import FastMCP
import psutil

# 初始化 FastMCP server
mcp = FastMCP("cpu and memory info server")

def get_system_info() -> str:
    """获取系统的 CPU 和内存使用情况。"""
    cpu_usage = psutil.cpu_percent(interval=1)
    cpu_logical_count = psutil.cpu_count(logical=True)
    cpu_physical_count = psutil.cpu_count(logical=False)
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    return f"CPU 使用率: {cpu_usage}%\n内存使用率: {memory_usage}%\nCPU 逻辑核心数: {cpu_logical_count}\nCPU 物理核心数: {cpu_physical_count}"

# 定义 MCP 工具函数
@mcp.tool()
def cpu_memory_info() -> str:
    """获取系统的 CPU 和内存使用情况。"""
    return get_system_info()


if __name__ == "__main__":
    # 初始化并运行 server
    mcp.run(transport='stdio')

```

然后我们就成功实现了一个简单的mcp server，接下来我们就可以在vs code中测试一下这个mcp server：
通过vscode打开mcp_system_info目录，将以下代码添加到目录下的.vscode目录下的mcp.json中：

```json
{
  "servers": {
    "system_info": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "main.py"]
    }
  },
  "inputs": []
}

```

然后我们就可以在copilot的聊天界面中使用这个mcp server了，直接输入"获取cpu和内存信息"就可以了，效果如下：
![效果图](vscode_mcp_tutorial/image-3.png)

在这里可以看到很多的mcp 项目，如果有相关需要可以在上面找，也可以学习更多的mcp server实现方式。
[mcp server 收集项目](https://github.com/punkpeye/awesome-mcp-servers)
