# wsl自定义内核教程

之前因为要使用的雷达串口驱动是CP210x类型，wsl默认没有对这种串口驱动的支持，
所以就自己编译了一个提供该类型驱动支持的内核，这里就简单记录一下：
如何完成wsl自定义内核的编译。

<!-- more -->

## 下载源码

首先我们可以在github上下载[wsl 内核源码](https://github.com/microsoft/WSL2-Linux-Kernel/releases)
如果有git的话也可以直接用
`git clone https://github.com/microsoft/WSL2-Linux-Kernel.git`
不过使用git clone 命令下载的文件会更多一些，因为会包含.git目录，
但是如果使用git的话可以通过
`git checkout -b <new_branch_name> tag`或者`git checkout -b <new_branch_name> origin/<remote_branch_name>`的方式轻松切换自己想要编译的内核版本，
如果有新的版本，也可以直接通过
`git pull` 或者是`git fetch`进行同步

## 安装依赖

使用以下命令安装相关内核构建的依赖
`sudo apt install build-essential flex bison dwarves libssl-dev libelf-dev cpio qemu-utils`

## 自定义配置

把Microsoft目录下config-wsl文件复制一份
到源代码根目录改名为.config
可以在源代码根目录下直接执行
`cp Microsoft/config-wsl .config`
然后使用make menuconfig命令进行配置
可以使用/进行查找，中间可能会出现模块依赖问题，可以通过h查看相关信息
(就比如说，我在打开cp210x选项支持时，就一直报错，在网上搜索也没有结果
后来搜了一下make menuconfig的一些教程了解到，可以通过h查看相关信息
才通过查看cp210x选项，进而找到它的依赖项)
最后通过make命令编译, 如果觉得比较慢，
可以通过-j参数指定编译使用的核心数量。
最终生成的镜像文件在arch/x86/boot/bzImage目录下

## 配置wsl配置文件使用自定义内核

首先把你编译好的镜像复制到windows系统中，
然后打开windows用户目录下的.wslconfig文件，如果没有就创建一个
添加以下配置

```bash
[wsl2]
kernel=path
```

把你自己的镜像路径复制到path的位置，然后再重启wsl，这样就成功实现了对
wsl内核的自定义。

ps. 这里也可以通过gui进行配置，

1. 在windows搜索wsl setting，打开wsl setting。
2. 在wsl setting中切换到开发商选项。
3. 将路径复制到自定义内核选项中。
