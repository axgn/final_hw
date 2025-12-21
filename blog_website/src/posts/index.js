import myhomeworkContent from "./myhomework.md?raw";
import wslTutorialContent from "./wsl_tutorial.md?raw";
import vscodemcpTutorialContent from "./vscode_mcp_tutorial.md?raw";
import inlineandstaticContent from "./inline_and_static.md?raw";

const posts = [
  {
    slug: "myhomework",
    title: "云计算和大数据概论作业",
    content: myhomeworkContent,
    postId: 1,
  },
  {
    slug: "wsl_tutorial",
    title: "WSL自定义内核教程",
    content: wslTutorialContent,
    postId: 2,
  },
  {
    slug: "vscode_mcp_tutorial",
    title: "VSCode远程连接MCP教程",
    content: vscodemcpTutorialContent,
    postId: 3,
  },
  {
    slug: "inline_and_static",
    title: "有关c++和c内联和静态的思考",
    content: inlineandstaticContent,
    postId: 4,
  }
];

export default posts;
