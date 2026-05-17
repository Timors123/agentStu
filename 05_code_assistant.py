"""
实践项目：代码助手 Agent
==========================================
功能：一个能读取你本地文件、帮你分析代码的 Agent

真实工具：
  - list_files: 列出项目里有哪些文件
  - read_file: 读取指定文件的内容
  - search_code: 在代码里搜索关键词

你可以问他：
  "我的项目里有哪些 Python 文件？"
  "帮我解释一下 03_tool_basic.py 里做了什么"
  "找一下所有用到 create_agent 的地方"
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from pathlib import Path

# 设定项目根目录
PROJECT_ROOT = Path(__file__).parent


# ============================================================
# 工具1：列出项目文件
# ============================================================
@tool
def list_files(extension: str = ".py") -> str:
    """列出项目中指定后缀的文件。参数 extension 是文件后缀，如 .py 或 .md，默认 .py"""
    files = list(PROJECT_ROOT.glob(f"*{extension}"))
    if not files:
        return f"没有找到 {extension} 文件"
    return "\n".join(f.name for f in sorted(files))


# ============================================================
# 工具2：读取文件内容
# ============================================================
@tool
def read_file(filename: str) -> str:
    """读取项目中指定文件的内容。参数 filename 是文件名（不是完整路径），如 '01_hello_llm.py'"""
    filepath = PROJECT_ROOT / filename
    if not filepath.exists():
        return f"错误：文件 {filename} 不存在。请先用 list_files 看看有哪些文件。"
    try:
        return filepath.read_text(encoding="utf-8")
    except Exception as e:
        return f"读取文件出错：{e}"


# ============================================================
# 工具3：搜索代码内容
# ============================================================
@tool
def search_code(keyword: str) -> str:
    """在项目所有 .py 文件中搜索包含 keyword 的行。返回文件名和匹配的行。"""
    results = []
    for py_file in PROJECT_ROOT.glob("*.py"):
        for i, line in enumerate(py_file.read_text(encoding="utf-8").splitlines(), 1):
            if keyword.lower() in line.lower():
                results.append(f"  {py_file.name}:{i}  {line.strip()}")
    if not results:
        return f"没有找到包含 '{keyword}' 的代码。"
    return f"找到 {len(results)} 处匹配：\n" + "\n".join(results[:20])


def main():
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-9a21d26b3e3b4d0c8b23b9df3f307b16",
        temperature=0.3,
    )

    # 创建 Agent，装上三个文件工具
    agent = create_agent(
        model=llm,
        tools=[list_files, read_file, search_code],
        system_prompt="你是一个代码助手。用户会问你关于这个项目代码的问题。"
                       "先用 list_files 了解项目结构，再用 read_file 读取文件，需要时用 search_code 搜索。"
                       "用中文回答。",
    )

    print("=" * 60)
    print("代码助手 Agent 已启动（输入 'quit' 退出）")
    print("试试问：项目里有哪些文件？")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("再见！")
            break

        print("\n>>> Agent 工作中...")
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_input}],
        })

        # 打印工具调用过程
        for msg in result["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    args_str = str(tc["args"])
                    if len(args_str) > 60:
                        args_str = args_str[:60] + "..."
                    print(f"  [调用工具] {tc['name']}({args_str})")
            elif hasattr(msg, 'content') and msg.content and msg.__class__.__name__ == "AIMessage":
                print(f"\nAgent: {msg.content}")


if __name__ == "__main__":
    main()