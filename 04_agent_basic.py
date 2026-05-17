"""
第四个 LangChain 程序：Agent（智能体）
==========================================
学习目标：
1. 理解 Agent 循环：LLM 决策 → 执行工具 → 结果喂回 LLM → 得到最终回答
2. 理解 create_react_agent —— LangGraph 内置的 Agent 创建函数
3. 理解 Agent 的自动循环机制（可能多次调用工具）

和 03_tool_basic.py 的区别：
  03: LLM 输出 tool_calls → 你得手动执行 → 手动喂回去
  04: LLM 输出 tool_calls → Agent 自动执行 → 自动喂回去 → 循环直到完成

Agent 循环示意图：
  ┌─────────────────────────────────────────┐
  │  用户问题 → LLM 决策                      │
  │           ↓         ↘                    │
  │      直接回答    需要工具                   │
  │                    ↓                      │
  │              执行工具函数                   │
  │                    ↓                      │
  │              结果喂回 LLM                   │
  │                    ↓                      │
  │           LLM 再决策（循环）                │
  └─────────────────────────────────────────┘
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


# ============================================================
# 定义工具（和 03 一样）
# ============================================================

@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。当你需要做加法时使用这个工具。"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。当你需要做乘法时使用这个工具。"""
    return a * b


@tool
def subtract(a: int, b: int) -> int:
    """计算 a 减 b 的差。当你需要做减法时使用这个工具。"""
    return a - b


@tool
def get_current_weather(city: str) -> str:
    """查询指定城市的当前天气。参数 city 是城市名（中文）。"""
    weather_data = {
        "北京": "晴天，25°C，湿度40%",
        "上海": "多云，28°C，湿度65%",
        "深圳": "阵雨，30°C，湿度80%",
    }
    return weather_data.get(city, f"没有{city}的天气数据")


def main():
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-9a21d26b3e3b4d0c8b23b9df3f307b16",
        temperature=0,
    )

    # ============================================================
    # 核心：create_react_agent —— 一行代码创建 Agent
    # ============================================================
    # ReAct = Reasoning + Acting（推理 + 行动）
    # Agent 自动完成：决策 → 执行工具 → 观察结果 → 继续决策 → ... → 最终回答
    agent = create_react_agent(model=llm, tools=[add, multiply, subtract, get_current_weather])

    # ============================================================
    # 场景1：多步骤数学题 —— Agent 需要多次调用工具
    # ============================================================
    print("=" * 60)
    print("场景1：(100 + 50) × 2 - 30 = ?  （需要调3次工具）")
    print("=" * 60)

    result = agent.invoke({
        "messages": [{"role": "user", "content": "计算 (100 + 50) × 2 - 30 等于多少？请一步步来。"}]
    })

    # Agent 返回的消息列表中，最后一条就是最终回答
    for i, msg in enumerate(result["messages"]):
        role = msg.__class__.__name__ if hasattr(msg, '__class__') else msg.get("role", "unknown")
        # 只打印 AI 的消息和工具调用的消息
        if hasattr(msg, 'content') and msg.content:
            content_preview = str(msg.content)[:200]
            print(f"[{role}] {content_preview}")
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"[TOOL_CALL] {tc['name']}({tc['args']})")

    print()
    print(">>> 看到了吗？Agent 自动调用了 3 次工具，最后汇总出结果")
    print(">>> 你不需要手动执行任何工具，Agent 全自动完成")

    # ============================================================
    # 场景2：需要外部知识的问题
    # ============================================================
    print()
    print("=" * 60)
    print("场景2：结合天气和计算 —— 多工具混合")
    print("=" * 60)

    result2 = agent.invoke({
        "messages": [{"role": "user", "content": "北京和上海的温度差是多少度？"}]
    })

    for msg in result2["messages"]:
        if hasattr(msg, 'content') and msg.content:
            content_preview = str(msg.content)[:200]
            print(f"[{msg.__class__.__name__}] {content_preview}")
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"[TOOL_CALL] {tc['name']}({tc['args']})")

    print()
    print(">>> Agent 自动：查北京天气 → 查上海天气 → 计算温差 → 回答")


if __name__ == "__main__":
    main()