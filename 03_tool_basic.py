"""
第三个 LangChain 程序：Tool（工具）
==========================================
学习目标：
1. 理解 @tool 装饰器 —— 把普通 Python 函数变成 LLM 能调用的工具
2. 理解 .bind_tools() —— 把工具"装备"到 LLM 身上
3. 理解 Tool Call —— LLM 决定"我需要用这个工具"，并告诉你参数

核心思想（Agent 的灵魂）：
  LLM 自己决定：要不要用工具？用哪个工具？传什么参数？
  你只需要提供工具，LLM 自主决策。

YOLO 类比：
  普通 LLM = 只能检测但不能定位的 YOLO
  LLM + Tool = YOLO 不仅能说出"图里有猫"，还能告诉你猫的坐标
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# ============================================================
# 第1步：定义工具 —— @tool 装饰器
# ============================================================
# 一个工具就是一个普通的 Python 函数
# 关键是：写清楚 docstring（函数的注释），LLM 会读这个来决定什么时候用这个工具！

@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。当你需要做加法时使用这个工具。"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。当你需要做乘法时使用这个工具。"""
    return a * b


@tool
def get_current_weather(city: str) -> str:
    """查询指定城市的当前天气。参数 city 是城市名（中文）。"""
    # 这里用模拟数据，真实场景下你会调一个天气 API
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
    # 第2步：把工具"绑定"到 LLM 上
    # ============================================================
    # .bind_tools() 就像是给 LLM 发了一张"说明书"
    # LLM 会读到每个工具的 docstring，知道：
    #   - 有哪些工具可用
    #   - 每个工具能做什么
    #   - 需要什么参数
    llm_with_tools = llm.bind_tools([add, multiply, get_current_weather])

    # ============================================================
    # 第3步：问一个问题，看 LLM 自己决定用哪个工具
    # ============================================================

    # 场景1：数学计算 —— LLM 应该自动选择 add 工具
    print("=" * 60)
    print("场景1：问 LLM 一个数学问题")
    print("=" * 60)
    response = llm_with_tools.invoke([
        HumanMessage(content="123 + 456 等于多少？"),
    ])
    print("LLM 的回复（不是计算结果，而是告诉你它想调用什么工具）：")
    print(f"  content: {response.content}")
    print(f"  tool_calls: {response.tool_calls}")
    print()
    print(">>> 看到了吗？LLM 没有直接回答，而是说我要调用 add(a=123, b=456)")
    print(">>> 这就是 Agent 的决策过程：LLM 负责'决策调用哪个工具'，不负责执行！")

    # 场景2：天气查询 —— LLM 应该自动选择天气工具
    print()
    print("=" * 60)
    print("场景2：问 LLM 天气")
    print("=" * 60)
    response2 = llm_with_tools.invoke([
        HumanMessage(content="北京今天天气怎么样？"),
    ])
    print(f"  content: {response2.content}")
    print(f"  tool_calls: {response2.tool_calls}")

    # 场景3：不需要工具的问题 —— LLM 直接回答
    print()
    print("=" * 60)
    print("场景3：不需要工具的问题")
    print("=" * 60)
    response3 = llm_with_tools.invoke([
        HumanMessage(content="你好，请问Python是什么时候发布的？"),
    ])
    print(f"  content: {response3.content}")
    print(f"  tool_calls: {response3.tool_calls}")
    print()
    print(">>> 这次 tool_calls 是空的，LLM 判断不需要用工具，直接回答")


if __name__ == "__main__":
    main()