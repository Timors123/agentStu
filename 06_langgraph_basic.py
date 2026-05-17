"""
第六个 LangChain 程序：LangGraph（状态图）
==========================================
学习目标：
1. 理解 State（状态）—— 在节点之间流转的"数据包"
2. 理解 Node（节点）—— 处理状态的函数
3. 理解 Edge（边）—— 节点之间的连接
4. 理解 Conditional Edge（条件边）—— 根据状态决定走哪条路

LangGraph vs create_agent 的区别：
  create_agent  = 开自动挡，你只管踩油门
  LangGraph     = 开手动挡，你能精确控制每个步骤和分支

图解这个例子：
  ┌──────────┐
  │  开始      │
  └─────┬────┘
        ↓
  ┌──────────┐
  │ analyze   │ ← 节点1：让 LLM 分析代码
  └─────┬────┘
        ↓
  ┌──────────────┐
  │ 条件判断       │ ← 根据 LLM 的判断，走不同分支
  └──┬───────┬──┘
     ↓       ↓
  ┌─────┐  ┌──────────┐
  │ 通过  │  │ 有问题    │ ← 节点2：给出修改建议
  └─────┘  └─────┬────┘
                  ↓
            ┌──────────┐
            │ 给出建议   │
            └──────────┘
"""

from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# ============================================================
# 第1步：定义 State（状态）—— 在节点间流转的数据结构
# ============================================================
# 类比 YOLO：这就好比你定义了一个数据结构来存储每层特征图
# 每个节点（函数）都会收到这个 State，并返回更新后的 State

class CodeReviewState(TypedDict):
    code: str           # 待审查的代码
    analysis: str       # LLM 的分析结果
    verdict: str        # "pass"（通过）或 "fail"（有问题）
    suggestion: str     # 修改建议（如果没通过）


# ============================================================
# 第2步：定义 Node（节点）—— 每个节点是一个函数
# ============================================================
# 约定：每个节点函数接收 State，返回 State 的部分更新

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-9a21d26b3e3b4d0c8b23b9df3f307b16",
    temperature=0,
)


def analyze_code(state: CodeReviewState) -> dict:
    """节点1：分析代码质量，给出评价和判断"""
    code = state["code"]

    response = llm.invoke(f"""请分析以下代码的质量，按这个格式回答（必须严格遵守）：

分析：<你的分析>
判定：<只写 pass 或 fail>

代码：
{code}

判定标准：
- 如果有明显的 bug、安全隐患或严重性能问题，判定为 fail
- 如果代码基本正确、清晰，判定为 pass
""")

    text = response.content
    # 解析 LLM 的输出
    analysis_part = ""
    verdict_part = "pass"

    for line in text.splitlines():
        if line.startswith("分析："):
            analysis_part = line.replace("分析：", "").strip()
        elif line.startswith("判定："):
            verdict_part = line.replace("判定：", "").strip().lower()

    print(f"  [analyze] 判定: {verdict_part}")
    return {"analysis": analysis_part, "verdict": verdict_part}


def suggest_fix(state: CodeReviewState) -> dict:
    """节点2：代码有问题时，给出修改建议"""
    response = llm.invoke(f"""代码分析结果：{state["analysis"]}
请针对上述问题，给出具体的修改建议，包括修改后的代码。""")

    print(f"  [suggest_fix] 建议已生成")
    return {"suggestion": response.content}


# ============================================================
# 第3步：条件判断函数 —— 决定走哪个分支
# ============================================================
def route_after_analysis(state: CodeReviewState) -> Literal["suggest_fix", "pass"]:
    """根据分析结果决定下一步"""
    if state["verdict"] == "fail":
        return "suggest_fix"   # 有问题 → 进入修改建议节点
    else:
        return "pass"          # 没问题 → 结束（END）


# ============================================================
# 第4步：构建 Graph（图）
# ============================================================
def build_graph():
    # 创建状态图，指定状态类型
    graph = StateGraph(CodeReviewState)

    # 添加节点
    graph.add_node("analyze", analyze_code)       # 分析节点
    graph.add_node("suggest_fix", suggest_fix)    # 建议节点

    # 设置入口：从 analyze 开始
    graph.set_entry_point("analyze")

    # 添加条件边：analyze 之后，根据判断走不同路径
    graph.add_conditional_edges(
        "analyze",                # 从这个节点出发
        route_after_analysis,     # 用这个函数判断走哪条路
        {
            "suggest_fix": "suggest_fix",  # fail → 去 suggest_fix
            "pass": END,                   # pass → 直接结束
        }
    )

    # suggest_fix 完成后也结束
    graph.add_edge("suggest_fix", END)

    return graph.compile()


def main():
    app = build_graph()

    # ============================================================
    # 测试1：一段有问题的代码
    # ============================================================
    print("=" * 60)
    print("测试1：审查有问题的代码")
    print("=" * 60)

    bad_code = """
def save_user(user):
    db.execute(f"INSERT INTO users VALUES ('{user.name}', '{user.password}')")
    """

    result = app.invoke({"code": bad_code})
    print(f"\n最终结果：")
    print(f"  判定：{result['verdict']}")
    if result.get("suggestion"):
        print(f"  建议：{result['suggestion'][:200]}...")

    # ============================================================
    # 测试2：一段正常的代码
    # ============================================================
    print()
    print("=" * 60)
    print("测试2：审查正常的代码")
    print("=" * 60)

    good_code = """
def add(a: int, b: int) -> int:
    return a + b
    """

    result2 = app.invoke({"code": good_code})
    print(f"\n最终结果：")
    print(f"  判定：{result2['verdict']}")
    print(f"  分析：{result2['analysis']}")


if __name__ == "__main__":
    main()