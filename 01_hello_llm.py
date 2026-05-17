"""
第一个 LangChain 程序：调用 DeepSeek 模型
==========================================
学习目标：
1. 理解 LLM 调用的最基本流程：Prompt → Model → Output
2. 理解 LangChain 中的 Message 类型（SystemMessage / HumanMessage）
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

def main():
    # ============================================================
    # 第1步：创建 LLM 实例（连接到模型）
    # ============================================================
    # 类比 YOLO：这一步就像是 model = YOLO("yolov8n.pt")
    # 你告诉 LangChain：用哪个模型，模型在哪，有什么参数
    llm = ChatOpenAI(
        model="deepseek-v4-flash",           # DeepSeek V3 模型
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # DashScope 地址
        api_key="sk-9a21d26b3e3b4d0c8b23b9df3f307b16",
        temperature=0.7,               # 温度：控制回答的随机性（0=严谨，1=创意）
    )

    # ============================================================
    # 第2步：构建消息（告诉模型"你是谁"和"我要问什么"）
    # ============================================================
    messages = [
        # SystemMessage：设定模型的角色/行为（类似给 YOLO 设定检测类别）
        SystemMessage(content="你是一个AI助手，请用中文回答，回答尽量简洁。"),
        # HumanMessage：用户实际的问题（类似给 YOLO 输入一张图片）
        HumanMessage(content="你好！请简单介绍一下LangChain是什么，用一句话概括。"),
    ]

    # ============================================================
    # 第3步：调用模型，获取回复
    # ============================================================
    # 类比 YOLO：这一步就像是 results = model(image)
    response = llm.invoke(messages)

    # ============================================================
    # 第4步：打印结果
    # ============================================================
    print("=" * 60)
    print("模型的回复：")
    print("=" * 60)
    print(response.content)


if __name__ == "__main__":
    main()