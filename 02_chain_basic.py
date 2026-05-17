"""
第二个 LangChain 程序：Chain（链）
==========================================
学习目标：
1. 理解 PromptTemplate —— 把用户输入"填入"模板
2. 理解 Chain —— 把 Prompt + LLM 串成一个管道
3. 理解 LCEL（LangChain Expression Language）—— 用 | 管道符串联步骤

YOLO 类比：
- PromptTemplate = 数据预处理/标准化（把原始输入变成模型能理解的格式）
- Chain = 推理管道（preprocess → model → postprocess）
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def main():
    # 创建 LLM（和第1个例子完全一样）
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-9a21d26b3e3b4d0c8b23b9df3f307b16",
        temperature=0.7,
    )

    # ============================================================
    # 新概念 1：PromptTemplate（提示词模板）
    # ============================================================
    # 用 {花括号} 做占位符，运行时自动填入
    # 类比：YOLO 里你把不同尺寸的图片 resize 到 640x640
    #       PromptTemplate 把不同的用户输入"格式化"成统一的提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，请用中文回答。"),
        ("human", "{user_input}"),
    ])

    # ============================================================
    # 新概念 2：LCEL 管道（| 符号）
    # ============================================================
    # | 的含义：把左边的输出 → 传给右边作为输入
    #
    # prompt | llm  意思是：
    #   用户输入 → 填入模板 → 变成消息列表 → 发给 LLM → 拿到回复
    #
    # 类比 YOLO 的推理管道：
    #   letterbox(img) | model | nms(preds)   （YOLO 没有这种语法，但逻辑一样）

    chain = prompt | llm

    # ============================================================
    # 调用 chain：invoke 的参数会去填充模板里的 {占位符}
    # ============================================================

    # 场景1：让它当翻译
    result1 = chain.invoke({
        "role": "英文翻译专家",
        "user_input": "把下面这句话翻译成英文：今天天气真好，我们去爬山吧。",
    })
    print("场景1 - 翻译：")
    print(result1.content)
    print()

    # 场景2：同一个链，换一个角色
    result2 = chain.invoke({
        "role": "代码审查员",
        "user_input": "这段代码有什么问题？\ndef add(a, b): return a + b",
    })
    print("场景2 - 代码审查：")
    print(result2.content)
    print()

    # 场景3：再换一个角色 —— 看到了吗，同一个链，不同用途
    result3 = chain.invoke({
        "role": "小学老师",
        "user_input": "请用小朋友能听懂的方式解释什么是黑洞。",
    })
    print("场景3 - 小学老师：")
    print(result3.content)


if __name__ == "__main__":
    main()