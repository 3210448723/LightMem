# 这是一个简单的示例，更多使用方法，请查阅openai库的使用.
# https://github.com/openai/openai-python
# 这里的模型都是校内本地的，无法校外调用，也不用担心数据出校.
# 模型有时会因负载过大而崩掉，可以联系高性能计算中心：hpc@csu.edu.cn
# Powered by HPC@csu.edu.cn, 2025.9.10

import os
import re
import time
import json
from openai import OpenAI

# ========== 基本配置 ==========
MY_API_KEY = os.getenv("MY_API_KEY", "sk-n5HTCSKC9XsLtn2zwAdBxaPxF7ubNWgAxwapGJ5Buxd24G80")  #这里修改为你获取的API-KEY
API_BASE_URL  = os.getenv("MY_API_BASE", "http://100.78.197.38:16868/")

# 初始化 OpenAI 客户端
client = OpenAI(api_key=MY_API_KEY, base_url=API_BASE_URL)

# 初始化使用模型deepseek-v3
MODEL_NAME    = os.getenv("MY_MODEL", "deepseek-v3")

# 要查看更多可用的模型，用下面两行获取。
for model in client.models.list():
    print(f"    - {model.id}")

def ask_model(question: str) -> str:
    """
    向模型提问并返回答案。
    """
    messages = [
        {"role": "system", "content": "你是一个有问必答的AI助手。"},
        {"role": "user", "content": question}
    ]

    try:
        # 调用模型 API
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            stream=False,
      #如果你使用deepseek-v3模型，我们部署的当前版本是V3.1，可以通过以下开关开启思考模式。
      #具体参考https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3
      #开启思考模式后，需要取思考内容，需从resp.choices[0].message.reasoning_content
      #extra_body={"chat_template_kwargs": {"thinking": True}}
        )

        # 从响应中提取答案文本
        if resp and resp.choices and resp.choices[0].message:
            return resp.choices[0].message.content.strip()
        else:
            return"模型没有返回有效的答案。"

    except Exception as e:
        return f"调用模型时发生错误: {e}"

# --- 主程序 ---
if __name__ == "__main__":
    question = "请介绍下你自己？"
    print("--- 开始提问 ---")
    print(f"问题: {question}")

    answer = ask_model(question)

    print("\n--- 模型答案 ---")
    print(answer)