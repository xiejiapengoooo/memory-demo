import argparse
import os

from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek


def build_agent():
    return create_agent(
        model=ChatDeepSeek(
            model="deepseek-flash",
        ),
        system_prompt=("你是一个智能体助手"),
    )


def main() -> None:
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
    print(result["messages"][-1].text)
