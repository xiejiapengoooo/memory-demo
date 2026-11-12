"""Local entity and semantic memory, exposed through the ``memory`` command."""


def main() -> None:
    import sys

    from memory.agent import create_conversation

    conversation = None
    try:
        conversation = create_conversation()
        print("记忆对话已启动。输入 /exit 或 /quit 退出；Ctrl-D 也可退出。")
        while True:
            try:
                text = input("你：")
            except EOFError:
                break
            if text.strip().lower() in {"/exit", "/quit"}:
                break
            result = conversation.respond(text)
            if result is None:
                continue
            print("助手：" + result.reply)
            memory = result.memory
            labels = {
                "completed": "已完成",
                "skipped": "未写入",
                "partial": "部分完成，尚未全部记住",
                "failed": "失败，未确认保存成功",
            }
            print(f"[记忆] {labels[memory.status]}")
            if memory.status in {"partial", "failed"}:
                for reason in memory.reasons:
                    print(f"  {reason}")
                for error in memory.errors:
                    unknown = "（结果未知，不自动重试）" if error.unknown else ""
                    print(f"  {error.step}: {error.message}{unknown}")
    except KeyboardInterrupt:
        print("\n对话已结束。")
    except Exception as error:
        # Startup and local data errors need an actionable message. Model errors
        # are already sanitized at their boundary in the conversation agent.
        print(f"启动或运行失败：{error}", file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        if conversation is not None:
            conversation.close()
