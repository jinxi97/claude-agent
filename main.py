import asyncio
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)


def print_response(message):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(block.text, end="", flush=True)
    elif isinstance(message, ResultMessage):
        cost = (
            f"${message.total_cost_usd:.4f}"
            if message.total_cost_usd is not None
            else "N/A"
        )
        print(f"\n[cost: {cost}]")


async def chat():
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch", "AskUserQuestion", "Skill"],
        setting_sources=["project"],  # Load skills from .claude/skills/
        permission_mode="acceptEdits",
    )

    print("Claude Agent — type 'quit' to exit\n")

    async with ClaudeSDKClient(options=options) as client:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input or user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            print("Claude: ", end="", flush=True)

            # ClaudeSDKClient tracks the session ID internally — each call
            # to client.query() automatically continues the same session.
            await client.query(user_input)
            async for message in client.receive_response():
                print_response(message)

            print()  # newline between turns


asyncio.run(chat())
