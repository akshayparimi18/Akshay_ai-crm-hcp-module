from dotenv import load_dotenv
load_dotenv(override=True)

from agent import get_agent
from langchain_core.messages import HumanMessage

agent = get_agent()
print("Agent compiled. Invoking...")

try:
    result = agent.invoke(
        {"messages": [HumanMessage(content="I met Dr. Jones today, he was positive, and I gave him 15 samples.")]},
        {"recursion_limit": 15}
    )
    print("\n=== FINAL RESULT ===")
    for i, m in enumerate(result["messages"]):
        print(f"\n--- Message {i} ({type(m).__name__}) ---")
        print(f"Content: {m.content[:200] if m.content else '(empty)'}")
        if hasattr(m, 'tool_calls') and m.tool_calls:
            for tc in m.tool_calls:
                print(f"  Tool: {tc['name']}, follow_up_actions: {tc['args'].get('follow_up_actions', 'N/A')}")
except Exception as e:
    print(f"ERROR: {e}")
