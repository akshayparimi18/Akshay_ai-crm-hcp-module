import os
import re
from datetime import datetime
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from tools import crm_tools

# Load environment variables
load_dotenv()

# Model & API configuration with whitespace stripping
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b").strip()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def create_groq_llm(model_name: str, temperature: float = 0.0):
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=GROQ_API_KEY or None
    )

def get_agent():
    # Primary model (120B / configured model)
    primary_tool_llm = create_groq_llm(GROQ_MODEL, temperature=0.0)
    primary_reasoning_llm = create_groq_llm(GROQ_MODEL, temperature=0.1)
    
    # Fallback models (20B and 8B instant)
    fallback_20b_tool_llm = create_groq_llm(GROQ_FALLBACK_MODEL, temperature=0.0)
    fallback_8b_tool_llm = create_groq_llm("llama-3.1-8b-instant", temperature=0.0)
    
    # Bind tools to models
    primary_tool_caller = primary_tool_llm.bind_tools(crm_tools)
    fallback_20b_caller = fallback_20b_tool_llm.bind_tools(crm_tools)
    fallback_8b_caller = fallback_8b_tool_llm.bind_tools(crm_tools)
    
    # LangChain declarative fallback chain
    tool_caller_with_fallbacks = primary_tool_caller.with_fallbacks([
        fallback_20b_caller,
        fallback_8b_caller
    ])
    
    fallback_20b_reasoning = create_groq_llm(GROQ_FALLBACK_MODEL, temperature=0.1)
    fallback_8b_reasoning = create_groq_llm("llama-3.1-8b-instant", temperature=0.1)
    reasoning_with_fallbacks = primary_reasoning_llm.with_fallbacks([
        fallback_20b_reasoning,
        fallback_8b_reasoning
    ])
    
    tool_node = ToolNode(crm_tools)

    current_date = datetime.now().strftime("%A, %Y-%m-%d")
    current_time = datetime.now().strftime("%I:%M %p")
    
    system_prompt = f"""You are an internal CRM assistant talking DIRECTLY to a pharmaceutical sales rep. Your job is to parse their messy natural language notes, extract structured information, and update the CRM database using your available tools.

CRITICAL RULES:
1. Today's date is {current_date}. The current time is {current_time}. Use this to infer "today" or relative times if the user doesn't specify.
2. ALWAYS use the `log_complete_interaction` tool first to capture the core data points.
3. After logging the interaction, ALWAYS use the `generate_ai_follow_ups` tool to create 2-3 actionable next steps for the rep based on the meeting's topics and outcomes. You must do this automatically.
4. For `sentiment`, strictly use "Positive", "Neutral", or "Negative".
5. If the user corrects you, use `edit_interaction`.
6. If you receive a COMPLIANCE ERROR in a tool response, follow its instructions EXACTLY to fix the tool call and retry. Do NOT repeat the same rejected values.
7. IMPORTANT: After ALL your tool calls have been successfully executed, you MUST generate a brief, friendly text summary confirming what was logged and any follow-up actions. Never end the conversation on a tool call — always finish with a human-readable message.

TOOL CALLING & SCHEMA RULES:
- Invoke only one tool at a time with strictly valid JSON arguments adhering exactly to the schema.
- For optional fields not mentioned in notes, pass null or omit them rather than fabricating values.
- Never invent attendees, topics, or materials if not stated by the rep.

PERSONA RULES:
- You are talking to the SALES REP, not to a doctor or HCP. Address the rep directly (e.g., "I've logged your meeting...").
- Do NOT draft emails, letters, or messages addressed to the doctor unless the rep explicitly asks you to.

COMPLIANCE TRANSPARENCY:
- If a compliance guardrail triggered and modified any value (e.g., samples reduced from a higher number to the legal limit of 10), your final message MUST explicitly notify the rep about this adjustment. For example: "Note: The sample quantity was reduced to the legal limit of 10."
- Always reflect the ACTUAL values that were logged, not the original values the rep mentioned."""

    def call_tool_model(state: AgentState):
        messages = state['messages']
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + messages
            
        try:
            response = tool_caller_with_fallbacks.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            print(f"⚠️ Tool caller primary with_fallbacks encountered error: {e}. Retrying rescue...")
            try:
                response = fallback_20b_caller.invoke(messages)
                return {"messages": [response]}
            except Exception as secondary_err:
                print(f"⚠️ Rescue fallback 20b failed: {secondary_err}. Falling back to 8b...")
                response = fallback_8b_caller.invoke(messages)
                return {"messages": [response]}

    def should_continue(state: AgentState):
        last_message = state['messages'][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "compliance"
        return "reasoning"

    def compliance_guardrail(state: AgentState):
        last_message = state['messages'][-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if tool_call["name"] in ["log_complete_interaction", "edit_interaction"]:
                    args = tool_call["args"]
                    samples_distributed = args.get("samples_distributed", "")
                    follow_up_actions = args.get("follow_up_actions", "")
                    
                    if "WARNING: Sample limit exceeded" in str(follow_up_actions):
                        print("✅ GUARDRAIL PASSED — AI self-corrected successfully")
                        continue
                        
                    if samples_distributed:
                        numbers = [int(n) for n in re.findall(r'\d+', str(samples_distributed))]
                        if any(n > 10 for n in numbers):
                            print("⚠️ GUARDRAIL TRIGGERED")
                            violation_count = max(numbers)
                            error_msg = (
                                f"COMPLIANCE ERROR: You attempted to log {violation_count} samples. "
                                f"The legal limit is 10. Your tool call was REJECTED and NOT executed. "
                                f"You MUST rewrite your tool call with these EXACT changes: "
                                f"1) Set samples_distributed to '10' (the maximum allowed). "
                                f"2) APPEND the text 'WARNING: Sample limit exceeded' to the follow_up_actions field. "
                                f"Do NOT attempt to log {violation_count} samples again or the call will be rejected again."
                            )
                            return {"messages": [ToolMessage(
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                                content=error_msg
                            )]}
                            
        return {"messages": []}

    def compliance_router(state: AgentState):
        last_message = state['messages'][-1]
        if isinstance(last_message, ToolMessage):
            return "agent"
        return "tools"

    def call_reasoning_model(state: AgentState):
        messages = state['messages']
        
        filtered = []
        for m in messages:
            if isinstance(m, HumanMessage):
                filtered.append(m)
            elif hasattr(m, "tool_calls") and m.tool_calls:
                filtered.append(m)
            elif isinstance(m, ToolMessage) and "COMPLIANCE ERROR" not in m.content:
                filtered.append(m)
        
        reasoning_prompt = (
            "You are an internal CRM assistant speaking DIRECTLY to a pharmaceutical sales rep. "
            "Based on the conversation, generate a brief, friendly confirmation message. "
            "Mention the HCP name, what was logged, and any follow-up actions. "
            "Keep it to 2-3 sentences. Do NOT use any tools. Just reply with plain text. "
            "IMPORTANT RULES: "
            "1) Address the sales rep directly (e.g., 'I\'ve logged your meeting...'). Do NOT draft emails or messages to the doctor. "
            "2) If samples were reduced due to compliance (e.g., the rep said 25 but only 10 were logged), you MUST mention this adjustment explicitly. "
            "3) Always state the ACTUAL values that were recorded, not the original values the rep mentioned."
        )
        summary_messages = [SystemMessage(content=reasoning_prompt)] + filtered
        try:
            response = reasoning_with_fallbacks.invoke(summary_messages)
            return {"messages": [response]}
        except Exception as e:
            print(f"⚠️ Reasoning model fallback: {e}")
            response = fallback_20b_reasoning.invoke(summary_messages)
            return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_tool_model)
    workflow.add_node("compliance", compliance_guardrail)
    workflow.add_node("tools", tool_node)
    workflow.add_node("reasoning", call_reasoning_model)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"compliance": "compliance", "reasoning": "reasoning"})
    workflow.add_conditional_edges("compliance", compliance_router, {"agent": "agent", "tools": "tools"})
    workflow.add_edge("tools", "agent")
    workflow.add_edge("reasoning", END)
    
    app = workflow.compile()
    return app
