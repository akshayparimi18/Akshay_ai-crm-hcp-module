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

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def get_agent():
    tool_caller_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    reasoning_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    
    tool_caller_llm_with_tools = tool_caller_llm.bind_tools(crm_tools)
    tool_node = ToolNode(crm_tools)

    current_date = datetime.now().strftime("%A, %Y-%m-%d")
    current_time = datetime.now().strftime("%I:%M %p")
    
    system_prompt = f"""You are an internal CRM assistant talking DIRECTLY to a pharmaceutical sales rep. Your job is to parse their messy natural language notes, extract structured information, and update the CRM database using your available tools.

CRITICAL RULES:
1. Today's date is {current_date}. The current time is {current_time}. Use this to infer "today" or relative times if the user doesn't specify.
2. ALWAYS use the `log_complete_interaction` tool first to capture the 11 core data points. 
3. After logging the interaction, ALWAYS use the `generate_ai_follow_ups` tool to create 2-3 actionable next steps for the rep based on the meeting's topics and outcomes. You must do this automatically.
4. For `sentiment`, strictly use "Positive", "Neutral", or "Negative".
5. If the user corrects you, use `edit_interaction`.
6. If you receive a COMPLIANCE ERROR in a tool response, follow its instructions EXACTLY to fix the tool call and retry. Do NOT repeat the same rejected values.
7. IMPORTANT: After ALL your tool calls have been successfully executed, you MUST generate a brief, friendly text summary confirming what was logged and any follow-up actions. Never end the conversation on a tool call — always finish with a human-readable message.

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
            
        response = tool_caller_llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last_message = state['messages'][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "compliance"
        return "reasoning"

    def compliance_guardrail(state: AgentState):
        last_message = state['messages'][-1]
        
        # Check if there are tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if tool_call["name"] in ["log_complete_interaction", "edit_interaction"]:
                    args = tool_call["args"]
                    samples_distributed = args.get("samples_distributed", "")
                    follow_up_actions = args.get("follow_up_actions", "")
                    
                    # If the warning is already there, it means the AI self-corrected! Let it pass.
                    if "WARNING: Sample limit exceeded" in str(follow_up_actions):
                        print("✅ GUARDRAIL PASSED — AI self-corrected successfully")
                        continue
                        
                    if samples_distributed:
                        # Extract all numbers from the string
                        numbers = [int(n) for n in re.findall(r'\d+', str(samples_distributed))]
                        # If any number is > 10, it's a violation
                        if any(n > 10 for n in numbers):
                            print("⚠️ GUARDRAIL TRIGGERED")
                            # Create a ToolMessage returning the error back to the LLM
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
                            
        # If no compliance violation, we return nothing to keep the state as is
        return {"messages": []}

    def compliance_router(state: AgentState):
        last_message = state['messages'][-1]
        # If the guardrail added a ToolMessage, route back to agent for self-correction
        if isinstance(last_message, ToolMessage):
            return "agent"
        # Otherwise, proceed to the actual tool execution
        return "tools"

    def call_reasoning_model(state: AgentState):
        messages = state['messages']
        
        # Filter to only relevant messages for the summary — skip compliance error noise
        filtered = []
        for m in messages:
            if isinstance(m, HumanMessage):
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
        response = reasoning_llm.invoke(summary_messages)
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
