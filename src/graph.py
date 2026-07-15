from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from google.genai import types

from src.config import ai
from src.memory import (
    MemoryPayload, 
    MemoryExtractor, 
    retrieve_memories, 
    save_memory_to_vector_db
)

# --- 1. Graph State Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retrieved_memories: list[str]
    extracted_memories: list[MemoryPayload]

# --- 2. State Nodes ---

def retrieve_node(state: AgentState):
    """Scans long-term memory for relevant semantic blocks."""
    last_user_message = state["messages"][-1].content
    memories = retrieve_memories(last_user_message, limit=3)
    return {"retrieved_memories": memories}

def agent_node(state: AgentState):
    """Executes core LLM tasks with personal context."""
    user_context = "\n".join([f"- {m}" for m in state["retrieved_memories"]])
    
    system_prompt = f"""You are an elite AI systems peer and collaborator. 
You speak to the user as a respected and direct partner.

Here is what you know about them from their long-term system memory:
{user_context if user_context else "No active long-term memories matching this context yet."}

Respond concisely, naturally, and directly. Do not repeat back what they said; instead, answer them as a helpful colleague."""

    # Build the conversation history payload
    contents = []
    for msg in state["messages"]:
        role = "user" if msg.type == "human" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg.content)])
        )

    # Call Gemini directly using the updated stable 3.5 flash model
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.7
    )

    response = ai.models.generate_content(
        model="gemini-3.5-flash",  # Updated model
        contents=contents,
        config=config
    )
    
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response.text)]}

def evaluate_memory_node(state: AgentState):
    """Checks if new knowledge about the user should be stored."""
    if len(state["messages"]) < 2:
        return {}
        
    last_user_message = state["messages"][-2].content
    last_ai_response = state["messages"][-1].content
    
    prompt = f"""Analyze this recent turn of a conversation and determine if the user shared any permanent details about themselves, their preferences, their tech stack, or their current projects.
    
    Recent Turn:
    User: {last_user_message}
    AI: {last_ai_response}"""

    try:
        response = ai.models.generate_content(
            model="gemini-3.5-flash",  # Updated model
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MemoryExtractor,
                temperature=0.0
            )
        )
        
        extracted = MemoryExtractor.model_validate_json(response.text)
        
        for memory in extracted.memories:
            save_memory_to_vector_db(memory)
            print(f"\n[System Log] Memory Captured and Saved to DB: '{memory.content}'")
            
        return {"extracted_memories": extracted.memories}
    except Exception as e:
        return {}

# --- 3. Build & Compile Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("agent", agent_node)
workflow.add_node("memory_eval", evaluate_memory_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "agent")
workflow.add_edge("agent", "memory_eval")
workflow.add_edge("memory_eval", END)

compiled_graph = workflow.compile()