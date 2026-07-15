from langchain_core.messages import HumanMessage
from src.graph import compiled_graph
from src.memory import ensure_collection

def run_agent_loop():
    print("🤖 Memory-Augmented Personal Agent Initialized!")
    print("Start chatting (Type 'exit' or 'quit' to stop).\n")
    
    ensure_collection()
    state = {"messages": []}
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            if not user_input.strip():
                continue
            
            # 1. Update State with current user message
            state["messages"] = state.get("messages", []) + [HumanMessage(content=user_input)]
            
            # 2. Run the graph compilation pipeline
            output = compiled_graph.invoke(state)
            
            # 3. Retrieve response from state and update conversation session
            state = output
            latest_response = state["messages"][-1].content
            print(f"\nAgent: {latest_response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    run_agent_loop()