
#  Memory-Augmented Personal Agent

A lightweight, local, state-of-the-art conversational AI agent designed to dynamically extract, store, and recall personal context, developer tech stacks, and user preferences across chat sessions.

This project was designed, architected, and developed by **[Valentina Kiyungi](https://github.com/Valentina14142000)**.

---

##  Key Features
* **Dynamic Memory Extraction:** Automatically analyzes conversational turns in the background to capture permanent details, preferences, and project updates.
* **Semantic Context Retrieval:** Searches an in-memory vector database using semantic similarity matching to inject relevant context into the model's system prompt.
* **Modern Orchestration:** Built utilizing **LangGraph** to model state, execution nodes, and background evaluations reliably.

---

##  Tech Stack
* **Orchestration:** LangGraph (StateGraph Workflow)
* **LLM Engine:** Google GenAI SDK (`gemini-2.5-flash` / `gemini-3.5-flash`)
* **Vector Database:** Qdrant Client (In-Memory Semantic Storage)
* **Language Runtime:** Python 3.13+

---

##  Project Structure
```text
memory-agent/
├── src/
│   ├── config.py       # Configuration & Secure Client Initializations
│   ├── graph.py        # LangGraph Workflow, State Nodes, and Pipeline Compilation
│   └── memory.py       # Vector DB schemas, schemas definition & extraction utility
├── .env                # Private API credentials (ignored by Git)
├── .gitignore          # Git exclusion rules
├── app.py              # Conversational CLI runtime loop
└── README.md           # Documentation
