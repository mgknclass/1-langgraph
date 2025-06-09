from loguru import logger
from langgraph.graph import StateGraph, END
from .supervisor import supervisor_handler
from .rag import handle_rag_call
from .llm import handle_llm_call
from .router import route_handler
from .model import AgentState


def main():
    logger.debug("Starting...")

    work_flow = StateGraph(state_schema=AgentState)
    work_flow.add_node("supervisor", supervisor_handler)
    work_flow.add_node("RAG", handle_rag_call)
    work_flow.add_node("LLM", handle_llm_call)
    work_flow.add_conditional_edges(
        "supervisor", route_handler, {"RAG CALL": "RAG", "LLM CALL": "LLM"}
    )
    work_flow.set_entry_point("supervisor")
    work_flow.add_edge("RAG", END)
    work_flow.add_edge("LLM", END)

    app = work_flow.compile()

    # state = {"messages": ["what is the gdp of usa?"]}
    state = {
        "messages": ["can you tell me the industrial growth of world's poor economy?"]
    }
    # state = {
    #     "messages": [
    #         "can you tell me the industrial growth of world's most powerful economy?"
    #     ]
    # }

    result = app.invoke(state)
    logger.debug(f"Final app response: {result}")

    logger.debug("Completed...")


if __name__ == "__main__":
    main()
