from loguru import logger
from .model import AgentState


def route_handler(state: AgentState):
    logger.debug("***router***")
    message = state["messages"][-1]
    if "usa" in message.lower():
        logger.debug("returning rag call from router")
        return "RAG CALL"
    else:
        logger.debug("returning llm call from router")
        return "LLM CALL"
