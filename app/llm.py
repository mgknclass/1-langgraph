from loguru import logger
from .model import AgentState
from .util import get_google_llm


def handle_llm_call(state: AgentState):
    logger.debug("handle llm call")
    question = state["messages"][0]
    logger.debug(f"{question=}")

    llm = get_google_llm()
    result = llm.invoke(
        "Answer the follow question with you knowledge of the real world. Following is the user question: "
        + question
    )
    logger.debug(f"response from llm call: {result}")
    return {"messages": [result.content]}
