from loguru import logger
from .model import AgentState, TopicSelectionParser
from .util import get_google_llm
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts.prompt import PromptTemplate


def supervisor_handler(state: AgentState):
    question = state["messages"][-1]
    logger.debug(f"{question=}")

    template = """Your task is to classify the given user query into one of the following categories: [USA,Not Related].
    Only respond with the category name and nothing else.

    User query: {question}
    {format_instructions}
    """
    parser = PydanticOutputParser(pydantic_object=TopicSelectionParser)

    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    llm = get_google_llm()
    chain = prompt | llm | parser
    result = chain.invoke({"question": question})
    logger.debug(f"response from supervisor llm:{result}")
    return {"messages": [result.topic]}
