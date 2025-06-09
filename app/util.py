from .config import settings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI


def get_google_llm() -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        google_api_key=settings.google_api_key,
        model=settings.llm_model_name,
        temperature=0,
    )
