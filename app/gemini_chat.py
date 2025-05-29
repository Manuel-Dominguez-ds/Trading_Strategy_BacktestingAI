from typing import ClassVar, List, Optional
from pydantic import PrivateAttr
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
import google.generativeai as genai

import pandas as pd

class GeminiChat(BaseChatModel, Runnable):
    model_name: ClassVar[str] = "gemini-2.0-flash"
    _model: genai.GenerativeModel = PrivateAttr()

    def __init__(self, api_key: str):
        super().__init__()  # necesario al heredar de pydantic.BaseModel
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name=self.model_name)

    def _generate(self, messages: List[HumanMessage], stop: Optional[List[str]] = None) -> ChatResult:
        prompt = "\n".join([msg.content for msg in messages])
        response = self._model.generate_content(prompt)
        return ChatResult(generations=[{
            "text": response.text,
            "message": AIMessage(content=response.text)
        }])

    @property
    def _llm_type(self) -> str:
        return "gemini"
