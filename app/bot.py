from app.gemini_chat import GeminiChat
from langchain_core.messages import HumanMessage


class Bot:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = GeminiChat(api_key=api_key)

    def generate_insights(self, resumen_resultados: str) -> str:
        prompt = f"""
Sos un analista financiero. Generá insights útiles para un trader, explicando de forma clara y breve qué significa el siguiente resultado de backtest:

{resumen_resultados}

No repitas los números tal como están. Enfocate en explicar qué funciona, qué puede mejorar y si conviene usar esta estrategia.
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()


