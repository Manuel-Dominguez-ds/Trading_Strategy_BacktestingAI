
from app.gemini_chat import GeminiChat
from langchain_core.messages import HumanMessage
import re

def build_code_prompt(user_prompt: str) -> str:
    return f"""
Sos un desarrollador experto en trading algorítmico con Pandas.

A partir de la siguiente descripción de estrategia escrita por un trader:
---
{user_prompt}
---

Tu tarea es generar únicamente una función válida de Python llamada `generate_signals(df)`, que:

- Reciba un DataFrame `df` con columnas típicas de datos financieros (Open, High, Low, Close, Volume).
- Agregue dos columnas booleanas:
  - `df['long_signal']` con `True` en los puntos donde se debe comprar.
  - `df['short_signal']` con `True` donde se debe vender.
- Asegurate de calcular cualquier indicador que se necesite dentro de la función (ej: SMA, AVWAP, RSI, etc.).
- Retorne el `df` modificado.

Requisitos:
- Usá solo código Python válido y ejecutable.
- No agregues comentarios ni explicaciones fuera del código.
- No uses funciones externas no definidas o librerías adicionales, solo NumPy y Pandas.
- El código debe poder ser ejecutado directamente.

Devolvé solo la función `generate_signals(df)` completa, sin ningún texto antes ni después.
"""

def generate_strategy_code(api_key: str,reglas) -> str:
    model = GeminiChat(api_key=api_key)
    prompt = build_code_prompt(reglas)
    response = model.invoke([HumanMessage(content=prompt)])
    content=response.content
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()

import os

def save_strategy_to_file(strategy_name: str, generate_signals_code: str, output_dir="estrategias"):
    os.makedirs(output_dir, exist_ok=True)
    class_name = strategy_name.replace(" ", "_").title().replace("-", "_")

    file_path = os.path.join(output_dir, f"{class_name}.py")

    method_body = extract_function_body(generate_signals_code)

    full_code = f"""
import pandas as pd
import numpy as np
from core.trading_utils import TradingStrategy

class {class_name}(TradingStrategy):
    def __init__(self, risk_reward_ratio=2, stop_loss_pct=0.02):
        super().__init__(strategy_name="{strategy_name}", risk_reward_ratio=risk_reward_ratio, stop_loss_pct=stop_loss_pct)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
{indent_code(method_body, 2)}
    """

    with open(file_path, "w") as f:
        f.write(full_code.strip() + "\n")

    return file_path

def extract_function_body(code: str) -> str:
    """
    Extrae el cuerpo de la función generate_signals(df) generada por la LLM.
    """
    lines = code.splitlines()
    inside_function = False
    body_lines = []

    for line in lines:
        if line.strip().startswith("def generate_signals"):
            inside_function = True
            continue
        if inside_function:
            if line.strip().startswith("return df"):
                body_lines.append(line)  # incluir el return
                break
            body_lines.append(line)

    return "\n".join(body_lines)

def indent_code(code: str, level: int = 1) -> str:
    indent = "    " * level
    return "\n".join(indent + line if line.strip() else "" for line in code.splitlines())










