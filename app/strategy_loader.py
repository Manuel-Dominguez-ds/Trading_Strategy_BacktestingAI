import importlib.util
import os

def load_strategy_class(file_path: str):
    """
    Carga dinámicamente una clase de estrategia desde un archivo `.py` generado.
    Retorna la clase (no la instancia).
    """
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    module_path = os.path.abspath(file_path)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Busca una clase que herede de TradingStrategy dentro del módulo
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and 'TradingStrategy' in [base.__name__ for base in obj.__bases__]:
            return obj  # retorna la clase

    raise ValueError(f"No se encontró clase válida de estrategia en {file_path}")
