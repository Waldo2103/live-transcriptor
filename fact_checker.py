"""
Capa opcional de fact-checking.
Detecta oraciones con datos verificables y las manda al LLM.
El resto pasa sin tocar.
"""
import re
from typing import Optional
from llm import LLMProvider

# Patrones que sugieren un dato verificable en el texto
_PATTERNS = re.compile(
    r"""
    \d+[.,]\d+\s*%         |   # porcentaje decimal: 7,2%
    \d+\s*%                |   # porcentaje entero:  7%
    \$\s*\d+               |   # monto en pesos/dólares
    \d+\s*(millones?|miles?|billones?) |
    \b\d{4}\b              |   # año: 2023
    \b\d+\s*(personas?|casos?|muertos?|heridos?|votos?|puntos?)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def has_verifiable_claim(text: str) -> bool:
    return bool(_PATTERNS.search(text))


class FactChecker:
    def __init__(self, provider: LLMProvider, enabled: bool = False):
        self.provider = provider
        self.enabled  = enabled

    def check(self, text: str) -> Optional[dict]:
        """
        Retorna None si el fact-check está apagado o no hay claim verificable.
        Retorna dict con status/explanation/confidence si hay claim.
        """
        if not self.enabled:
            return None
        if not has_verifiable_claim(text):
            return None
        result = self.provider.fact_check(text)
        if result.get("status") in ("skip", "error"):
            return None
        return result
