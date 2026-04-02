"""
Abstracción de providers LLM para fact-checking.
Para agregar uno nuevo: heredar de LLMProvider e implementar fact_check().
Registrarlo en get_provider().
"""
import json
import os
import urllib.request
from abc import ABC, abstractmethod
from typing import Optional

FACT_CHECK_PROMPT = """Sos un verificador de datos. Se dijo esto en una transmisión en vivo:

"{claim}"

Tu tarea:
1. Si no hay dato verificable (cifra, estadística, fecha, nombre de ley, resultado electoral), respondé solo: SKIP
2. Si hay dato verificable, respondé en este formato exacto:
ESTADO: VERIFICADO | DUDOSO | FALSO | NO_VERIFICABLE
CONFIANZA: 0.0-1.0
EXPLICACIÓN: (1-2 oraciones en español)

No agregues nada más."""


class LLMProvider(ABC):
    @abstractmethod
    def fact_check(self, claim: str) -> dict:
        """
        Retorna:
          {"status": "verified"|"questionable"|"false"|"unverifiable"|"skip"|"error",
           "explanation": str, "confidence": float}
        """

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @staticmethod
    def _parse_response(text: str) -> dict:
        text = text.strip()
        if text.upper() == "SKIP":
            return {"status": "skip", "explanation": "", "confidence": 1.0}

        status = "unverifiable"
        confidence = 0.5
        explanation = ""

        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("ESTADO:"):
                val = line.split(":", 1)[1].strip().upper()
                if "VERIFICADO" in val:    status = "verified"
                elif "DUDOSO" in val:      status = "questionable"
                elif "FALSO" in val:       status = "false"
                elif "NO_VERIFICABLE" in val: status = "unverifiable"
            elif line.upper().startswith("CONFIANZA:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.upper().startswith("EXPLICACIÓN:") or line.upper().startswith("EXPLICACION:"):
                explanation = line.split(":", 1)[1].strip()

        return {"status": status, "explanation": explanation, "confidence": confidence}


class OllamaProvider(LLMProvider):
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3"):
        self.host  = host.rstrip("/")
        self.model = model

    def fact_check(self, claim: str) -> dict:
        prompt  = FACT_CHECK_PROMPT.format(claim=claim)
        payload = json.dumps({
            "model": self.model, "prompt": prompt,
            "stream": False, "options": {"temperature": 0.1}
        }).encode()
        try:
            req = urllib.request.Request(
                f"{self.host}/api/generate", data=payload,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data   = json.loads(resp.read())
                return self._parse_response(data.get("response", ""))
        except Exception as e:
            return {"status": "error", "explanation": str(e), "confidence": 0.0}

    def is_available(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.host}/api/tags", timeout=5)
            return True
        except Exception:
            return False


class ClaudeProvider(LLMProvider):
    """
    Requiere: pip install anthropic
    Variable de entorno: ANTHROPIC_API_KEY
    """
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "claude-haiku-4-5-20251001"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model   = model

    def fact_check(self, claim: str) -> dict:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            msg = client.messages.create(
                model=self.model, max_tokens=256,
                messages=[{"role": "user",
                           "content": FACT_CHECK_PROMPT.format(claim=claim)}]
            )
            return self._parse_response(msg.content[0].text)
        except Exception as e:
            return {"status": "error", "explanation": str(e), "confidence": 0.0}

    def is_available(self) -> bool:
        return bool(self.api_key)


# ── Factory ──────────────────────────────────────────────────────────────────

def get_provider(name: str = "ollama", **kwargs) -> LLMProvider:
    """
    Uso:
        get_provider("ollama", host="http://...", model="llama3")
        get_provider("claude", api_key="sk-...")
    """
    name = name.lower()
    if name == "ollama":
        return OllamaProvider(
            host  = kwargs.get("host",  os.getenv("OLLAMA_HOST",   "http://localhost:11434")),
            model = kwargs.get("model", os.getenv("OLLAMA_MODELO", "llama3")),
        )
    if name == "claude":
        return ClaudeProvider(
            api_key = kwargs.get("api_key"),
            model   = kwargs.get("model", "claude-haiku-4-5-20251001"),
        )
    raise ValueError(f"Provider desconocido: '{name}'. Disponibles: ollama, claude")
