# tests/test_llm_agent_concurrency.py
"""
Verifica el límite de concurrencia y la contabilidad de fallos de LLMAgent.
No requiere Ollama: sustituye el cliente por uno falso.

Ejecutar:  python tests/test_llm_agent_concurrency.py
"""

import asyncio
import os
import sys

# Los mensajes de descarte llevan emojis; sin esto Windows crasea al redirigir la salida.
sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import BaseModel
from agents.llm_agent import LLMAgent


class _Salida(BaseModel):
    prompt: str


class ClienteFalso:
    """Cuenta cuántas llamadas están en vuelo simultáneamente."""

    def __init__(self, respuesta='{"prompt": "ok"}', excepcion=None):
        self.respuesta = respuesta
        self.excepcion = excepcion
        self.en_vuelo = 0
        self.pico = 0

    async def chat(self, **kwargs):
        self.en_vuelo += 1
        self.pico = max(self.pico, self.en_vuelo)
        try:
            await asyncio.sleep(0.01)  # simula la generación
            if self.excepcion:
                raise self.excepcion
            return {"message": {"content": self.respuesta}}
        finally:
            self.en_vuelo -= 1


async def _lanzar(agente, n):
    tareas = [
        agente.call_llm("sys", "user", _Salida, timeout_segundos=5)
        for _ in range(n)
    ]
    return await asyncio.gather(*tareas)


async def main():
    # 1. El semáforo limita las peticiones en vuelo, aunque se disparen 30 a la vez.
    agente = LLMAgent(max_concurrent=3)
    agente.client = ClienteFalso()
    resultados = await _lanzar(agente, 30)

    assert agente.client.pico <= 3, f"Concurrencia sin límite: pico={agente.client.pico}"
    assert agente.client.pico > 1, "El semáforo serializó todo; se esperaba paralelismo"
    assert all(r is not None for r in resultados), "Se perdieron respuestas válidas"
    assert agente.stats["ok"] == 30, agente.stats
    assert agente.failures == 0, agente.stats

    # 2. JSON malformado se cuenta como invalid_json, no como error genérico.
    agente = LLMAgent(max_concurrent=2)
    agente.client = ClienteFalso(respuesta='{"campo_equivocado": 1}')
    assert await _lanzar(agente, 4) == [None] * 4
    assert agente.stats["invalid_json"] == 4, agente.stats
    assert agente.failures == 4

    # 3. Una caída del runner de Ollama se cuenta como error y no propaga.
    agente = LLMAgent(max_concurrent=2)
    agente.client = ClienteFalso(excepcion=RuntimeError("llama runner process no longer running: 2"))
    assert await _lanzar(agente, 3) == [None] * 3
    assert agente.stats["error"] == 3, agente.stats

    # 4. El timeout mide generación, no cola: 10 llamadas de 0.01s con
    #    concurrencia 1 no deben expirar contra un timeout de 1s por llamada.
    agente = LLMAgent(max_concurrent=1)
    agente.client = ClienteFalso()
    await _lanzar(agente, 10)
    assert agente.stats["timeout"] == 0, agente.stats
    assert agente.stats["ok"] == 10, agente.stats

    print("OK: concurrencia limitada y fallos contabilizados")


if __name__ == "__main__":
    asyncio.run(main())
