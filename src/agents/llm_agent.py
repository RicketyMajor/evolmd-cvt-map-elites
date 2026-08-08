# agents/llm_agent.py

import ollama
import asyncio
import os
from typing import Optional, Type
from pydantic import BaseModel, ValidationError

# Límite de peticiones simultáneas a Ollama.
# El GA dispara asyncio.gather sobre toda la población (N=100+), lo que abre N
# contextos KV en paralelo y termina matando al runner ("llama runner process no
# longer running: 2", status 500). Los individuos perdidos se vuelven
# generated_data="" y contaminan las métricas en silencio.
# Configurable por entorno para reproducir las condiciones de otra máquina:
#   EVOLMD_MAX_CONCURRENT=999 restaura el comportamiento sin límite.
MAX_CONCURRENT = int(os.getenv("EVOLMD_MAX_CONCURRENT", "6"))

# Ventana de contexto y tope de generación, FIJOS y explícitos.
# El código antes no fijaba num_ctx, así que heredaba el default del servidor Ollama
# (2048 en la 0.4.1 del servidor, distinto en versiones nuevas). Eso hacía que la
# versión de Ollama instalada fuera una variable oculta del experimento.
# Además num_predict=2048 con num_ctx=2048 no deja presupuesto para el prompt:
# la respuesta se corta a mitad del JSON. La salida más larga observada en 6962
# textos fue de 134 palabras (~180 tokens), así que 1024 es un techo holgado.
NUM_CTX = int(os.getenv("EVOLMD_NUM_CTX", "4096"))
NUM_PREDICT = int(os.getenv("EVOLMD_NUM_PREDICT", "1024"))


class LLMAgent:
    """
    Clase centralizada asíncrona para manejar todas las interacciones con el modelo LLM.
    """
    def __init__(self, model: str = "llama3", max_concurrent: int = MAX_CONCURRENT):
        self.model = model
        self.client = ollama.AsyncClient(host='http://127.0.0.1:11434')
        self.max_concurrent = max_concurrent
        self.num_ctx = NUM_CTX
        self.num_predict = NUM_PREDICT
        self._sem = asyncio.Semaphore(max_concurrent)
        # Contabilidad de descartes: sin esto un fallo del LLM es indistinguible
        # de un individuo legítimamente malo al leer los resultados.
        self.stats = {"ok": 0, "timeout": 0, "invalid_json": 0, "error": 0}

    @property
    def failures(self) -> int:
        """Total de llamadas que no produjeron un individuo válido."""
        return self.stats["timeout"] + self.stats["invalid_json"] + self.stats["error"]

    async def preflight(self):
        """
        Verifica que Ollama responda y que el modelo exista.
        Sin esto, una corrida de 10 h termina con todos los generated_data vacíos
        y un mensaje de '✅ Proceso completado'.
        """
        return await self.client.show(self.model)

    async def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        output_model: Type[BaseModel],
        temperatura: float = 0.7,
        timeout_segundos: int = 900
    ) -> Optional[BaseModel]:
        """
        Método genérico para llamar al LLM, forzar JSON y validar la salida.
        """
        try:
            # El semáforo se toma FUERA del wait_for: el tiempo de cola no cuenta
            # contra el timeout, que así mide solo la generación real.
            async with self._sem:
                response = await asyncio.wait_for(
                    self.client.chat(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        format="json",
                        options={
                            "temperature": temperatura,
                            "num_ctx": self.num_ctx,
                            "num_predict": self.num_predict
                        }
                    ),
                    timeout=timeout_segundos
                )

            content = response["message"]["content"]

            if isinstance(content, dict):
                obj = output_model.model_validate(content)
            elif isinstance(content, str):
                obj = output_model.model_validate_json(content)
            else:
                raise ValueError("Tipo de respuesta inesperado del LLM")

            self.stats["ok"] += 1
            return obj

        except asyncio.TimeoutError:
            self.stats["timeout"] += 1
            print(f"\n⚠️ Timeout: El modelo superó los {timeout_segundos}s de generación. Descartando individuo.")
            return None
        except ValidationError:
            self.stats["invalid_json"] += 1
            # Capturamos el error silenciosamente sin manchar la consola con todo el log rojo
            print(f"\n⚠️ LLM devolvió un formato incorrecto para {output_model.__name__}. Descartando individuo.")
            return None
        except Exception as e:
            self.stats["error"] += 1
            print(f"\n❌ Error en LLMAgent.call_llm (Modelo: {output_model.__name__}): {e}")
            return None
