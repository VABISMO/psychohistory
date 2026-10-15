"""
Mecanismos de control de carga para un recurso de inferencia unico (Ollama en CPU).

- Bottleneck: serializa las llamadas al LLM (concurrencia limitada) y respeta un
  intervalo minimo entre llamadas para no saturar la CPU ni provocar contención.
- RateLimiter: ventana deslizante por cliente para limitar peticiones a la API.
- with_retries: estrategia de reintentos con backoff exponencial.
"""
import asyncio
import time
from collections import defaultdict, deque
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class Bottleneck:
    """Limita concurrencia + tasa de acceso a un recurso unico (el modelo)."""

    def __init__(self, concurrency: int = 1, min_interval: float = 0.0):
        self._sem = asyncio.Semaphore(concurrency)
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self._sem.acquire()
        async with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()
        return self

    async def __aexit__(self, *_):
        self._sem.release()


class RateLimiter:
    """Ventana deslizante por clave (p.ej. IP). Devuelve False si se excede el limite."""

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self._hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> bool:
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > self.period:
            q.popleft()
        if len(q) >= self.max_calls:
            return False
        q.append(now)
        return True

    def retry_after(self, key: str) -> float:
        q = self._hits[key]
        if not q:
            return 0.0
        return max(0.0, self.period - (time.monotonic() - q[0]))


# Recurso unico: solo una inferencia a la vez, con separacion minima entre llamadas.
ollama_bottleneck = Bottleneck(concurrency=1, min_interval=0.5)

# Limites de la API
predict_limiter = RateLimiter(max_calls=6, period=60.0)  # 6 predicciones/min por IP
general_limiter = RateLimiter(max_calls=120, period=60.0)  # 120 req/min por IP


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    attempts: int = 3,
    base_delay: float = 2.0,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """Ejecuta fn() con backoff exponencial. Relanza el ultimo error si falla todo."""
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return await fn()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if on_retry:
                on_retry(i + 1, e)
            if i < attempts - 1:
                await asyncio.sleep(base_delay * (2 ** i))
    assert last_exc is not None
    raise last_exc
