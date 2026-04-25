import os
from contextlib import contextmanager

LLMOBS_ENABLED = os.environ.get("LLMOBS_ENABLED", "false").lower() == "true"

_obs = None

if LLMOBS_ENABLED:
    from llmobs import LLMObs

    _obs = LLMObs(
        app=os.environ.get("LLMOBS_APP", "chat-agent"),
        endpoint=os.environ.get("LLMOBS_ENDPOINT", "http://localhost:8000"),
        api_key=os.environ.get("LLMOBS_API_KEY", ""),
    )


def get_obs():
    return _obs


@contextmanager
def span(kind, name, **kwargs):
    if _obs is None:
        yield None
        return
    method = getattr(_obs, kind)
    with method(name=name, **kwargs) as s:
        yield s


def shutdown():
    if _obs is not None:
        _obs.flush()
        _obs.shutdown()
