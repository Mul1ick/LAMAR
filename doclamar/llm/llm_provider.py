from .base import BaseLLM
from .hf_llm import HFLocalLLM

_llm_instance: BaseLLM | None = None


def get_llm() -> BaseLLM:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = HFLocalLLM()
    return _llm_instance
