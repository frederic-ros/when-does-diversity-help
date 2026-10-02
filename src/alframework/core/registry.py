from __future__ import annotations
from typing import Callable, Dict, Type, Any

STRATEGIES: Dict[str, Type[Any]] = {}

def register(name: str) -> Callable[[Type[Any]], Type[Any]]:
    def _decorator(cls: Type[Any]) -> Type[Any]:
        STRATEGIES[name] = cls
        cls.name = name
        return cls
    return _decorator
