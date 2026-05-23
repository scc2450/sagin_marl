from __future__ import annotations

import gzip
import os
import pickle
from pathlib import Path
from typing import Any


def load_runtime_state_bank_payload(path: str | os.PathLike[str]) -> Any:
    bank_path = Path(path)
    opener = gzip.open if str(bank_path).endswith(".gz") else open
    with opener(bank_path, "rb") as f:
        return pickle.load(f)


def extract_runtime_state_bank_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        entries = payload["entries"]
    else:
        raise TypeError("Runtime-state bank payload must be a list or a dict with an 'entries' list.")
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("Each runtime-state bank entry must be a dictionary.")
        normalized.append(entry)
    return normalized


def save_runtime_state_bank_payload(path: str | os.PathLike[str], payload: Any) -> None:
    bank_path = Path(path)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(bank_path).endswith(".gz") else open
    with opener(bank_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
