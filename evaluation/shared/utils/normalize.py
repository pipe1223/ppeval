from __future__ import annotations

from typing import Iterable, List


def normalize_id_list(values: Iterable[object]) -> List[str]:
    return [str(v).strip() for v in values]
