from typing import List, Optional, Dict, Tuple, Iterable, Callable


def total(xs: List[float]) -> float:
    return sum(xs)

values = []
best_so_far = None

values: List[int] = []
best_so_far: Optional[float] = None

counts: Dict[str, int] = {"data": 1, "science": 2}

lazy = None
if lazy:
    events: Iterable[int] = (x for x in range(10) if x % 2 == 0)
else:
    events = [0, 2, 4, 6, 8]

triple: Tuple[int, float, int] = (10, 2,3, 5)

def twice(repeater: Callable[[str, int], str], s: str) -> str:
    return repeater(s, 2)

def comma_repeater(s: str, n: int) -> str:
    n_copies = [s for _ in range(n)]
    return ", ".join(n_copies)

assert twice(comma_repeater, "type hints") == "type hints, type hints"

Number = int
Numbers = List[Number]

def total(xs: Numbers) -> float:
    return sum(xs)