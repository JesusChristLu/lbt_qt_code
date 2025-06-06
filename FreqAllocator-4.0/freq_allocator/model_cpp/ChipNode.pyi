from typing import List

class ChipNode:
    used : int
    x : int
    y : int
    frequency : float
    t_sq : float
    anharm : float
    sweet_point : float
    allow_freq : List[float]
    allow_freq_int : List[int]
    ac_spectrum : List[float]
    t1_freq : List[float]
    t1_t1 : List[float]
    xy_crosstalk : List[float]

    def __init__(self) -> None: ...


