from typing import List

class ChipCoupler:
    used : int
    qubit1 : int
    qubit2 : int
    coupler_id : int
    neighbor_couplers : List[int]
    frequency : float
    t_twoq : float

    def __init__(self) -> None: ...
    def assign_frequency(self, frequency : float) -> None: ...


