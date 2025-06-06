from .chip_model import *
from typing import Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .ChipError import ChipError
    from .InternalStateTwoQubit import InternalStateTwoQubit
    from freq_allocator.stubs import FreqAllocatorCpp
else:
    class InternalStateTwoQubit: ...

def twoq_err_model(chip : ChipError, 
                   axeb : List[float] = [4e-4, 1e-7, 1e-2, 1e-5, 1e-2, 1, 10, 0.7, 10], 
                   record_internal_state : bool = False) -> Tuple[float, InternalStateTwoQubit]:
    return FreqAllocatorCpp.twoq_err_model(chip, axeb, record_internal_state)

def loss_two_qubit(frequencies : List[float]) -> float:
    return FreqAllocatorCpp.loss_two_qubit(frequencies)
