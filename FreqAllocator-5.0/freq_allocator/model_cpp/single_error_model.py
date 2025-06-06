from .chip_model import *
from typing import Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .ChipError import ChipError
    from .InternalStateSingleQubit import InternalStateSingleQubit
    from freq_allocator.stubs import FreqAllocatorCpp
else:
    class InternalStateSingleQubit: ...

def single_err_model(chip : ChipError, 
                     arb : List[float] = [2e-4, 1e-7, 1, 0.3, 10, 1e-2, 0.5, 10], 
                     record_internal_state : bool = False) -> Tuple[float, InternalStateSingleQubit]:
    return FreqAllocatorCpp.single_err_model(chip, arb, record_internal_state)

def loss_single_qubit(frequencies : List[float]) -> float:
    return FreqAllocatorCpp.loss_single_qubit(frequencies)

def loss_on_range_single_qubit(ranges : List[float]) -> float:
    return FreqAllocatorCpp.loss_on_range_single_qubit(ranges)

def random_loss_single_qubit() -> float:
    return FreqAllocatorCpp.random_loss_single_qubit()

def random_allow_freq_loss_single_qubit() -> float:    
    return FreqAllocatorCpp.random_allow_freq_loss_single_qubit()
