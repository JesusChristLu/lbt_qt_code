from typing import TYPE_CHECKING, Tuple, List

if TYPE_CHECKING:
    from freq_allocator.model_cpp.ChipError import ChipError
    from freq_allocator.model_cpp.InternalStateSingleQubit import InternalStateSingleQubit
    from freq_allocator.model_cpp.InternalStateTwoQubit import InternalStateTwoQubit

def single_err_model(chip : ChipError) -> Tuple[float, InternalStateSingleQubit]: ...
def loss_single_qubit(frequencies : List[float]) -> float: ...
def loss_single_qubit_on_range(ranges : List[float]) -> float: ...
def random_loss_single_qubit() -> float: ...
def random_allow_freq_loss_single_qubit() -> float: ...

def twoq_err_model(chip : ChipError) -> Tuple[float, InternalStateTwoQubit]: ...
def loss_two_qubit(frequencies : List[float]) -> float: ...
# from typing import List, Tuple
# from ChipNode import ChipNode

# class ChipError:
#     H : int
#     W : int
#     nodes : List[ChipNode]
#     n_available_nodes : int

#     xy_crosstalk : List[List[float]]
#     error_arr : List[List[List[float]]]
#     alpha_list : List[float]
#     mu_list : List[float]
#     detune_list : List[float]

#     def __init__(self) -> None: ...

#     def qubit_name_idx(i : int, j : int) -> int: ...
#     def qubit_idx(i : int, j : int) -> int: ...
#     def check_available_pair(i : int, j : int) -> bool: ...
#     def get_neighbors(i : int, j : int) -> List[Tuple[int, int]]: ...
#     def get_neighbors_distance_sqrt2(i : int, j : int) -> List[Tuple[int, int]]: ...
#     def get_neighbors_distance_2(i : int, j : int) -> List[Tuple[int, int]]: ...

#     def load_file(qubit_data_filename : str, 
#                   xy_crosstalk_sim_filename : str) -> None: ...

#     def initialize_all_nodes() -> None: ...
#     def list_all_unallocated() -> List[Tuple[int, int]]: ...
#     def list_all_allocated() -> List[Tuple[int, int]]: ...
#     def assign_frequencies(frequencies : List[float]) -> None : ...
#     def list_freq_ranges() -> List[Tuple[float, float]]: ...


# class ChipNode:
#     used : int
#     x : int
#     y : int
#     frequency : float
#     t_sq : float
#     anharm : float
#     sweet_point : float
#     allow_freq : List[float]
#     allow_freq_int : List[int]
#     ac_spectrum : List[float]
#     t1_freq : List[float]
#     t1_t1 : List[float]
#     xy_crosstalk : List[float]

#     def __init__(self) -> None: ...


# class InternalState:
#     pass


