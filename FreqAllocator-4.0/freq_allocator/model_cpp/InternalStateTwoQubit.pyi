from typing import List

class InternalStateTwoQubit:
    T1_err : float
    T2_err : float
    pulse_distortion_err : float
    XTalk_spectator_err : float    
    XTalk_parallel_err : float    
    inner_leakage_err : float    

    coupler_err_list : List[float]
    T1_err_list : List[float]
    T2_err_list : List[float]
    pulse_distortion_err_list : List[float]
    XTalk_spectator_err_list : List[float]
    XTalk_parallel_err_list : List[float]
    inner_leakage_err_list : List[float]
    
    def __init__(self) -> None: ...

    def to_string(self) -> str: ...

    


