from typing import List

class InternalState:
    T1_err : float
    T2_err : float
    XTalk_err : float
    ZZ_err : float
    allocate_fail_err : float

    T1_err_list : List[float]
    T2_err_list : List[float]
    XTalk_err_list : List[float]
    ZZ_err_list : List[float]
    allocate_fail_err_list : List[float]
    
    def __init__(self) -> None: ...

    


