import FreqAllocatorCpp
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    from .ChipError import ChipError
    from .ChipQubit import ChipQubit
    from .ChipCoupler import ChipCoupler
else:
    class ChipError:...
    class ChipQubit:...
    class ChipCoupler:...

class ChipModel:    
    def __init__(self, basepath):
        self.chip_inst = FreqAllocatorCpp.FrequencyAllocator.get_chip()
        self.chip_inst.load_file(
            str(Path(basepath) / 'qubit_data.json'), 
            str(Path(basepath) / 'xy_crosstalk_sim.json'))
    
    @property
    def H(self) -> int:
        '''Height of the chip (12)

        Returns:
            int: Height of the chip (12)
        '''
        return self.chip.H

    @property
    def W(self) -> int:
        '''Width of the chip (6)

        Returns:
            int: With of the chip (6)
        '''
        return self.chip.W
    
    @property
    def chip(self) -> ChipError:
        '''ChipError object itself (see definition in C++)

        Returns:
            ChipError: Export from C++ ChipError class.
        '''
        return self.chip_inst
    
    @property
    def n_available_qubits(self) -> int:
        '''Get number of available qubits

        Returns:
            int: number of available couplers (without unused qubits)
        '''
        return self.get_n_available_qubits()
    
    @property
    def n_available_couplers(self) -> int:
        '''Get number of available couplers

        Returns:
            int: number of available couplers (without broken edges)
        '''
        return self.get_n_available_couplers()
    
    @property
    def qubits(self) -> List[ChipQubit]:
        '''Get all qubits in List[ChipQubit] object.

        Returns:
            List[ChipQubit]: Container of ChipQubit, indexed by qubit id (from 0 to 71).
        '''
        return self.get_qubits()
    
    @property
    def couplers(self) -> List[ChipCoupler]:
        '''Get all couplers in List[ChipCoupler] object.

        Returns:
            List[ChipCoupler]: Container of ChipCoupler, indexed by qubit id (from 0 to 125).
        '''
        return self.get_couplers()
    
    @property
    def allocated_qubits(self) -> List[Tuple[int, int]]:
        '''List all allocated qubits (except not-allocated or unused). Not important.

        Returns:
            List[Tuple[int, int]]: Each qubit provided in (x,y)
        '''
        return self.list_all_allocated_qubits()
        
    @property
    def unallocated_qubits(self) -> List[Tuple[int, int]]:        
        '''List all unallocated qubits (except allocated or unused). Not important.

        Returns:
            List[Tuple[int, int]]: Each qubit provided in (x,y)
        '''
        return self.list_all_unallocated_qubits()
    
    @property
    def allocated_couplers(self) -> List[int]:
        '''List all allocated couplers (except not-allocated or unused). Not important.

        Returns:
            List[int]: Each coupler provided in its index (0~125)
        '''
        return self.list_all_allocated_couplers()
        
    @property
    def unallocated_couplers(self) -> List[int]:
        '''List all allocated couplers (except allocated or unused). Not important.

        Returns:
            List[int]: Each coupler provided in its index (0~125)
        '''
        return self.list_all_unallocated_couplers()
    
    def get_qubits(self) -> List[ChipQubit]:   
        '''Same as qubits, not important.
        ''' 
        return self.chip.qubits
            
    def get_n_available_qubits(self) -> int:
        '''Same as n_available_qubits, not important.
        ''' 
        return self.chip.n_available_qubits
            
    def get_couplers(self) -> List[ChipCoupler]:
        '''Same as couplers, not important.
        ''' 
        return self.chip.couplers
            
    def get_n_available_couplers(self) -> int:
        '''Same as n_available_couplers, not important.
        ''' 
        return self.chip.n_available_couplers
    
    def qubit_name_idx(self, x : int, y : int) -> int:
        '''Obtain qubit name index from x,y, which is from 1~72. Computed by x*W+y+1

        Args:
            x (int): from 0~11 (H-1)
            y (int): from 0~5 (W-1)

        Returns:
            int: qubit name index.
        '''
        return self.chip.qubit_name_idx(x, y)
    
    def qubit_idx(self, x : int, y : int) -> int:
        '''Obtain qubit name index from x,y, which is from 0~71. Computed by x*W+y

        Args:
            x (int): from 0~11 (H-1)
            y (int): from 0~5 (W-1)

        Returns:
            int: qubit index.
        '''
        return self.chip.qubit_idx(x, y)
    
    def check_available_pair(self, x : int, y : int) -> bool:        
        '''Check if (x,y) pair is a valid pair for this chip

        Args:
            x (int): Should be from 0~11 (H-1). If not, return false.
            y (int): Should be from 0~5 (W-1). If not, return false.

        Returns:
            bool: Whether x and y is in their valid ranges.
        '''
        return self.chip.check_available_pair(x, y)
    
    def get_neighbors(self, x, y) -> List[Tuple[int, int]]:
        '''Get all neighbor positions from (x,y), and return in List of (x',y').
        We have abs(x-x') + abs(y-y') == 1

        Args:
            x (int): from 0~11 (H-1)
            y (int): from 0~5 (W-1)

        Returns:
            List[Tuple[int, int]]: A list of positions (x',y').
        '''
        return self.chip.get_neighbors(x, y)
        
    def get_neighbors_distance_sqrt2(self, x, y) -> List[Tuple[int, int]]:        
        '''Get all distance sqrt2 positions from (x,y), and return in List of (x',y').
        We have abs(x-x') == 1 and abs(y-y') == 1

        Args:
            x (int): from 0~11 (H-1)
            y (int): from 0~5 (W-1)

        Returns:
            List[Tuple[int, int]]: A list of positions (x',y') .
        '''
        return self.chip.get_neighbors_distance_sqrt2(x, y)
        
    def get_neighbors_distance_2(self, x, y) -> List[Tuple[int, int]]:  
        '''Get all distance 2 positions from (x,y), and return in List of (x',y').
        We have abs(x-x') == 2 and abs(y-y') == 0 or abs(x-x') == 0 and abs(y-y') == 2

        Args:
            x (int): from 0~11 (H-1)
            y (int): from 0~5 (W-1)

        Returns:
            List[Tuple[int, int]]: A list of positions (x',y') .
        '''
        return self.chip.get_neighbors_distance_2(x, y)
        
    def from_qubits_to_coupler_idx(self, q1 : int, q2 : int) -> int:
        '''Get coupler index from two qubits' index

        Args:
            q1 (int): qubit 1
            q2 (int): qubit 2

        Returns:
            int: coupler index
        '''
        return self.chip.from_qubits_to_coupler_idx(q1, q2)
    
    def from_qubit_pos_to_coupler_idx(self, q1 : Tuple[int, int], q2 : Tuple[int, int]) -> int:
        '''Get coupler index from two qubits' positions

        Args:
            q1 (Tuple[int, int]): qubit 1 (x,y)
            q2 (Tuple[int, int]): qubit 2 (x,y)

        Returns:
            int: coupler index
        '''
        return self.chip.from_qubit_pos_to_coupler_idx((q1[0], q1[1]), (q2[0], q2[1]))
    
    def from_coupler_idx_to_qubits(self, coupler_idx : int) -> Tuple[int, int]:
        '''Get two qubits from coupler's index.

        Args:
            coupler_idx (int): coupler's index

        Returns:
            Tuple[int, int]: two qubits' index (such as (70,71))
        '''
        return self.chip.from_coupler_idx_to_qubits(coupler_idx)

    def get_neighbor_couplers(self, coupler_id: int) -> List[int]:
        '''Get coupler's neighbor coupler (not important)

        Args:
            coupler_id (int): coupler's index

        Returns:
            List[int]: neighbor couplers
        '''
        return self.chip.get_neighbor_couplers(coupler_id)

    def initialize_all_qubits(self) -> None:
        '''Set all qubits to frequency -1 
        '''
        return self.chip.initialize_all_qubits()
    
    def list_all_unallocated_qubits(self) -> List[Tuple[int, int]]:
        '''Same as unallocated_qubits, not important.
        '''
        return self.chip.list_all_unallocated()
    
    def list_all_allocated(self) -> List[Tuple[int, int]]:
        '''Same as allocated_qubits, not important.
        '''
        return self.chip.list_all_allocated()
    
    def assign_qubit_frequencies(self, frequencies : List[float]) -> None:
        '''Assign qubits frequencies with a list. Note that len(frequencies) should be just as all_available_qubits (must be less than 72). Support partial allocation, just set the frequency to -1 to mark it as "unallocated".

        Args:
            frequencies (List[float]): the input frequencies.
        '''
        return self.chip.assign_qubit_frequencies(frequencies)
    
    def assign_qubit_frequencies_full(self, frequencies) -> None:
        '''Assign qubits frequencies with a list. Note that len(frequencies) should be 72. Support partial allocation, just set the frequency to -1 to mark it as "unallocated".

        Args:
            frequencies (List[float]): the input frequencies.
        '''
        return self.chip.assign_qubit_frequencies_full(frequencies)
    
    def assign_qubit_frequencies_by_idx_dict(self, frequency_dict : Dict[int, float]) -> None:
        '''Assign qubits frequencies with a dict where each key is the index (from 0~71). Support partial allocation, only the qubits in the dict will be considered.

        Example:       
            chip.assign_qubit_frequencies_by_idx_dict({18: 4131, 20: 4027, ...})
        
        Args:
            frequency_dict (Dict[int, float]): the input frequencies
        '''
        return self.chip.assign_qubit_frequencies_by_idx_dict(frequency_dict)
    
    def assign_qubit_frequencies_by_pair_dict(self, frequency_dict : Dict[Tuple[int, int], float]) -> None:        
        '''Assign qubits frequencies with a dict where each key is the position (x,y). Support partial allocation, only the qubits in the dict will be considered.

        Example:       
            chip.assign_qubit_frequencies_by_idx_dict({(2,5): 4131, (3,7): 4027, ...})
        
        Args:
            frequency_dict (Dict[int, float]): the input frequencies
        '''
        return self.chip.assign_qubit_frequencies_by_pair_dict(frequency_dict)

    def assign_qubit_frequencies_by_dict(self, frequency_dict: Union[Dict[int, float], Dict[Tuple[int, int], float]]) -> None:
        '''Assign qubits frequencies with a dict where each key is the qubit index (0~71) or its position (x,y). 
        
        Note:
            Auto select from assign_qubit_frequencies_by_idx_dict or assign_qubit_frequencies_by_pair_dict.

        Args:
            frequency_dict (Union[Dict[int, float], Dict[Tuple[int, int], float]]): the input frequencies

        Raises:
            TypeError: Invalid input types.
        '''
        if isinstance(frequency_dict, dict):
            for key in frequency_dict:
                if isinstance(key, int):
                    # Dict[int, float]
                    self.assign_qubit_frequencies_by_idx_dict(frequency_dict)
                    return
                elif isinstance(key, tuple):
                    self.assign_qubit_frequencies_by_pair_dict(frequency_dict)
                    return

                break

        raise TypeError('frequency_dict is either Dict[int, float] '
                        'where each key indicates a qubit idx (from 0~71), '
                        'or Dict[Tuple[int, int], float] where each key '
                        'indicates a qubit position pair (like (11, 5)).')
        
    def list_freq_ranges(self) -> List[Tuple[float, float]]:
        '''List qubit's allow_freq ranges (from RB spectrum, not to be freq_max&freq_min).

        Returns:
            List[Tuple[float, float]]: Frequencies ranges.
        '''
        return self.chip.list_qubit_freq_ranges()
    
    def initialize_all_couplers(self) -> None:
        '''Set all couplers to frequency -1 
        '''
        return self.chip.initialize_all_couplers()

    def list_all_unallocated_couplers(self) -> None:
        '''Same as unallocated_couplers, not important.
        '''
        return self.chip.list_all_unallocated_couplers()

    def list_all_allocated_couplers(self) -> None:
        '''Same as allocated_couplers, not important.
        '''
        return self.chip.list_all_allocated_couplers()

    def assign_coupler_frequencies(self, frequencies : List[float]) -> None:
        '''Assign couplers frequencies with a list (just like assign_qubit_frequencies_full). Note that len(frequencies) should be 126. Support partial allocation, just set the frequency to -1 to mark it as "unallocated". 

        Args:
            frequencies (List[float]): the input frequencies.
        '''
        return self.chip.assign_coupler_frequencies(frequencies)

    def assign_coupler_frequencies_by_idx_dict(self, frequency_dict : Dict[int, float]) -> None:        
        '''Assign couplers frequencies with a dict where each key is the index (from 0~125). Support partial allocation, only the couplers in the dict will be considered.

        Example:       
            chip.assign_coupler_frequencies_by_idx_dict({18: 4131, 20: 4027, ...})
        
        Args:
            frequency_dict (Dict[int, float]): the input frequencies
        '''
        return self.chip.assign_coupler_frequencies_by_idx_dict(frequency_dict)

    def assign_coupler_frequencies_by_pair_dict(self, 
        frequency_dict : Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]) -> None:
        '''Assign couplers frequencies with a dict where each key is the two qubit's position (such as ((0,0), (0,1)), (1,0), (1,1))). Support partial allocation, only the couplers in the dict will be considered.

        Example:       
            chip.assign_coupler_frequencies_by_idx_dict({18: 4131, 20: 4027, ...})
        
        Args:
            frequency_dict (Dict[int, float]): the input frequencies
        '''

        return self.chip.assign_coupler_frequencies_by_pair_dict(frequency_dict)
    
    def assign_coupler_frequencies_by_dict(self, 
        frequency_dict: Union[Dict[int, float], 
                              Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]]
    ) -> None:
        '''Assign couplers frequencies with a dict where each key is the index (from 0~125), or each key is the two qubit's position (such as ((0,0), (0,1)), (1,0), (1,1))). (Auto select just like assign_qubit_frequencies_by_dict). Support partial allocation, only the couplers in the dict will be considered.

        Note:
            Auto select from assign_coupler_frequencies_by_idx_dict or assign_coupler_frequencies_by_pair_dict.

        Args:
            frequency_dict (Union[Dict[int, float], Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]): the input frequencies

        Raises:
            TypeError: Invalid input types.
        '''
        if isinstance(frequency_dict, dict):
            for key in frequency_dict:
                if isinstance(key, int):
                    # Dict[int, float]
                    self.assign_coupler_frequencies_by_idx_dict(frequency_dict)
                    return
                elif isinstance(key, tuple):
                    self.assign_coupler_frequencies_by_pair_dict(frequency_dict)
                    return

                break

        raise TypeError('frequency_dict is either Dict[int, float] '
                        'where each key indicates a qubit idx (from 0~125), '
                        'or Dict[Tuple[int, int], float] where each key '
                        'indicates a qubit position pair (like (11, 5)).')