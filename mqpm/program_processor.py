import numpy as np
from qiskit import QuantumCircuit
from copy import deepcopy

class Processor():
    def __init__(self):
        self.lookahead = 5
        self.queue = {}

    def schedule(self, run_information, occupied_precent_rounds, program_accumulate, programs_id, input_program, chip, time, ti):
        self.queue.update(input_program)
        program_accumulate.append(len(input_program))
        while len(self.queue) > 0:
            self.priority_management(time)
            mergeLookahead = len(self.queue)
            merge_program, origin_programs = self.merge(list(self.queue.values())[:mergeLookahead], chip)

            print('Trying the merged program with', len(origin_programs), 'programs,', merge_program[0].width() // 2, 'qubits.')
            partition = chip.find_partition(merge_program)
            print('We have find the partition.')
            out_program, ini_layout, relabel_map, recover_map, sub_prop, coupling_map = \
                chip.compilation(merge_program, partition)
            merged_program = merge_program[0]
            qubit_belong = merge_program[1]

            programs_timeline = [list(self.queue.keys())[list(self.queue.values()).index(pg)] for pg in origin_programs]
            program_id = [programs_id[list(input_program.values()).index(pg)] for pg in origin_programs]
            self.update_queue(origin_programs)
            run_inf, free_b_round = chip.run(program_id, programs_timeline, 
            origin_programs, merged_program, out_program, partition, ini_layout, qubit_belong, 
            relabel_map, recover_map, sub_prop, coupling_map, ti)
            run_information.append(run_inf)
            occupied_precent_rounds.append(free_b_round)
            if len(self.queue) == 0:
                break
        if len(self.queue) == 0:
            print('Finish the programs in the queue.\n')
        return run_information, occupied_precent_rounds, program_accumulate

    def merge(self, c_list, chip):
        print('Trying to merge the programs.')
        cs = [(c, {0 : list(range(c.width() // 2))}) for c in c_list]
        merge_program = cs[0]
        origin_programs = [cs[0][0]]
        for c2t in cs[1:]:
            c1, c2 = merge_program[0], c2t[0]
            if c1.width() // 2 + c2.width() // 2 <= len(chip.chip.nodes):
                c = QuantumCircuit(c1.width() // 2 + c2.width() // 2, c1.width() // 2 + c2.width() // 2)
                c.compose(c1, qubits=list(range(c1.width() // 2)), inplace=True)
                c.compose(c2, qubits=list(range(c1.width() // 2, c1.width() // 2 + c2.width() // 2)), inplace=True)
                qubit_belong = deepcopy(merge_program[1])
                qubit_belong.update({len(qubit_belong) : list(np.array(c2t[1][0]) + c1.width() // 2)})
                merge_program = ((c, qubit_belong))
                origin_programs.append(c2)
        print('We can run at most', len(origin_programs), 'programs together.')
        return merge_program, origin_programs

    def priority_management(self, time):
        print('Sorting the programs according to FCFS principle.')
        self.queue = dict(sorted(self.queue.items(), key=lambda item:(time - item[0]), reverse=True))

    def update_queue(self, run_program):
        for dp in run_program:
            del_value = list(self.queue.values()).index(dp)
            del_key = list(self.queue.keys())[del_value]
            del self.queue[del_key]