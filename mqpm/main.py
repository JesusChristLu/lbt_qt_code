import os
from quantum_chip import Quantum_chip
from time import time
from program_processor import Processor
from experiment import Simulator
import numpy as np
from qiskit.converters import circuit_to_dag
from qiskit.converters.dag_to_circuit import dag_to_circuit
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import passes

def random_generator(qubit_num, coherence_time, chip):
    bit_number = np.random.randint(2, max(3, qubit_num // 5))
    while 1: 
        depth = np.random.randint(2, 200)
        circ = random_circuit(bit_number, depth, measure=True)
        circ = circuit_to_dag(circ)
        remove_node = []
        for node in circ.gate_nodes():
            if node.name == 'id':
                remove_node.append(node)
        for node in remove_node:
            circ.remove_op_node(node)
        unroller = passes.Unroller(['u1', 'u3', 'cx'])
        circ = unroller.run(circ)
        if not 'cx' in circ.count_ops():
            continue
        unitary_evolution_time = chip.get_unitary_evolution_time(circ)
        if unitary_evolution_time < coherence_time and bit_number <= qubit_num ** 2:
            break
    circ = dag_to_circuit(circ)
    return circ, unitary_evolution_time, circuit_to_dag(circ).count_ops(), bit_number

def poisson_sim(total_time, rate):
    t = 0
    event_number = 0
    time_line = []
    while t < total_time:
        t += -1 / rate * np.log(np.random.random())
        event_number += 1
        time_line.append(t * 60)
    return time_line, event_number

if __name__=="__main__":
    try:
        height = 4
        width = 4
        qubit_num = height * width
        test_time = 0.2
        frequency = 60
        chip_size = [height, width]
        simulate = True
        xtalk_distance = 2
        chip = Quantum_chip(chip_size, xtalk_distance=xtalk_distance)
        time_line, event_number = poisson_sim(test_time, frequency)
        program_list = []
        run_information = []
        occupied_qubit_round = []
        program_accumulate = []
        coherence_time = chip.get_avg_coherence_time()
        print('Because we are not in real situation, we have to generate the random circuits at the beginning.')
        for en in range(event_number):
            program, evolution_time, gate_number, bit_number = \
                random_generator(qubit_num, coherence_time, chip) 
            print('program', en, 'of', event_number, 'input time', "{:.2f}".format(time_line[en]),
                    'evolution time', "{:.0f}".format(evolution_time *1e9), 'ns',
                    'gate number', gate_number, 'bit number', bit_number, '.')
            program_list.append(program)
        processor = Processor()
        test_number = 0
        old_test_number = test_number
        ti = time()
        while test_number < event_number:
            tf = time()
            while tf - ti < 20:
                tf = time()
            while test_number < len(time_line) and tf - ti > time_line[test_number]:
                test_number += 1
            if old_test_number < test_number:
                run_information, occupied_qubit_round, program_accumulate = \
                processor.schedule(run_information, occupied_qubit_round, program_accumulate, list(range(old_test_number, test_number)),
                                dict(zip(time_line[old_test_number : test_number], program_list[old_test_number : test_number])),
                                chip, (tf - ti), ti)
            old_test_number = test_number

        if simulate:
            print('simulation')
            for run_inf in run_information:
                print('simulating new programs my')
                simulator = Simulator(qubit_belong=run_inf['qubit belong'])
                my_fidelities = simulator.forward(run_inf['out program'], chip, run_inf['sub prop'], run_inf['layout'], run_inf['coupling map'], chip.xtalk_matrix)
                print(my_fidelities) 

        os.system('pause')
    except BaseException as e:
        print('error')
        print(e)
        raise
    finally:
        os.system('pause')

