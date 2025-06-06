from copy import deepcopy
import json
from pathlib import Path
import pickle
from random import random
import time
import numpy as np
import freq_allocator


def make_pattern_1():
    # from 0,2,4, then add 11
    pattern = []
    for i in range(6):
        pattern.append(0 + i * 22)
        pattern.append(2 + i * 22)
        pattern.append(4 + i * 22)
        pattern.append(12 + i * 22)
        pattern.append(14 + i * 22)

    pattern = list(filter(lambda x: x<126, pattern))
    return pattern

def make_pattern_2():
    # from 1,3 then add 11
    pattern = []
    for i in range(6):
        pattern.append(1 + i * 22)
        pattern.append(3 + i * 22)
        pattern.append(11 + i * 22)
        pattern.append(13 + i * 22)
        pattern.append(15 + i * 22)

    pattern = list(filter(lambda x: x<126, pattern))
    return pattern

def make_pattern_3():
    # from 5,6,7,8,9,10 then add 22
    pattern = []
    for i in range(6):
        pattern.append(5 + i * 22)
        pattern.append(7 + i * 22)
        pattern.append(9 + i * 22)
        pattern.append(17 + i * 22)
        pattern.append(19 + i * 22)
        pattern.append(21 + i * 22)

    pattern = list(filter(lambda x: x<126, pattern))
    return pattern

def make_pattern_4():
    # from 16,17,18,19,20 then add 22
    pattern = []
    for i in range(6):
        pattern.append(6 + i * 22)
        pattern.append(8 + i * 22)
        pattern.append(10 + i * 22)
        pattern.append(16 + i * 22)
        pattern.append(18 + i * 22)
        pattern.append(20 + i * 22)
    pattern = list(filter(lambda x: x<126, pattern))
    return pattern

def make_patterns():
    pattern1 = make_pattern_1()
    pattern2 = make_pattern_2()
    pattern3 = make_pattern_3()
    pattern4 = make_pattern_4()

    print(pattern1)
    print(pattern2)
    print(pattern3)
    print(pattern4)
    return pattern1, pattern2, pattern3, pattern4


def load_preset_qubit_frequencies(filename = Path.cwd() / 'chipdata' / 'qubit_freq_real.json'):
    with open(filename) as fp:
        data = json.load(fp)
    qubit_frequencies = [-1]*72;    
    for qname in data:
        qubit_frequencies[int(qname[1:]) - 1] = data[qname]

    return qubit_frequencies


def test_all_error():
    model = freq_allocator.model_cpp.ChipModel(basepath=Path.cwd() / 'chipdata')

    with open(Path.cwd() / 'test' / 'allocate_two_qubit' / 'chip_final.pickle', 'rb') as fp:
        chip_final = pickle.load(fp)

    chip_frequency_and_error = {}
    for node in chip_final.nodes():        
        q1 = model.qubit_idx(node[0], node[1])
        for node2 in chip_final[node]:
            q2 = model.qubit_idx(node2[0], node2[1])
            coupler_id = model.from_qubits_to_coupler_idx(q1, q2)
            chip_frequency_and_error[str(coupler_id)] = {}
            chip_frequency_and_error[str(coupler_id)]['frequency'] = float(chip_final[node][node2]['frequency'])
            chip_frequency_and_error[str(coupler_id)]['error_all'] = float(chip_final[node][node2]['error_all'])

    print(chip_frequency_and_error)
    print(json.dumps(chip_frequency_and_error))
    patterns = make_patterns()    

    pattern_chip_frequency_and_error = [{}, {}, {}, {}]

    for coupler_id in chip_frequency_and_error:
        for i, pattern in enumerate(patterns):
            if int(coupler_id) in pattern:
                pattern_chip_frequency_and_error[i][coupler_id] = chip_frequency_and_error[coupler_id]

    for i, chip_frequency_and_error in enumerate(pattern_chip_frequency_and_error):
        print(f'---------- Pattern {i} ----------')
        # print(chip_frequency_and_error)
        print(json.dumps(chip_frequency_and_error))

    qubit_frequencies = load_preset_qubit_frequencies(Path.cwd() / 'chipdata' / 'qubit_freq_real.json')
    model.assign_qubit_frequencies_full(qubit_frequencies)
    
    loss, internal_1q = freq_allocator.model_cpp.single_err_model(
        model.chip, 
        arb= [2e-4, 1e-7, 1, 0.3, 10, 1e-2, 0.5, 10], 
        record_internal_state=True)
    
    print('Total 1q loss (without averaging): ', loss)
    print('Total error (for each qubit): ', internal_1q.qubit_err_list)
    print('isolated_err_list: ', internal_1q.isolated_err_list)
    print('XTalk_err_list: ', internal_1q.XTalk_err_list)
    print('NN_residual_err_list: ', internal_1q.NN_residual_err_list)
    print('NNN_residual_err_list: ', internal_1q.NNN_residual_err_list)
    print('allocate_fail_err_list: ', internal_1q.allocate_fail_err_list)
    
    for i, chip_freq_and_err in enumerate(pattern_chip_frequency_and_error):
        coupler_frequencies = {}
        print(f'---------- Handling Pattern {i} ----------')
        for qubit_idx in chip_freq_and_err:
            coupler_frequencies[int(qubit_idx)] = chip_freq_and_err[qubit_idx]['frequency']
        print(len(chip_freq_and_err))
        print(coupler_frequencies)
        print(len(coupler_frequencies))
        model.assign_coupler_frequencies_by_dict(coupler_frequencies)

        loss, internal_2q = freq_allocator.model_cpp.twoq_err_model(
            model.chip, 
            axeb= [4e-4, 1e-7, 1e-2, 1e-5, 1e-2, 1, 10, 0.7, 10], 
            record_internal_state=True)

        print('Total loss (without averaging): ', loss)
        print('Total error (for each coupler): ', internal_2q.coupler_err_list)
        print('T1_err_list: ', internal_2q.T1_err_list)
        print('T2_err_list: ', internal_2q.T2_err_list)
        print('pulse_distortion_err_list: ', internal_2q.pulse_distortion_err_list)
        print('XTalk_spectator_err_list: ', internal_2q.XTalk_spectator_err_list)
        print('XTalk_parallel_err_list: ', internal_2q.XTalk_parallel_err_list)


if __name__ == '__main__':
    test_all_error()

