from copy import deepcopy
import time

import numpy as np
import freq_allocator
import scipy.optimize
from pathlib import Path
import random
import logging

logger = logging.getLogger()

def profiling_loss(test_loops = 100):
    model = freq_allocator.model_cpp.ChipModel(basepath=Path.cwd() / 'chipdata')
    frequencies = []

    for node in model.chip.nodes:
        if not node.used:
            continue
        freq = random.choice(node.allow_freq)
        frequencies.append(freq)
    
    t1 = time.time()
    for i in range(test_loops):
        loss = freq_allocator.model_cpp.loss(frequencies)
    t2 = time.time()
    print(f'model_cpp.loss {test_loops} times = {(t2-t1) * 1000} ms (avg. {(t2-t1)*1000/test_loops} ms)')

    t1 = time.time()
    for i in range(test_loops):
        ranges = np.random.rand(model.chip.n_available_nodes)
        loss = freq_allocator.model_cpp.loss_on_range(ranges)
    t2 = time.time()
    print(f'model_cpp.loss_on_range {test_loops} times = {(t2-t1) * 1000} ms (avg. {(t2-t1)*1000/test_loops} ms)')
    
    t1 = time.time()
    for i in range(test_loops):
        for node in model.chip.nodes:
            if not node.used:
                continue
            
            freq = random.choice(node.allow_freq)
            node.assign_frequency(freq)
        loss, _ = freq_allocator.model_cpp.single_err_model(model.chip, False)
    t2 = time.time()
    print(f'model_cpp.single_err_model (record_internal=False, assign_frequency) {test_loops} times = {(t2-t1) * 1000} ms (avg. {(t2-t1)*1000/test_loops} ms)')

    t1 = time.time()
    for i in range(test_loops):
        for node in model.chip.nodes:
            if not node.used:
                continue
            
            freq = random.choice(node.allow_freq)
            node.assign_frequency(freq)
        loss, _ = freq_allocator.model_cpp.single_err_model(model.chip, True)
    t2 = time.time()
    print(f'model_cpp.single_err_model (record_internal=True, assign_frequency) {test_loops} times = {(t2-t1) * 1000} ms (avg. {(t2-t1)*1000/test_loops} ms)')
        
    t1 = time.time()
    for i in range(test_loops):
        for node in model.chip.nodes:
            if not node.used:
                continue            
            freq = random.random()
            node.assign_frequency_on_range(freq)
        loss, _ = freq_allocator.model_cpp.single_err_model(model.chip, False)
    t2 = time.time()
    print(f'model_cpp.single_err_model (record_internal=False, assign_frequency_on_range) {test_loops} times = {(t2-t1) * 1000} ms (avg. {(t2-t1)*1000/test_loops} ms)')

    t1 = time.time()
    for i in range(test_loops):
        for node in model.chip.nodes:
            if not node.used:
                continue
            freq = random.random()
            node.assign_frequency_on_range(freq)
        loss, _ = freq_allocator.model_cpp.single_err_model(model.chip, True)
    t2 = time.time()
    print(f'model_cpp.single_err_model (record_internal=True, assign_frequency_on_range) {test_loops} times = {(t2-t1) * 1000} ms (avg. {(t2-t1)*1000/test_loops} ms)')

def example_assign_frequencies():
    model = freq_allocator.model_cpp.ChipModel(basepath=Path.cwd() / 'chipdata')
    frequencies = []
    for node in model.chip.nodes:
        if not node.used:
            continue

        freq = random.choice(node.allow_freq)
        frequencies.append(freq)

    print(frequencies)
    print(len(frequencies))
    print(model.n_available_nodes)

def callback_for_dual_annealing(x, f, context):
    # if context == 0:
    #     print(f'Minimum detected. f = {f}, x = {x}')
    # if context == 1:
    #     print(f'Local search detected. f = {f}, x = {x}')
    if context == 2:
        print(f'Dual annealing detected. f = {f}, x = {x}')

def example_optimizer(maxiter = 10000):
    model = freq_allocator.model_cpp.ChipModel(basepath=Path.cwd() / 'chipdata')


    ranges = model.list_freq_ranges()
    t1 = time.time()
    ret = scipy.optimize.dual_annealing(freq_allocator.model_cpp.loss, bounds=ranges, maxiter=maxiter, 
                                        callback = callback_for_dual_annealing)
    t2 = time.time()

    print(ret)
    print(f'Time = {t2-t1} s')

def example_optimizer_on_range(maxiter = 10000):
    model = freq_allocator.model_cpp.ChipModel(basepath=Path.cwd() / 'chipdata')
    ranges = [(0, 1) for i in range(model.n_available_nodes)]
    t1 = time.time()
    ret = scipy.optimize.dual_annealing(freq_allocator.model_cpp.loss, bounds=ranges, maxiter=maxiter, 
                                        callback = callback_for_dual_annealing)
    t2 = time.time()

    print(ret)
    print(f'Time = {t2-t1} s')

def example_optimizer2(maxiter = 10000):    
    def loss(frequencies):
        model.assign_frequencies(frequencies)
        loss, internal_state = freq_allocator.model_cpp.single_err_model(model.chip)
        
        return loss

    model = freq_allocator.model_cpp.ChipModel(basepath=Path.cwd() / 'chipdata')

    ranges = model.list_freq_ranges()
    t1 = time.time()
    ret = scipy.optimize.dual_annealing(loss, bounds=ranges, maxiter=maxiter, 
                                        callback = callback_for_dual_annealing)
    t2 = time.time()

    print(ret)
    print(f'Time = {t2-t1} s')

def example_optimizer2_on_range(maxiter = 10000):    
    def loss_on_range(ranges):
        model.assign_frequencies_with_ranges(ranges)
        loss, internal_state = freq_allocator.model_cpp.single_err_model(model.chip)
        
        return loss

    model = freq_allocator.model_cpp.ChipModel(basepath=Path.cwd() / 'chipdata')

    ranges = [(0, 1) for i in range(model.n_available_nodes)]
    t1 = time.time()
    ret = scipy.optimize.dual_annealing(loss_on_range, bounds=ranges, maxiter=maxiter, 
                                        callback = callback_for_dual_annealing)
    t2 = time.time()

    print(ret)
    print(f'Time = {t2-t1} s')

def test_loss():
    # frequencies = [4175.0, 4053.0, 4660.0, 4107.0, 4181.0, -1, 4094.0, 4618.0, 4146.0, 4978.0, 4252.0, 4799.0, 4658.0, 4000.0, 4514.0, 4075.0, 4634.0, 4147.0, 4302.0, 4098.0, 4188.0, -1, 4210.0, 4743.0, 4491.0, 4122.0, 4653.0, 4106.0, 4568.0, 4036.0, 4159.0, 4570.0, 4007.0, 4620.0, 4072.0, 4454.0, 4622.0, 4045.0, 4498.0, 4178.0, 4763.0, 4106.0, 4244.0, -1, 4343.0, 4804.0, 3998.0, 4391.0, 4178.0, 4055.0, 4699.0, -1, 4464.0, -1, 4087.0, 4578.0, 4419.0, 4394.0, -1, 4907.0, 4615.0, 4003.0, 4532.0, -1, 4991.0, 4078.0, 4204.0, 4663.0, 4085.0, 4808.0, 4227.0, 4642.0]

    # frequencies = [4175.0, 4053.0, 4660.0, 4107.0, 4181.0, 4094.0, 4618.0, 4146.0, 4978.0, 4252.0, 4799.0, 4658.0, 4000.0, 4514.0, 4075.0, 4634.0, 4147.0, 4302.0, 4098.0, 4188.0, 4210.0, 4743.0, 4491.0, 4122.0, 4653.0, 4106.0, 4568.0, 4036.0, 4159.0, 4570.0, 4007.0, 4620.0, 4072.0, 4454.0, 4622.0, 4045.0, 4498.0, 4178.0, 4763.0, 4106.0, 4244.0, 4343.0, 4804.0, 3998.0, 4391.0, 4178.0, 4055.0, 4699.0, 4464.0, 4087.0, 4578.0, 4419.0, 4394.0, 4907.0, 4615.0, 4003.0, 4532.0, 4991.0, 4078.0, 4204.0, 4663.0, 4085.0, 4808.0, 4227.0, 4642.0]

    model = freq_allocator.model_cpp.ChipModel(basepath=Path.cwd() / 'chipdata')

    print('------------PYTHON MODEL------------')
    import pickle
    with open(Path.cwd() /'results'/'2023-12-16'/'23.06.02' / 'epoch=47,chip_process.pickle', 'rb') as fp:
        obj = pickle.load(fp)
    print(obj)
    error_sum = 0
    frequencies = []
    isolated_error_list = []
    xtalk_error_list = []
    xtalk_error_dict = {}
    xtalk_error_sum_list = []
    nn_error_list = []
    nnn_error_list = []
    nn_error_sum_list = []
    nnn_error_sum_list = []
    error_list = []
    for node in obj.nodes:
        error_sum += obj.nodes[node]['error_all']
        frequencies.append(obj.nodes[node]['frequency'])
        # isolated_error_list.append(obj.nodes[node]['isolated_error'])
        xtalk_error_list.append(obj.nodes[node]['xy_crosstalk_error'])
        xtalk_error_dict[node] = obj.nodes[node]['xy_crosstalk_error']
        xtalk_error_sum_list.append(sum(obj.nodes[node]['xy_crosstalk_error'].values()))
        nn_error_list.append(obj.nodes[node]['NN_error'])
        nn_error_sum_list.append(sum(obj.nodes[node]['NN_error'].values()))
        nnn_error_list.append(obj.nodes[node]['NNN_error'])
        nnn_error_sum_list.append(sum(obj.nodes[node]['NNN_error'].values()))
        error_list.append(obj.nodes[node]['error_all'])

    print('Total Error = ', error_sum)
    print('Avg Error = ', error_sum / 65)
    # print(error_list)
    #print(isolated_error_list)
    # for i, xtalk in enumerate(xtalk_error_list):
    #     r = {j : k for j, k in xtalk.items() if k > 1e-6}
    #     xtalk_error_list[i] = r

    #print('Total Isolated Error (guess) = ', error_sum - sum(xtalk_error_sum_list) - sum(nn_error_sum_list) - sum(nnn_error_sum_list))
    # for node in xtalk_error_dict:
    #     print()
    #     print()
    #     print(f'---------{node}----------')
    #     print(xtalk_error_dict[node])
    #     print()
    #     print()
    print('Total XTalk Error = ', sum(xtalk_error_sum_list))
    #print('Total NN Error = ', sum(nn_error_sum_list))
    #print('Total NNN Error = ', sum(nnn_error_sum_list))
    print('------------CPP MODEL------------')
    
    print(frequencies)
    model.assign_frequencies(frequencies)
    loss, internal_state = freq_allocator.model_cpp.single_err_model(model.chip)

    print('Total Error = ', loss)
    # print(loss / model.chip.n_available_nodes)
    print('Avg Error = ', loss / 65)
    # print(internal_state.qubit_err_list)
    # print(sum(internal_state.qubit_err_list) / 65)
    #print(internal_state.isolated_err_list)
    # xtalk_error_list = deepcopy(internal_state.XTalk_err_list)
    # for i, xtalk in enumerate(xtalk_error_list):
    #     r = {j : k for j, k in xtalk.items() if k > 1e-6}
    #     xtalk_error_list[i] = r
    #print(internal_state.XTalk_err_list)
    #print('Total Isolated Error = ', internal_state.isolated_err)
    print('Total XTalk Error = ', internal_state.XTalk_err)
    #print('Total NN Error = ', sum([sum(NN_residual_err.values()) for NN_residual_err in internal_state.NN_residual_err_list]))
    #print('Total NNN Error = ', sum([sum(NNN_residual_err.values()) for NNN_residual_err in internal_state.NNN_residual_err_list]))

if __name__ == '__main__':    
    # profiling_loss(test_loops=1000)
    # test_loss()
    example_optimizer(maxiter=200000)
    # example_optimizer_on_range()
