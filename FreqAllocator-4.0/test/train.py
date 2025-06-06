from pathlib import Path
import numpy as np
import freq_allocator


H = 12
W = 6

xy_crosstalk_sim_path = Path.cwd() / 'chipdata' / r"xy_crosstalk_sim.json"
freq_data = Path.cwd() / 'chipdata' / r"qubit_freq_real.json"
qubit_data = Path.cwd() / 'chipdata' / r"qubit_data.json"

varType = 'int'
arb = [1e-3, 1e-2, 1e-2]

chip = freq_allocator.load_chip_data_from_file(H, W, qubit_data, freq_data, xy_crosstalk_sim_path, varType=varType)

frequencys = np.zeros((1000, len(chip.nodes)))
for batch in range(len(frequencys)):
    for qubit in chip.nodes:
        frequencys[batch, list(chip.nodes).index(qubit)] = np.random.choice(chip.nodes[qubit]['allow freq'], 1)
        
err = np.zeros((len(frequencys), len(chip.nodes)))
for frequency in frequencys:
    err[list(frequencys).index(frequency)] = freq_allocator.model.single_err_model(frequency, chip, chip.nodes, arb, varType)