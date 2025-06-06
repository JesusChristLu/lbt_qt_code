from pathlib import Path
import freq_allocator

if __name__ == '__main__':

    arb = [0.2, 10, 0.5, 10, 1e-7]
    axeb = [4e-4, 1e-7, 1e-2, 1e-5, 1e-2, 1, 10, 0.7, 10]

    H = 12
    W = 6

    varType = 'int'

    xy_crosstalk_sim_path = Path.cwd() / '..' / 'chipdata' / r"xy_crosstalk_sim.json"
    freq_data = Path.cwd() / '..' / 'chipdata' / r"qubit_freq_real.json"
    qubit_data = Path.cwd() / '..' / 'chipdata' / r"qubit_data.json"

    chip = freq_allocator.load_chip_data_from_file(H, W, qubit_data, freq_data, xy_crosstalk_sim_path, varType=varType)
    chip = freq_allocator.sing_alloc(chip, arb, s=2, varType=varType)
    if not(chip.nodes[(0, 0)].get('frequency', False)):
        chip = freq_allocator.sing_alloc(chip, arb, s=2, varType=varType)

    xtalkG = freq_allocator.two_alloc(chip, axeb)