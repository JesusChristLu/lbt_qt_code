# FreqAllocator

## Install

Every time you make a change to the code, run this in your conda environment
```bash
pip install .
```

## Usage

### Example
```python
import freq_allocator
H = 8
W = 6
xy_crosstalk_sim_path = Path.cwd() / 'chipdata' / r"xy_crosstalk_sim.json"
qubit_data = Path.cwd() / 'chipdata' / r"qubit_data.json"

chip, xy_crosstalk_sim_dic = freq_allocator.load_chip_data_from_file(H, W, qubit_data, xy_crosstalk_sim_path)
chip, conflictNodeDict = freq_allocator.sigq_alloc(chip, H, W, arb, xy_crosstalk_sim_dic, s=2)
for qubit in chip.nodes:
    print(
        qubit,
        chip.nodes[qubit]['frequency'],
        chip.nodes[qubit]['frequency'] - chip.nodes[qubit]['sweet point'],
    )
```

### Run single qubit allocation script
```bash
python ./test/allocate_single_qubit/main.py
```