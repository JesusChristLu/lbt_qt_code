from pathlib import Path
import freq_allocator
import matplotlib.pyplot as plt
import json
import numpy as np
import networkx as nx

if __name__ == '__main__':
     H = 12
     W = 6

     xy_crosstalk_sim_path = Path.cwd() / 'chipdata' / r"xy_crosstalk_sim.json"
     freq_data = Path.cwd() / 'chipdata' / r"qubit_freq_real.json"
     qubit_data = Path.cwd() / 'chipdata' / r"qubit_data.json"

     chip = freq_allocator.load_chip_data_from_file(H, W, qubit_data, freq_data, xy_crosstalk_sim_path)

     # chip = freq_allocator.alloc(chip, a, 2)
     chip = freq_allocator.alloc_nn(chip, 1)

     with open(Path.cwd() / 'results' / 'gates.json', 'r') as f:
          # 使用 json.load() 函数加载 JSON 数据
          data = json.load(f)
     
     errLabel = ['err1', 'err2']

     err1 = ['all err']
     err2 = ['all err']

     meanerr1 = np.mean([data[qubit][err1[0]] for qubit in chip.nodes])
     stderr1 = np.std([data[qubit][err1[0]] for qubit in chip.nodes])

     qcqs = [qcq for qcq in data.keys() if len(qcq) > 4]
     qcqs = [eval(qcq) for qcq in qcqs]

     meanerr2 = np.mean([data[str(qcq)][err2[0]] for qcq in qcqs])
     stderr2 = np.std([data[str(qcq)][err2[0]] for qcq in qcqs])
     
     plt.ylabel('error')
     plt.bar(x=errLabel, 
               height=[meanerr1, meanerr2],
               yerr=[stderr1, stderr2],
               capsize=5)
     plt.xticks(rotation=45)
     plt.show()