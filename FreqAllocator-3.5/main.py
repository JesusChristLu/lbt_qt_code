from pathlib import Path
import freq_allocator
import matplotlib.pyplot as plt
import json
import numpy as np
import networkx as nx
from freq_allocator.model.data_gen import MyDataset

if __name__ == '__main__':
     # H = 2
     # W = 4

     # czList = ["('q32', 'q33')", "('q32', 'q38')", "('q33', 'q34')", "('q33', 'q39')", "('q34', 'q35')", "('q34', 'q40')", "('q35', 'q41')", "('q38', 'q39')", "('q39', 'q40')", "('q40', 'q41')"]
     # qList = ['q32', 'q33', 'q34', 'q35', 'q38', 'q39', 'q40', 'q41']
     # qDict = {(0, 0) : 'q32', (0, 1) : 'q33', (0, 2) : 'q34', (0, 3) : 'q35', (1, 0) : 'q38', (1, 1) : 'q39', (1, 2) : 'q40', (1, 3) : 'q41'}

     H = 5
     W = 5

     qList = ['q0', 'q1', 'q2', 'q3', 'q4',
             'q6', 'q7', 'q8', 'q9', 'q10',
             'q12', 'q13', 'q14', 'q15', 'q16',
             'q18', 'q19', 'q20', 'q21', 'q22',
             'q24', 'q25', 'q26', 'q27', 'q28']

     chipGrid = nx.grid_2d_graph(H, W)
     chipGrid = nx.relabel_nodes(chipGrid, dict(zip(nx.grid_2d_graph(H, W).nodes, qList)))
     czList = [str(edge) for edge in chipGrid.edges]

     qDict = dict(zip(nx.grid_2d_graph(H, W).nodes, qList))

     xy_crosstalk_sim_path = Path.cwd() / 'chipdata' / r"xy_crosstalk_sim.json"
     freq_data = Path.cwd() / 'chipdata' / r"qubit_freq_real.json"
     qubit_data = Path.cwd() / 'chipdata' / r"qubit_data.json"

     chip = freq_allocator.load_chip_data_from_file(H, W, qubit_data, freq_data, xy_crosstalk_sim_path, loc=qDict)

     errcwd = Path.cwd() / 'chipdata' / r"err_list.json"
     freqcwd = Path.cwd() / 'chipdata' / r"freq_list.json"
     with open(freqcwd, 'r') as f:
          # 读取文件内容，返回一个Python对象
          data = json.load(f)
          
     with open(errcwd, 'r') as f:                                                                    
          # 读取文件内容，返回一个Python对象
          labels = json.load(f)

     # 假设你已经有了一个包含所有数据的数据集对象 dataset
     dataset = MyDataset(H, W, chip, data=data, labels=labels)
     
     freq_allocator.alloc_nn(chip, 1, minMaxErr=(dataset.minErr, dataset.maxErr))

     with open(Path.cwd() / 'results' / 'alloc' / 'gates_nn.json', 'r') as f:
          # 使用 json.load() 函数加载 JSON 数据
          data = json.load(f)
     
     errLabel = ['single-qubit err', 'two-qubit err']

     meanerr1_nn = np.mean([data[qubit]['all err'] for qubit in data if qubit in qList])
     stderr1_nn = np.std([data[qubit]['all err'] for qubit in data if qubit in qList])

     qcqs = [qcq for qcq in data.keys() if len(qcq) > 4]
     qcqs = [eval(qcq) for qcq in qcqs]

     meanerr2_nn = np.mean([data[str(qcq)]['all err'] for qcq in data if qcq in czList])
     stderr2_nn = np.std([data[str(qcq)]['all err'] for qcq in data if qcq in czList])

     plt.title('Final Avg Gate Error', fontsize=15)
     plt.ylabel('error', fontsize=15)
     plt.bar(x=errLabel, 
               height=[meanerr1_nn, meanerr2_nn],
               yerr=[stderr1_nn, stderr2_nn],
               capsize=5)
     plt.xticks(fontsize=15)
     plt.yticks(fontsize=15)
     plt.semilogy()
     plt.tight_layout()
     plt.savefig(Path.cwd() / 'results' / 'alloc' / 'avg_err_nn.pdf', dpi=300)
     plt.close()

     aIni = [1, 1, 
            1, 25, 1, 50,
            4e-4, 1e-7, 1e-2, 1e-2, 1e-5,
            1, 80, 1, 50, 
            1, 
            1, 23, 1, 60]   
     # aIni = np.random.random(len(aIni))
     
     freq_allocator.alloc(chip, aIni, 2)

     with open(Path.cwd() / 'results'  / 'alloc' / 'gates_mod.json', 'r') as f:
          # 使用 json.load() 函数加载 JSON 数据
          data = json.load(f)

     errLabel = ['single-qubit err', 'two-qubit err']

     meanerr1_mod = np.mean([data[qubit]['all err'] for qubit in data if qubit in qList])
     stderr1_mod = np.std([data[qubit]['all err'] for qubit in data if qubit in qList])

     qcqs = [qcq for qcq in data.keys() if len(qcq) > 4]
     qcqs = [eval(qcq) for qcq in qcqs]

     meanerr2_mod = np.mean([data[str(qcq)]['all err'] for qcq in data if qcq in czList])
     stderr2_mod = np.std([data[str(qcq)]['all err'] for qcq in data if qcq in czList])

     plt.title('Final Avg Gate Error', fontsize=15)
     plt.ylabel('error', fontsize=15)
     plt.bar(x=errLabel, 
               height=[meanerr1_mod, meanerr2_mod],
               yerr=[stderr1_mod, stderr2_mod],
               capsize=5)
     plt.xticks(fontsize=15)
     plt.yticks(fontsize=15)
     plt.semilogy()
     plt.tight_layout()
     plt.savefig(Path.cwd() / 'results' / 'alloc' / 'avg_err_mod.pdf', dpi=300)
     plt.close()

     print(meanerr1_nn, stderr1_nn)
     print(meanerr2_nn, stderr2_nn)

     print(meanerr1_mod, stderr1_mod)
     print(meanerr2_mod, stderr2_mod)     
     
     # loss = [0.037829507, 0.01669425, 0.012443605, 0.0076161902, 0.0045820693, 0.004925775, 0.0043993182, 0.004364799, 0.006006949, 0.007594791, 0.0065618, 0.005727738, 0.008985559, 0.0064420355, 0.0055418867, 0.0061328136, 0.0075018345, 0.006340784, 0.006165202, 0.0057334905, 0.0062765325, 0.0062571773, 0.0055245496, 0.005859074, 0.006674338, 0.005786214, 0.005109448, 0.006866731, 0.007250325, 0.005962259, 0.006527642, 0.0064962013, 0.0062837326, 0.007049222, 0.00742368, 0.0061294492, 0.0057643154, 0.0074254395, 0.006159226, 0.0069753216]

     # plt.title('Avg Gate Error per Epoch', fontsize=20)
     # plt.xlabel('epoch', fontsize=20)
     # plt.ylabel('avg error', fontsize=20)
     # plt.plot(loss)
     # plt.xticks(fontsize=20)
     # plt.yticks(fontsize=20)
     # plt.semilogy()
     # plt.tight_layout()
     
     # plt.savefig(Path.cwd() / 'results' / 'alloc' / 'loss_nn.pdf', dpi=300)
     # plt.close()

     # plt.title('Avg Error & Config Time vs. S Radii', fontsize=20)
     # # 创建一些示例数据
     # x = [1, 2, 3, 4]  # x轴数据
     # y = [0.2, 0.009, 0.006, 0.008]                # y轴数据
     # y_err = [0.02, 0.003, 0.005, 0.002]  # y方向的误差

     # # 配置时间的数据
     # config_time = [20, 390, 2542, 21330]  # 配置时间 (s)

     # # 创建带有误差条的折线图
     # plt.errorbar(x, y, yerr=y_err, fmt='o', color='b', 
     #           ecolor='green', elinewidth=2, capsize=4)

     # # 添加标题和标签
     # plt.xlabel('S Radius', fontsize=20)
     # plt.ylabel('avg err', fontsize=20)

     # # 设置 x 轴刻度为整数
     # plt.xticks(x, fontsize=20)
     # plt.yticks(fontsize=20)
     # plt.semilogy()  # 设置 y 轴为对数坐标

     # # 创建右侧的 y 轴
     # ax2 = plt.gca().twinx()
     # ax2.set_ylabel('Configuration Time (s)', fontsize=20)

     # # 绘制配置时间的图线
     # ax2.plot(x, config_time, 'r-', marker='x', label='Configuration Time', linewidth=2)

     # # 添加右侧的 y 轴的刻度
     # ax2.tick_params(labelsize=20)
     # ax2.semilogy()

     # # 显示网格
     # plt.grid(True)
     # plt.tight_layout()

     # # 保存图形
     # plt.savefig(Path.cwd() / 'results' / 'alloc' / 's_nn.pdf', dpi=300)
     # plt.close()

     # # Function to plot the CDF and bar plots
     # def plot_cdf_and_bars(mean_vals, std_vals, random_probs, maxErr, filename):
     #      # Create subplots (two rows: CDF plot and bar plot)
     #      fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

     #      # CDF Plot
     #      cdfs = []
     #      for i, data in enumerate(random_probs):
     #           data = np.where(data < 0, 0, data)  # Make sure values are >= 0
     #           hist, bin_edges = np.histogram(data, bins=100, range=(0, maxErr))
     #           cdf = np.cumsum(hist * np.diff(bin_edges))
     #           cdf /= cdf[-1]  # Normalize CDF
     #           cdfs.append(cdf)

     #      # Plot the CDFs
     #      labels = ['nn.', 'snake', 'rand.']
     #      for i, cdf in enumerate(cdfs):
     #           axs[0].plot(np.linspace(0, maxErr, len(cdf)), cdf, linestyle='-', label=labels[i])
          
     #      axs[0].set_title('Error CDF', fontsize=15)
     #      axs[0].set_xlabel('Error(%)', fontsize=15)
     #      axs[0].set_ylabel('CDF', fontsize=15)
     #      axs[0].tick_params(axis='x', labelsize=15)
     #      axs[0].tick_params(axis='y', labelsize=15)
     #      axs[0].legend(fontsize=15)
     #      axs[0].grid(True)
     #      axs[0].semilogx()

     #      # Bar Plot below CDF plot
     #      # Create error bar plots for means and std deviations
     #      axs[1].errorbar(labels, mean_vals, yerr=std_vals, fmt='o', label='Mean ± Std Dev', 
     #                          color='skyblue', ecolor='lightcoral', capsize=5, linestyle='None', markersize=8)

     #      axs[1].set_title('Means and Standard Deviations', fontsize=15)
     #      axs[1].set_xlabel('Distributions', fontsize=15)
     #      axs[1].set_ylabel('Error(%)', fontsize=15)
     #      axs[1].tick_params(axis='x', labelsize=15)
     #      axs[1].tick_params(axis='y', labelsize=15)
     #      # axs[1].legend(fontsize=15)
     #      axs[1].semilogy()
     #      axs[1].grid(True)

     #      # Tight layout and save the plot
     #      plt.tight_layout()
     #      plt.savefig(filename, dpi=300)
     #      plt.close()

     # # First plot data (for 1q-gate RB error CDF)
     # mean_vals_1 = [0.28, 0.85, 1.02]
     # std_vals_1 = [0.15, 0.36, 0.56]
     # size_1 = 25
     # random_probs_1 = [np.random.normal(mean_vals_1[i], std_vals_1[i], size_1) for i in range(3)]
     # maxErr_1 = np.max([np.max(prob) for prob in random_probs_1])

     # plot_cdf_and_bars(mean_vals_1, std_vals_1, random_probs_1, maxErr_1, Path.cwd() / 'results' / 'alloc' / 'rb.pdf')

     # # Second plot data (for 2q-gate XEB error CDF)
     # mean_vals_2 = [1.24, 1.38, 23]
     # std_vals_2 = [0.46, 0.45, 3.97]
     # size_2 = 40
     # random_probs_2 = [np.random.normal(mean_vals_2[i], std_vals_2[i], size_2) for i in range(3)]
     # maxErr_2 = np.max([np.max(prob) for prob in random_probs_2])

     # plot_cdf_and_bars(mean_vals_2, std_vals_2, random_probs_2, maxErr_2, Path.cwd() / 'results' / 'alloc' / 'xeb.pdf')