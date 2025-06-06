import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

groups = ['Simon n=4', 'Simon n=9', 'Simon n=16', 'Simon n=25', 
          'QFT n=4', 'QFT n=9', 'QFT n=16', # 'QFT n=25', 
          'qaoa n=4', 'qaoa n=9', 'qaoa n=16', # 'qaoa n=25', 
          'QGAN n=4', 'QGAN n=9', 'QGAN n=16', 'QGAN n=25', 
          'VQE n=4', 'VQE n=9', 'VQE n=16', 'VQE n=25'] # labels for the subgroups
subgroups = ["OQMGS", "N", "S", 'SF', 'DF'] # labels for the groups

values = np.array([[0.9939190418327755, 0.9644888722352167, 0.9644714097040555, 0.9414767381321493, 0.975536468154203],
                [0.9908324971663308, 0.7995406723157004, 0.859618303977925, 0.8492119054572987, 0.9630568803158281],
                [0.9876490330712944, 0.4668509990439088, 0.8915027836297846, 0.765934443463395, 0.9499431376230844],
                [0.9844965196445922, 0.5102208534422418, 0.9095801063310662, 0.896127455620029, 0.9371355393070175],
                
                [0.9210237942951931, 0.8253368448010511, 0.982992296287261, 0.9716059896367338, 0.7155065149897242],
                [0.457765018960472, 0.15388195020443891, 0.28146418770752996, 0.12111066278189259, 0.14910630431398095],
                [0.0558623481939765, 0.00134356, 0, 0, 0.00234356],
                
                [0.9419482071099648, 0.9401894141173012, 0.9401681359431087, 0.9245412141779803, 0.9291043481022276],
                [0.7212246152766337, 0.417714978268463, 0.47081365968381517, 0.47688001277412967, 0.5065703083051832],
                [0.2687107382238431, 0.01256787370800986, 0, 0, 0.040992703591928005],
                
                [0.9682211602851537, 0.761325155965441, 0.7612148897118736, 0.6594270197683292, 0.8757887404175256],
                [0.709756050119437, 0.6063314613333686, 0.5650766478125884, 0.5386115763393382, 0.4748951821361534],
                [0.7489897445585567, 0.03640138643516537, 0.49087323359532536, 0.35463775694522226, 0.4558235401448571],
                [0.7083060033544496, 0.04153836936746, 0.6001335698306766, 0.597264574119361, 0.29192368943092895],
                
                [0.962344327969271, 0.7567060827283719, 0.7565964854766071, 0.6437340513410669, 0.8549446083658989],
                [0.44360226020625065, 0.31327002232665524, 0.4272602706839119, 0.24653366480283043, 0.1746424490963266],
                [0.20376444956851564, 0.025026675035349007, 0, 0, 0.010040475654155901],
                [0.23239802834091855, 0.15032243965430866, 0, 0, 0.025986615611532085]
                ]) # random values for the bars


values[:, 0] = values[:, 0]
values[:, 1] = values[:, 1] / values[:, 0]
values[:, 2] = values[:, 2] / values[:, 0]
values[:, 3] = values[:, 3] / values[:, 0]
values[:, 4] = values[:, 4] / values[:, 0]

# set the width and position of the bars
width = 0.2 # width of each bar
x = np.arange(len(groups)) # x-coordinates of the groups
x_A = x - 2 * width 
x_B = x - width
x_C = x
x_D = x + width 
x_E = x + 2 * width 

# plot the bar chart
plt.figure(figsize=(16, 5))
# plt.semilogy()
# plt.bar(x_A, values[:, 0], width, label=r"$F_{OQMGS}") # create the A bars
plt.bar(x_B, values[:, 1], width, label=r"$F_{N}/F_{CAMEL}$") # create the C bars
plt.bar(x_C, values[:, 2], width, label=r"$F_{S}/F_{CAMEL}$") # create the C bars
plt.bar(x_D, values[:, 3], width, label=r"$F_{SF}/F_{CAMEL}$") # create the C bars
plt.bar(x_E, values[:, 4], width, label=r"$F_{DF}/F_{CAMEL}$") # create the C bars
plt.axhline(y=1, linestyle='--', color='gray')
# 画一个箭头：从点(0.5, 0.5)指向点(0.2, 0.2)
# plt.arrow(5, 1.12, 0, -0.1, head_width=0.05, head_length=0.02, fc='k', ec='k', width=0.0005)
# 在箭尾添加文字
plt.text(10, 1.02, r'better than CAMEL', style='italic', fontsize=15)
# 画一个箭头：从点(0.5, 0.5)指向点(0.2, 0.2)
# plt.arrow(8, 0.88, 0, 0.1, head_width=0.05, head_length=0.02, fc='k', ec='k', width=0.0005)
# 在箭尾添加文字
plt.text(10, 0.94, r'worse than CAMEL', style='italic', fontsize=15)
plt.tick_params(axis='both', labelsize=15)
plt.xticks(x, groups, fontsize=15) # set the x-axis ticks to the group labels
plt.xticks(x, groups, rotation=45) # set the x-axis ticks to the group labels and rotate them 90 degrees
# plt.xlabel("") # add a label for the x-axis
plt.ylabel("Fidelity ratio", fontsize=15) # add a label for the y-axis
plt.tight_layout()
# plt.title("Grouped Bar Chart Example") # add a title for the plot
plt.legend(fontsize=15, loc=1) # add a legend for the subgroups
plt.show() # show the plot


groups = ['Simon n=4', 'Simon n=9', 'Simon n=16', 'Simon n=25', 
          'QFT n=4', 'QFT n=9', 'QFT n=16', 'QFT n=25', 
          'qaoa n=4', 'qaoa n=9', 'qaoa n=16', 'qaoa n=25', 
          'QGAN n=4', 'QGAN n=9', 'QGAN n=16', 'QGAN n=25', 
          'VQE n=4', 'VQE n=9', 'VQE n=16', 'VQE n=25'] # labels for the subgroups
subgroups = ["OQMGS", "N", "S", 'SF', 'DF'] # labels for the groups
values = np.array([[120, 300, 300, 300, 120],
                    [180, 660, 900, 810, 180],
                    [240, 660, 2250, 1320, 240],
                    [300, 300, 1860, 540, 300],
                    
                    [1620, 2100, 2340, 1980, 1620],
                    [7140, 7050, 12930, 10750, 7170],
                    [26940, 15750, 43050, 30830, 26790],
                    [72930, 27270, 171550, 127300, 72840],
                    
                    [360, 390, 390, 390, 360],
                    [2250, 2700, 3300, 3030, 2250],
                    [11940, 8400, 19590, 18540, 11910],
                    [34590, 16650, 82140, 62950, 34680],
                    
                    [630, 1070, 2070, 2070, 630],
                    [2520, 1430, 3780, 2970, 2520],
                    [3120, 2970, 11010, 10130, 3060],
                    [5400, 810, 16990, 10170, 5400],
                    
                    [750, 1190, 2190, 2190, 750],
                    [5850, 5070, 6450, 6330, 5850],
                    [18420, 9210, 71530, 27670, 18540],
                    [27330, 4530, 87110, 58490, 27390],
                ], dtype=float) # random values for the bars

values[:, 0] = values[:, 0]
values[:, 1] = values[:, 1] / values[:, 0]
values[:, 2] = values[:, 2] / values[:, 0]
values[:, 3] = values[:, 3] / values[:, 0]
values[:, 4] = values[:, 4] / values[:, 0]

# set the width and position of the bars
width = 0.2 # width of each bar
x = np.arange(len(groups)) # x-coordinates of the groups
x_A = x - 2 * width 
x_B = x - width
x_C = x
x_D = x + width 
x_E = x + 2 * width 

# plot the bar chart
plt.figure(figsize=(16, 5))
# plt.semilogy()
# plt.bar(x_A, values[:, 0], width, label=r"$T_{OQMGS}") # create the A bars
plt.bar(x_B, values[:, 1], width, label=r"$d_{N}/d_{CAMEL}$") # create the C bars
plt.bar(x_C, values[:, 2], width, label=r"$d_{S}/d_{CAMEL}$") # create the C bars
plt.bar(x_D, values[:, 3], width, label=r"$d_{SF}/d_{CAMEL}$") # create the C bars
plt.bar(x_E, values[:, 4], width, label=r"$d_{DF}/d_{CAMEL}$") # create the C bars
plt.axhline(y=1, linestyle='--', color='gray')
# 画一个箭头：从点(0.5, 0.5)指向点(0.2, 0.2)
# plt.arrow(5, 1.12, 0, -0.1, head_width=0.05, head_length=0.02, fc='k', ec='k', width=0.0005)
# 在箭尾添加文字
plt.text(-1, 7, r'worse than CAMEL', style='italic', fontsize=15)
# 画一个箭头：从点(0.5, 0.5)指向点(0.2, 0.2)
# plt.arrow(8, 0.88, 0, 0.1, head_width=0.05, head_length=0.02, fc='k', ec='k', width=0.0005)
# 在箭尾添加文字
plt.text(-1.2, 0.4, r'better', style='italic', fontsize=15)
plt.tick_params(axis='both', labelsize=15)
plt.xticks(x, groups, fontsize=15) # set the x-axis ticks to the group labels
plt.xticks(x, groups, rotation=45) # set the x-axis ticks to the group labels and rotate them 90 degrees
# plt.xlabel("") # add a label for the x-axis
plt.ylabel("Circuit Depth ratio", fontsize=15) # add a label for the y-axis
plt.tight_layout()
# plt.title("Grouped Bar Chart Example") # add a title for the plot
plt.legend(fontsize=15) # add a legend for the subgroups
plt.show() # show the plot
