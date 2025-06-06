import matplotlib.pyplot as plt
import qpandalite
import qpandalite.task.origin_qcloud as originq
from pychemiq import Molecules
from pychemiq.Transform.Mapping import jordan_wigner
import numpy as np
import json

def generate_geometry(dist):
    # return f"H 0 0 0, H 0 0 {dist}"
    # return f"H 0 0 0, H 0 0 {dist}, H 0 0 {2 * dist}, H 0 0 {3 * dist}"
    # return f"H 0 0 0, Li 0 0 {dist}"
    # return f"N 0 0 0, N 0 0 {dist}"
    return f"H {dist} {dist} {dist}, H {-dist} {-dist} {dist}, H {-dist} {dist} {-dist}, H {dist} {-dist} {-dist}"

# finished, task_count = originq.query_all_task()

# while 1:
#     if finished == task_count:
#         break

# taskid = originq.get_last_taskid()

patternid = 1
taskidses = [
                [
                    [
                        ['1E1FE641E03D3F9C84ABBC74927676D4'],
                        ['F0C5F8BC73AFCA0A7CF2FA0B9E89FF62'],
                        ['A85A59E45760BDAFE7E39DCDA72E5907'],
                        ['9B03257E5C192C085D087CCA46E5AE85'],
                        ['10BBD7DECD301F612263803558DAB5B6']
                    ],
                    [
                        ['7E26D3A155B9C2264A42D37DCAAF4454'],
                        ['99E365993C3347CC2D2951563F257262'],
                        ['85D82B67B57DCC09EDD6AA12FF2704BD'],
                        ['A2CE5CA05F9A230187333884FC7DDF41'],
                        ['0306559D1B514708A1818759CE4CF0D5']
                    ],
                    [
                        ['0D7D3822B35FFA4397566D9635344DAA'],
                        ['5DE0645E003A99BAD671685818BC2035'],
                        ['F3BECED6C68D086EB871E3AC7CB92B33'],
                        ['52D5A397AA6C928DD9FCFC7A8CF84004'],
                        ['E569267920466ABAECD96716D4D48D70']
                    ],
                    [
                        ['DAE40440C7403D2FD0181F9F1F50F61A'],
                        ['40DD1DA43F819A100F6768EFFA033FBF'],
                        ['E3966F83BEC1163B8C9CD481CFD0ADA8'],
                        ['FD9D5FFCF928FD7E8063D70435E9A06D'],
                        ['C0B276948A9173A050760FF0373D175B']
                    ],
                    [
                        ['9FEEBD1F334CFBDBFA0243E706AFA76A'],
                        ['DF4EF5A4B3B81CEFE46D8274F86E2337'],
                        ['79101832B06A16D22DBDA360FB172BE8'],
                        ['77B7ED25C33DF0910BB3F207125EF084'],
                        ['7C715E7142FB6EA3B75CB97B9497B443']
                    ]
                ],
                [
                    [
                        ['4609803A25CBD46D2A85BF45314D2899'],
                        ['68573563B0736BDAD1A3D53244882686'],
                        ['6528D15EFC44219E17A061101D9AF2CE'],
                        ['07557CE486AE800429637C6484DCC14D'],
                        ['CDA603EC7B089EC72500F20B6C577BD0']
                    ],
                    [
                        ['601283DA7F70DCA626831AB3E4454085'],
                        ['6340976CC5547C80A70D79EA9C34DDA8'],
                        ['D627B702F3A82F4EDFFA0876E21BD076'],
                        ['07349B92EC87B426C7AA69FEC684722C'],
                        ['77309A8A9216ABA8660CB66D604E8B1A']
                    ],
                    [
                        ['82B8070B4B52EFADF4C0071328385345'],
                        ['EC46504EC17A5D3F79F2F6B6E9F76A06'],
                        ['5D78716ED401EBC82415BE8228FE3606'],
                        ['466CA5777D388CB4E0B48CDC48DBC9FE'],
                        ['0DD930C1F51C181342E3216171BF780B']
                    ],
                    [
                        ['706DEE349E4AE915A69982E7E6D381CE'],
                        ['459D3B7CD2B733DD6BCE349C9BDA697C'],
                        ['7E885F3D83EA62AB0E6A660B4FDEF534'],
                        ['4DDBD936EB6D8C9B4E4F665C3C041D40'],
                        ['B2587196B753F304679BBA6B5FC6FA63']
                    ],
                    [
                        ['F701E153EF5BABB91E5B6519B76BA5B4'],
                        ['2534CD115591F0298DEF9D14A6024105'],
                        ['F481B1211B2AFDB140BCB737424DC665'],
                        ['8CC78C7BB3850893492F6245A7A445CB'],
                        ['D5E3C46A0998C6D454E54D2C3255AB59']
                    ]
                ]
            ]

taskids = taskidses[patternid]

output = np.zeros((5, 5))

rs = np.linspace(0.18, 3.5, 5)

for j, r in enumerate(rs):
    geom = generate_geometry(r)

    # 创建分子对象
    # molecule = Molecules(geom, charge=0, multiplicity=1, basis="sto-3g")
    molecule = Molecules(geom, charge=1, multiplicity=2, basis="sto-3g")
    n_qubits = molecule.n_qubits
    n_elec = molecule.n_electrons
    # print(n_qubits, n_elec)

    # 获取分子哈密顿量
    hamiltonian = molecule.get_molecular_hamiltonian()

    # 使用 Jordan-Wigner 变换将费米子哈密顿量转换为量子比特哈密顿量
    qubit_hamiltonian = jordan_wigner(hamiltonian)

    # 将生成的哈密顿量转换为列表形式，以便在 TensorCircuit 中使用

    for i, taskid in enumerate(taskids[j]):
        results = originq.query_by_taskid_sync(taskid)

        results = qpandalite.convert_originq_result(
            results, 
            style='list', 
            prob_or_shots='shots',
            key_style='dec'
        )

        circuitId = 0
        # 遍历 PauliOperator 对象的项
        for term, coeff in qubit_hamiltonian.data():
            ops = []
            for index, op in term[0].items():
                ops.append((index, op))
            if ops == []:
                energyExp = np.real(coeff)
            else:
                result = results[circuitId]
                assert len(result) == 2 ** len(ops)
                for k, prob in enumerate(result):
                    binI = bin(k)[2:]
                    pm = binI.count('1')
                    if pm % 2:
                        energyExp -= np.real(coeff * prob)
                    else:
                        energyExp += np.real(coeff * prob)
                circuitId += 1
        output[i, j] = energyExp

# 沿着第一个维度（行）计算均值，结果是一个 1xN 的数组
print(output)
mean_array = np.mean(output, axis=0)
max_array = np.max(output, axis=0)
min_array = np.min(output, axis=0)
# 沿着第一个维度（行）计算方差，结果是一个 1xN 的数组
var_array = np.var(output, axis=0)

print("mean:", mean_array)
print("max:", max_array)
print("min:", min_array)
print("var:", var_array)