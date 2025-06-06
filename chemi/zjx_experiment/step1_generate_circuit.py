from copy import deepcopy
import qpandalite
import qpandalite.task.origin_qcloud as originq

# def circuit1():
#     c = qpandalite.Circuit()
#     c.h(18)
#     c.h(19)
#     c.cnot(12,6)
#     c.cnot(13,7)
#     c.cnot(24,30)
#     c.cnot(25,31)
#     c.cnot(18,24)
#     c.cnot(19,25)
#     c.cnot(30,36)
#     c.cnot(31,37)
#     c.cnot(6,0)
#     c.cnot(7,1)
#     c.cnot(18,12)
#     c.cnot(19,13)
#     c.cnot(12,6)
#     c.cnot(13,7)
#     c.cnot(24,30)
#     c.cnot(25,31)
#     c.h(18)
#     c.h(19)
#     c.measure(0,1,18,19,36,37)
#     return c.circuit

# def circuit2():
#     c = qpandalite.Circuit()
#     c.h(18)
#     c.h(19)
#     c.h(6)
#     c.h(7)
#     c.h(12)
#     c.h(13)
#     c.h(24)
#     c.h(25)
#     c.h(30)
#     c.h(31)
#     c.cnot(12,6)
#     c.cnot(13,7)
#     c.cnot(24,30)
#     c.cnot(25,31)
#     c.cnot(18,24)
#     c.cnot(19,25)
#     c.cnot(30,36)
#     c.cnot(31,37)
#     c.cnot(6,0)
#     c.cnot(7,1)
#     c.cnot(18,12)
#     c.cnot(19,13)
#     c.cnot(12,6)
#     c.cnot(13,7)
#     c.cnot(24,30)
#     c.cnot(25,31)
#     c.h(18)
#     c.h(19)
#     c.h(6)
#     c.h(7)
#     c.h(12)
#     c.h(13)
#     c.h(24)
#     c.h(25)
#     c.h(30)
#     c.h(31)
#     c.measure(0,1,6,7,18,19,24,25,30,31,36,37)    
#     return c.circuit

# def circuit3():
#     c = qpandalite.Circuit()    
#     c.h(18)
#     c.h(19)
#     c.h(6)
#     c.h(12)
#     c.h(24)
#     c.h(30)
#     c.cnot(12,6)
#     c.cnot(13,7)
#     c.cnot(24,30)
#     c.cnot(25,31)
#     c.cnot(18,24)
#     c.cnot(19,25)
#     c.cnot(30,36)
#     c.cnot(31,37)
#     c.cnot(6,0)
#     c.cnot(7,1)
#     c.cnot(18,12)
#     c.cnot(19,13)
#     c.cnot(12,6)
#     c.cnot(13,7)
#     c.cnot(24,30)
#     c.cnot(25,31)
#     c.h(18)
#     c.h(19)
#     c.h(6)
#     c.h(7)
#     c.h(12)
#     c.h(13)
#     c.h(24)
#     c.h(25)
#     c.h(30)
#     c.h(31)
#     c.cnot(6,7)
#     c.cnot(12,13)
#     c.cnot(24,25)
#     c.cnot(30,31)
#     c.measure(0,1,6,7,12,13,18,19,24,25,30,31,36,37)
#     return c.circuit


# circuits = [circuit1(), circuit2(), circuit3()]

header = '''
QINIT 38
CREG 14
'''.strip()

circuit_left = '''
H q[18]
CNOT q[12], q[6]
CNOT q[24], q[30]
CNOT q[18], q[24]
CNOT q[30], q[36]
CNOT q[6], q[0]
CNOT q[18], q[12]
CNOT q[12], q[6]
CNOT q[24], q[30]
H q[18]
'''.strip()

circuit_right = '''
H q[19]
CNOT q[13], q[7]
CNOT q[25], q[31]
CNOT q[19], q[25]
CNOT q[31], q[37]
CNOT q[7], q[1]
CNOT q[19], q[13]
CNOT q[13], q[7]
CNOT q[25], q[31]
H q[19]
'''.strip()

measure_left = '''
MEASURE q[0], c[0]
MEASURE q[6], c[2]
MEASURE q[12], c[4]
MEASURE q[18], c[6]
MEASURE q[24], c[8]
MEASURE q[30], c[10]
MEASURE q[36], c[12]
'''.strip()

measure_right = '''
MEASURE q[1], c[1]
MEASURE q[7], c[3]
MEASURE q[13], c[5]
MEASURE q[19], c[7]
MEASURE q[25], c[9]
MEASURE q[31], c[11]
MEASURE q[37], c[13]'''.strip()


circuits = [
    '\n'.join([header, circuit_left, measure_left]),
    '\n'.join([header, circuit_right, measure_right]),
    '\n'.join([header, circuit_left, circuit_right, measure_left]),
    '\n'.join([header, circuit_left, circuit_right, measure_right]),
    '\n'.join([header, circuit_left, circuit_right, measure_left, measure_right]),
]

def remapping(c, mapping):
    c = deepcopy(c)
    for old_qubit, new_qubit in mapping.items():
        c = c.replace(f'q[{old_qubit}]', f'q[_{old_qubit}]')

    for old_qubit, new_qubit in mapping.items():
        c = c.replace(f'q[_{old_qubit}]', f'q[{new_qubit}]')

    return c

mapping = {1:37,7:31,13:25,19:19,25:13,31:7,37:1}

# for i in range(len(circuits)):
for i in [0]:
    circuits[i] = remapping(circuits[i], mapping)

print(circuits[0])

# exit(0)

taskid = originq.submit_task(circuits, shots=10, auto_mapping=False, measurement_amend=True)
print(taskid)