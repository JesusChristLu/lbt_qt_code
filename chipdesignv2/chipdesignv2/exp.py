from qiskit.test.mock import FakeTenerife
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

import logging


# compilation
from qiskit.transpiler import Layout, CouplingMap
from qiskit.transpiler import PassManager
from qiskit.transpiler import passes
from qiskit.transpiler import preset_passmanagers
from qiskit.transpiler import PropertySet
from qiskit.compiler import transpile


logging.basicConfig(level='DEBUG')

log_circ = QuantumCircuit(2, 2)
log_circ.h(0)
log_circ.h(1)
log_circ.h(1)
log_circ.x(1)
log_circ.cx(0, 1)
log_circ.measure([0,1], [0,1])

backend = FakeTenerife()

transpile(log_circ, backend, optimization_level=3);


#INFO:qiskit.transpiler.runningpassmanager:Pass: SetLayout - 0.00000 (ms)
#INFO:qiskit.transpiler.runningpassmanager:Pass: CSPLayout - 0.00000 (ms)
#INFO:qiskit.transpiler.runningpassmanager:Pass: FullAncillaAllocation - 0.00000 (ms)
#INFO:qiskit.transpiler.runningpassmanager:Pass: EnlargeWithAncilla - 0.00000 (ms)
#INFO:qiskit.transpiler.runningpassmanager:Pass: ApplyLayout - 0.00000 (ms)
#INFO:qiskit.transpiler.runningpassmanager:Pass: CheckMap - 0.00000 (ms)