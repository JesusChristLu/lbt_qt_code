import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import pi

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit

from qiskit.compiler import transpile
from qiskit.transpiler import PassManager

from qiskit.transpiler import passes
from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passes import Collect2qBlocks

from qiskit.converters import circuit_to_dag
from qiskit.tools.visualization import dag_drawer

def QAOA(V, p, E, beta):
    # Generating the ring graph with 6 nodes
    precision = 0.01

    # prepare the quantum and classical resisters
    QAOA = QuantumCircuit(len(V))
    # apply the layer of Hadamard gates to all qubits
    QAOA.h(range(len(V)))

    # apply the Ising type gates with angle gamma along the edges in E
    for layer in range(p):
        for edge in E:
            k = edge[0]
            l = edge[1]
            QAOA.u1(beta[layer][0], k)
            QAOA.u1(beta[layer][0], l)
            QAOA.cu1(-2 * beta[layer][0] + precision * (np.random.random() - 0.5), k, l)
        # then apply the single qubit X - rotations with angle beta to all qubits
        QAOA.ry(-pi / 2 + precision * (np.random.random() - 0.5), range(len(V)))
        QAOA.u1(2 * beta[layer][1], range(len(V)))
        QAOA.ry(pi / 2 + precision * (np.random.random() - 0.5), range(len(V)))
        # Finally measure the result in the computational basis
    return QAOA

def grover(n):
    work_bit = [0, 1, 3]

    b = np.random.randint(1, 2 ** 3)
    # Next, format 'b' as a binary string of length 'n', padded with zeros:
    b_str = format(b, '0'+str(3)+'b')

    print(b_str)

    qc = QuantumCircuit(6, 6)

    qc.reset(range(6))

    qc.rx(np.pi, 5)
    
    qc.barrier()
    
    # And set up the input register:
    qc.h([0, 1, 3, 5]) 

    qc.barrier()

    # Set up the output qubit:
    for qubit in work_bit:
        if b_str[work_bit.index(qubit)] == '0':
            qc.rx(np.pi, qubit)
    
    qc.barrier()
    qc.toffoli(0, 1, 2)
    qc.toffoli(2, 3, 4)
    qc.cx(4, 5)
    qc.toffoli(2, 3, 4)
    qc.toffoli(0, 1, 2)
    qc.barrier()
    for qubit in work_bit:
        if b_str[work_bit.index(qubit)] == '0':
            qc.rx(np.pi, qubit)
    qc.barrier()

    qc.h([0, 1, 3]) 

    for qubit in work_bit:
        qc.rx(np.pi, qubit)

    qc.barrier()

    qc.h(3)  
    qc.toffoli(0, 1, 3)
    qc.h(3) 

    qc.barrier()

    for qubit in work_bit:
        qc.rx(np.pi, qubit) 

    qc.h([0, 1, 3])

    qc.barrier()
    return qc

def cnx(n, control_bit, ancilla_bit, target):
    cnx = QuantumCircuit(n)
    cnx.toffoli(control_bit[0], control_bit[1], ancilla_bit[0])
    for i in control_bit[2:]:
        cnx.toffoli(control_bit[i], ancilla_bit[i - 2], ancilla_bit[i - 1])
    cnx.cx(ancilla_bit[i - 1], target)
    for i in control_bit[::-1][:-2]:
        cnx.toffoli(control_bit[i], ancilla_bit[i - 2], ancilla_bit[i - 1])
    cnx.toffoli(control_bit[0], control_bit[1], ancilla_bit[0])
    return cnx

def grover(work_bit_num):
    work_bit = range(0, work_bit_num)
    n = work_bit_num * 2
    b = np.random.randint(1, 2 ** work_bit_num)
    # Next, format 'b' as a binary string of length 'n', padded with zeros:
    b_str = format(b, '0' + str(work_bit_num) + 'b')

    print(b_str)

    qc = QuantumCircuit(n, n)

    qc.reset(range(n))

    qc.x(n - 1)

    qc.barrier()

    # And set up the input register:
    qc.h(work_bit) 
    qc.h(n - 1)

    qc.barrier()

    # Set up the output qubit:
    for qubit in work_bit:
        if b_str[work_bit.index(qubit)] == '0':
            qc.x(qubit)
    
    qc.barrier()

    qc = qc + cnx(n, work_bit, range(work_bit_num, n - 1), n - 1)

    qc.barrier()

    for qubit in work_bit:
        if b_str[work_bit.index(qubit)] == '0':
            qc.x(qubit)

    qc.barrier()

    qc.h(work_bit) 

    qc.x(work_bit)

    qc.barrier()

    qc.h(work_bit_num - 1)  

    qc = qc + cnx(n, work_bit[:-1], range(work_bit_num, n - 1), work_bit[-1])

    qc.h(work_bit_num - 1)  

    qc.barrier()

    qc.x(work_bit)

    qc.h(work_bit) 

    qc.barrier()
    return qc

def BVAlgorithm(work_bit_num):
    work_bit = range(0, work_bit_num)
    n = work_bit_num + 1
    b = np.random.randint(1, 2 ** work_bit_num)
    # Next, format 'b' as a binary string of length 'n', padded with zeros:
    b_str = format(b, '0' + str(work_bit_num) + 'b')

    print(b_str)

    qc = QuantumCircuit(n, n)

    qc.reset(range(n))

    qc.x(n - 1)

    qc.barrier()

    # And set up the input register:
    qc.h(range(n)) 

    qc.barrier()

    # oracle:
    for qubit in range(len(b_str)):
        if b_str[qubit] == '1':
            qc.cx(qubit, n - 1)

    qc.barrier()

    qc.h(work_bit)
    return qc

def dj_algorithm(case, work_bit):
    n = work_bit + 1
    qc = QuantumCircuit(n, n - 1)

    qc.reset(range(n))

    # Set up the output qubit:
    qc.x(n - 1)
    qc.barrier()
    # And set up the input register:
    for qubit in range(n):
        qc.h(qubit)
    # Let's append the oracle gate to our circuit:
    # dj_circuit.append(oracle, range(n+1))
    # First, let's deal with the case in which oracle is balanced
    if case == "balanced":
        # First generate a random number that tells us which CNOTs to
        # wrap in X-gates:
        b = np.random.randint(1, 2**(n - 1))
        # Next, format 'b' as a binary string of length 'n', padded with zeros:
        b_str = format(b, '0'+str(n - 1)+'b')
        # Next, we place the first X-gates. Each digit in our binary string 
        # corresponds to a qubit, if the digit is 0, we do nothing, if it's 1
        # we apply an X-gate to that qubit:
        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                qc.x(qubit)            
                #dj_circuit.x(qubit)
        # Do the controlled-NOT gates for each qubit, using the output qubit 
        # as the target:
        for qubit in range(n - 2, -1, -1):     
            qc.cx(qubit, n - 1)

        qc.barrier()
        # Next, place the final X-gates
        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                qc.x(qubit)
                #dj_circuit.x(qubit)
    # Case in which oracle is constant
    if case == "constant":
        # First decide what the fixed output of the oracle will be
        # (either always 0 or always 1)
        output = np.random.randint(2)
        if output == 1:
            qc.x(n - 1)
    qc.barrier()
    # Finally, perform the H-gates again and measure:
    for qubit in range(n - 1):
        qc.h(qubit)
    qc.barrier()
    return qc

def simonAlg(work_bit_num):

    n = work_bit_num * 2

    work_bit = range(0, work_bit_num)

    b = np.random.randint(1, 2 ** work_bit_num)
    #b = 2 ** work_bit_num - 1
    # Next, format 'b' as a binary string of length 'n', padded with zeros:
    b_str = format(b, '0' + str(work_bit_num) + 'b')

    print(b_str)

    qc = QuantumCircuit(n, n)

    qc.reset(range(n))
    qc.barrier()

    # And set up the input register:
    qc.h(range(work_bit_num)) 

    qc.barrier()

    # oracle:
    for i in range(work_bit_num):
        if b_str[i] == '1':
            qc.cx(i, work_bit_num + (i - 1) % work_bit_num)
            qc.cx(i, work_bit_num + i)

    qc.barrier()

    qc.h(work_bit)

    return qc

def QFT(n):
    work_bit_num = n
    work_bit = range(0, work_bit_num)

    qc = QuantumCircuit(n, n)

    qc.reset(range(n))
    qc.barrier()

    # And set up the input register:
    qc.h(range(work_bit_num)) 

    qc.barrier()

    # oracle:
    for i in range(work_bit_num):
        qc.h(i)
        for j in range(1 + i, work_bit_num):
            qc.cu1(np.pi / (2 ** (j - i)), j, i)
        qc.barrier()
    return qc

def phase_estimation(work_bit_num):
    n = work_bit_num + 1
    work_bit = range(0, work_bit_num)

    qc = QuantumCircuit(n, n)

    qc.reset(range(n))

    qc.x(n - 1)

    qc.barrier()

    # And set up the input register:
    qc.h(range(work_bit_num)) 

    qc.barrier()

    for i in range(work_bit_num):
        for j in range(2 ** i):
            qc.cu1(np.pi / 4, i, n - 1)

    qc.barrier()
    for i in range(work_bit_num - 1, -1, -1):
        for j in range(work_bit_num - 1, i, -1):
            qc.cu1(-np.pi / (2 ** (j - i)), j, i)
        qc.h(i)

    return qc


def cnx(n, control_bit, ancilla_bit, target):
    cnx = QuantumCircuit(n)
    cnx.toffoli(control_bit[0], control_bit[1], ancilla_bit[0])
    for i in control_bit[2:]:
        cnx.toffoli(control_bit[i], ancilla_bit[i - 2], ancilla_bit[i - 1])
    cnx.cx(ancilla_bit[i - 1], target)
    for i in control_bit[::-1][:-2]:
        cnx.toffoli(control_bit[i], ancilla_bit[i - 2], ancilla_bit[i - 1])
    cnx.toffoli(control_bit[0], control_bit[1], ancilla_bit[0])
    return cnx

def c_amod15(a, power):
    """Controlled multiplication by a mod 15"""
    if a not in [2,7,8,11,13]:
        raise ValueError("'a' must be 2,7,8,11 or 13")
    U = QuantumCircuit(4)        
    for iteration in range(power):
        if a in [2,13]:
            U.swap(0,1)
            U.swap(1,2)
            U.swap(2,3)
        if a in [7,8]:
            U.swap(2,3)
            U.swap(1,2)
            U.swap(0,1)
        if a == 11:
            U.swap(1,3)
            U.swap(0,2)
        if a in [7,11,13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = "%i^%i mod 15" % (a, power)
    c_U = U.control()
    return c_U

def qft_dagger(n):
    """n-qubit QFTdagger the first n qubits in circ"""
    qc = QuantumCircuit(n)
    # Don't forget the Swaps!
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cu1(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc

def shor():

    # Specify variables
    n_count = 3 # number of counting qubits
    a = 7

    # Create QuantumCircuit
    qc = QuantumCircuit(4+n_count, n_count)
    
    # Initialise counting qubits
    # in state |+>
    for q in range(n_count):
        qc.h(q)
        
    # And ancilla register in state |1>
    qc.x(3+n_count)

    # Do controlled-U operations
    for q in range(n_count):
        qc.append(c_amod15(a, 2**q), 
                 [q] + [i+n_count for i in range(4)])

    # Do inverse-QFT
    qc.append(qft_dagger(n_count), range(n_count))

    return qc

