import qpandalite
import math
import numpy as np
import time
import random
from datetime import datetime
from itertools import combinations
import qpandalite.task.originq_dummy as originq_virt #run on virtual QC chip
import qpandalite.task.origin_qcloud as  originq_real#run on originQ QC chip


def read_fer_pauli_dic(fer_to_pauli_file):
    #Read pauli Hamiltonian, result in a dictionary like {"Z1 Z2": 0.12, "Z1 X Y": 1.11}
    fo = open(fer_to_pauli_file, "r")
    fer_pauli_dic = {}
    for line in fo.readlines():
        line = line.strip('\n')
        op_str, tmp_val = line.split(':')
        fer_pauli_dic[op_str] = float(tmp_val)
    fo.close()
    return fer_pauli_dic

def gen_rand_Pauli(n_q, n_it):
    rand_Hami = {}
    random.seed(42)
    while(len(rand_Hami) < n_it):
        P_str = ""
        for q in range(n_q):
            rint = random.randint(0, 2)

            if(rint == 0):
                P_str += "X" + str(q) + " "
            if(rint == 1):
                P_str += "Y" + str(q) + " "
            if(rint == 2):
                P_str += "Z" + str(q) + " "

        rand_Hami[P_str.strip()] = 1.0

    return rand_Hami


def build_circuit(mapping):    
    # Start from here.
    c = qpandalite.Circuit()

    # Define the gates with the simplest format.
    c.x(0)
    c.x(1)
    #c.cnot(0,1)
    # Measure the qubits you want.
    c.measure(*[0,1,2])

    # Use this to create a remapping (this is NOT an inplace operation.)
    c = c.remapping(mapping)
    return c.circuit

def prepare_init_stat(n_q, n_e, stat_type, tmp_qmap, para_vec):
    if(stat_type == "HF"):
        cirq = qpandalite.Circuit()
        for i in range(n_e):
            cirq.x(i)
        return cirq
    elif(stat_type == "any_rand"):
        if(tmp_qmap == {}):
            para_id = 0
            cirq = qpandalite.Circuit()
            for j in range(n_q):
                cirq.ry(j, para_vec[para_id])
                para_id += 1
            for j in range(n_q-1):
                cirq.cnot(j, j+1)
            for j in range(n_q):
                cirq.rx(j, para_vec[para_id])
                para_id += 1
            return cirq

    elif(stat_type == "sig_rand"):
        if(tmp_qmap == {}):
            para_id = 0
            cirq = qpandalite.Circuit()
            for j in range(n_q):
                cirq.ry(j, para_vec[para_id])
                para_id += 1
            for j in range(n_q):
                cirq.rx(j, para_vec[para_id])
                para_id += 1
            return cirq
    
            
        key_vec = []
        for key, val in tmp_qmap.items():
            key_vec.append(key)
        para_id = 0
        cirq = qpandalite.Circuit()
        for j in range(n_q):
            cirq.ry(key_vec[j], para_vec[para_id])
            para_id += 1
        
        for j in range(len(key_vec)-1):
            cirq.cnot(key_vec[j], key_vec[j + 1])
        for j in range(n_q):
            cirq.rx(key_vec[j], para_vec[para_id])
            para_id += 1
        return cirq
    else:
        return "ERR"

def meas_one_term(pstr, ocirq, meas_type):# void empty pstr!!!!!!!!!!!!
    m_qubits = []# measure qubits
    for op in pstr.split():
        if(op[0] == 'X'):
            ocirq.h(int(op[1:]))
        if(op[0] == 'Y'):
            ocirq.rx(int(op[1:]), np.pi/2)
        m_qubits.append(int(op[1:]))              
    if(meas_type == "NO_CNOT" or len(pstr.split()) < 2):
        ocirq.measure(*m_qubits)
        meas_str = ''
        for j in m_qubits:
            meas_str += str(j) + ' '
        return ocirq, meas_str.rstrip()
    elif(meas_type == "USE_CNOT"):
        op_vec = pstr.split()
        for j in range(len(op_vec)-1):
            if(abs(int(op_vec[j][1:]) - int(op_vec[j+1][1:])) < 1.5):
                ocirq.cnot(int(op_vec[j][1:]), int(op_vec[j+1][1:]))
                m_qubits.remove(int(op_vec[j][1:]))
        ocirq.measure(*m_qubits)
        meas_str = ''
        for j in m_qubits:
            meas_str += str(j) + ' '
        return ocirq, meas_str.rstrip()
    else:
        return "meas_type ERR!!!"

def count_parity(res_str):
    parity = 0
    for s in res_str:
        if(s == "1"):
            parity += 1
    return parity % 2

def trans_result_expec(result, resp_M_inv):

    
    h_expec = 0.0
    res_str = []
    val = []
    if(resp_M_inv != []):
        for k in range(resp_M_inv.shape[0]):
            res_str.append(k)
            if(k not in result):
                val.append(0.0)
            else:
                val.append(result[k])
        #print("resp_M")
        #print(np.linalg.pinv(resp_M_inv))
        #print("resp_M_inv:")
        #print(resp_M_inv)
       # print("val:")
        #print(val)
        val = np.array(resp_M_inv) @ np.array(val)
        #print("res:")
        #print(val)
        for j in range(len(val)):
            h_expec += 2 * (0.5 - count_parity(bin(int(res_str[j]))[2:])) * float(val[j])
        #print("123344555")
    else:
        for key, prb in result.items():
            h_expec += 2 * (0.5 - count_parity(bin(int(key))[2:])) * float(prb)
        
    return h_expec

        
def std_trans_result_expec(result):
    h_expec = 0.0
    res_str = result["key"]
    val = result["value"]
    for j in range(len(val)):
        h_expec += 2 * (0.5 - count_parity(res_str[j])) * val[j]
    return h_expec    
    
def get_all_M_circuit(n_q):
    resp_M = {}
    resp_cir = []
    qu_list = [i for i in range(n_q)]
    for j in range(1, n_q + 1):
        n_j_list_list = list(combinations(qu_list, j))
        for n_j_list in n_j_list_list:

            bstr = [format(i, '0' + str(j) + 'b') for i in range(2**j)]
            for s in bstr:
                cirq = qpandalite.Circuit()
                for q in range(j):
                    if(s[q] == '1'):
                        cirq.x(n_j_list[q])
                        
                cirq.measure(*n_j_list)
                resp_cir.append(cirq.remapping(qubit_map).circuit)
    #            print(resp_cir[-1])
    return resp_cir

def get_part_M_circuit(n_q):
    resp_M = {}
    resp_cir = []
    qu_list = [i for i in range(n_q)]
    for j in [1, n_q]:
        n_j_list_list = list(combinations(qu_list, j))
        for n_j_list in n_j_list_list:

            bstr = [format(i, '0' + str(j) + 'b') for i in range(2**j)]
            for s in bstr:
                cirq = qpandalite.Circuit()
                for q in range(j):
                    if(s[q] == '1'):
                        cirq.x(n_j_list[q])
                        
                cirq.measure(*n_j_list)
                resp_cir.append(cirq.remapping(qubit_map).circuit)
    #            print(resp_cir[-1])
    return resp_cir



def get_all_inverse_matrix(n_q, result_real):
    resp_M = {}
    resp_cir = get_all_M_circuit(n_q)
    qu_list = [i for i in range(n_q)]
    for j in range(1, n_q + 1):
        n_j_list_list = list(combinations(qu_list, j))
        for n_j_list in n_j_list_list:
            resp_M[' '.join(str(nu) for nu in n_j_list)] = np.zeros((2**j, 2**j))
    if(result_real == {}):
        return resp_M
    #for key, val in resp_M.items():
    #    print(key)
    
    res_ix = 0
    for key, val in resp_M.items():
        #print("key: ", key)
        bstr = [format(i, '0' + str(len(key.split()))+ 'b') for i in range(2**len(key.split()))]
        for s in bstr:
            #print("s: ", s)
            res_now = result_real[res_ix]
            #print("res_now:", res_now)
            res_ix += 1
            m_col = int(s, 2)
            if(len(key.split()) <= 0):
                print("s is:")
                print(s)
                print("res is:")
                print(res_now)
            for bas, prob in res_now.items():
                #bas_bin =  "{:0" + str(len(key.split())) + "d}".format(original_integer)
                #bas_bin = str(bas).zfill(len(key.split()))
                #bas = int(bas_bin[::-1], 2)                
                resp_M[key][bas][m_col] = prob

    for key, val in resp_M.items():
        resp_M[key] = np.linalg.inv(val)
    return resp_M


def get_part_inverse_matrix(n_q, result_real):
    resp_M = {}
    resp_cir = get_part_M_circuit(n_q)
    qu_list = [i for i in range(n_q)]
    for j in [1, n_q]:
        n_j_list_list = list(combinations(qu_list, j))
        for n_j_list in n_j_list_list:
            resp_M[' '.join(str(nu) for nu in n_j_list)] = np.zeros((2**j, 2**j))
    if(result_real == {}):
        return resp_M
    #for key, val in resp_M.items():
    #    print(key)
    
    res_ix = 0
    for key, val in resp_M.items():
        bstr = [format(i, '0' + str(len(key.split()))+ 'b') for i in range(2**len(key.split()))]
        for s in bstr:
            res_now = result_real[res_ix]
            res_ix += 1
            m_col = int(s, 2)
            if(len(key.split()) <= 0):
                print("s is:")
                print(s)
                print("res is:")
                print(res_now)
            for bas, prob in res_now.items():
                #bas_bin =  "{:0" + str(len(key.split())) + "d}".format(original_integer)
                #bas_bin = str(bas).zfill(len(key.split()))
                #bas = int(bas_bin[::-1], 2)                
                resp_M[key][bas][m_col] = prob

    for key, val in resp_M.items():
        resp_M[key] = np.linalg.inv(val)
    return resp_M





def write_all_inv_M(n_q):
    resp_cir = get_all_M_circuit(n_q)
    
    taskid = originq_real.submit_task(resp_cir, shots = shots, auto_mapping = False, measurement_amend=False)
    taskid = originq_real.get_last_taskid()
    result = originq_real.query_by_taskid_sync(taskid, interval=60.0, # query interval (seconds)
                                              timeout=30000000.0, # max timeout (seconds)
                                              retry=30000000) # max retries for exceptions
    result = originq_real.query_by_taskid(taskid)
    #result_real = qpandalite.convert_originq_result(result, style='keyvalue',
    #                                                prob_or_shots='shots', key_style='bin', reverse_key = True)
    result_ky_lst = result['result']
    result_real = []
    for dict_it in result_ky_lst:
        #print(dict_it)
        key_lst = dict_it['key']
        #print(key_lst)
        val_lst = dict_it['value']
        result_dic = {}
        key_max = 0
        for j in range(len(key_lst)):
            #print(key_lst[j])
            key_max = max(key_max, int(key_lst[j][2:], 16))
        n_q_now = 0
        while(2**n_q_now < key_max):
            n_q_now += 1
        for j in range(len(key_lst)):
            #print(key_lst[j])
            base_id = int(key_lst[j][2:], 16)
            base_bin = format(base_id, '0' + str(n_q_now)+ 'b')
            #print(base_bin)
            base_inv = base_bin[::inv_b]
            base_int = int(base_inv, 2)
            if(base_int > 15 or base_int < 0):
                print(base_int)
            result_dic[base_int] = val_lst[j]
        result_real.append(result_dic)
        
    inv_M = get_all_inverse_matrix(n_q, result_real)


    res_path = "C:/Users/52824/Desktop/Work/Qpanda_Experiment/MResult/all_" + run_time
    for key, val in inv_M.items():
        key_line = ""
        for s in key.split():
            key_line += s + "_"
        file_path = res_path + "_" +  key_line.strip() + ".txt"
        np.savetxt(file_path, val, fmt='%.10f', delimiter='\t')

    return inv_M

def write_part_inv_M(n_q):
    resp_cir = get_part_M_circuit(n_q)
    
    taskid = originq_real.submit_task(resp_cir, shots = shots, auto_mapping = False, measurement_amend=False)
    taskid = originq_real.get_last_taskid()
    result = originq_real.query_by_taskid_sync(taskid, interval=60.0, # query interval (seconds)
                                              timeout=30000000.0, # max timeout (seconds)
                                              retry=30000000) # max retries for exceptions
    result = originq_real.query_by_taskid(taskid)
    #result_real = qpandalite.convert_originq_result(result, style='keyvalue',
    #                                                prob_or_shots='shots', key_style='bin', reverse_key = True)
    result_ky_lst = result['result']
    result_real = []
    for dict_it in result_ky_lst:
        #print(dict_it)
        key_lst = dict_it['key']
        #print(key_lst)
        val_lst = dict_it['value']
        result_dic = {}
        key_max = 0
        for j in range(len(key_lst)):
            #print(key_lst[j])
            key_max = max(key_max, int(key_lst[j][2:], 16))
        n_q_now = 0
        while(2**n_q_now < key_max):
            n_q_now += 1
        for j in range(len(key_lst)):
            #print(key_lst[j])
            base_id = int(key_lst[j][2:], 16)
            base_bin = format(base_id, '0' + str(n_q_now)+ 'b')
            base_inv = base_bin[::inv_b]
            base_int = int(base_inv, 2)
            result_dic[base_int] = val_lst[j]
        result_real.append(result_dic)
    
    inv_M = get_part_inverse_matrix(n_q, result_real)


    #res_path = "/home/lqs/Work/Qpanda_Experiment/MResult/" + run_time
    res_path = "C:/Users/52824/Desktop/Work/Qpanda_Experiment/MResult/part_" + run_time
    for key, val in inv_M.items():
        key_line = ""
        for s in key.split():
            key_line += s + "_"
        file_path = res_path +  "_" +  key_line.strip() + ".txt"
        np.savetxt(file_path, val, fmt='%.10f', delimiter='\t')

    return inv_M



def get_Hami_meas_cirq(Hami_dic):
    cirq_grp = []
    cirq_std = []
    meas_no_str = {}
    meas_cnot_str = {}
    np.random.seed(int(time.time()))
    para_vec = (np.random.rand(2 * n_q) - 0.5) * 4 * np.pi
    for pstr, h_val in Hami_dic.items():
        #print("pstr: ", pstr)
        if(len(pstr) < 2):
            continue
        m_qubits = []
        op_vec = pstr.split()
        for op in op_vec:
            m_qubits.append(int(op[1:]))
            tmp_qmap = {}
        q_id = 0
        for j in range(len(m_qubits)):
            tmp_qmap[m_qubits[j]] = qubit_map[q_id]
            q_id += 1
        for k in range(n_q):
            if(k not in m_qubits):
                tmp_qmap[k] = qubit_map[q_id]
                q_id += 1
        #print(tmp_qmap)
            
        #orig_cirq =   
        #init_cirq = prepare_init_stat(n_q, n_e, stat_type, {}, para_vec)
        init_cirq = qpandalite.Circuit()
        init_cirq.x(0)
        tmp_cirq, meas_str = meas_one_term(pstr, init_cirq, "NO_CNOT")
        meas_no_str[pstr] = meas_str
  
        tmp_cirq = tmp_cirq.remapping(qubit_map)
        cirq_grp.append(tmp_cirq.circuit)
        cirq_std.append(tmp_cirq.circuit)
        
        init_cirq = prepare_init_stat(n_q, n_e, stat_type, {}, para_vec) 
        tmp_cirq, meas_str = meas_one_term(pstr, init_cirq, "USE_CNOT")
        meas_cnot_str[pstr] = meas_str
        tmp_cirq = tmp_cirq.remapping(qubit_map)
        cirq_grp.append(tmp_cirq.circuit)
    return cirq_grp, cirq_std, meas_no_str, meas_cnot_str



def get_expectation(n_it, resp_inv_M):

    #set Hamiltonian path
    Hami_path = work_path + "my_source/Pauli_Hamiltonian/PH_230515/" + mole + "_" + fq_map + ".txt"
    #get Hamiltonian dictionaryi
    if(Hami_type == "rand"):
        Hami_dic = gen_rand_Pauli(n_q, n_it)
    else:
        Hami_dic = read_fer_pauli_dic(Hami_path)
    #print(Hami_dic)
    #Hami_dic = {"Z1 Z2": 1.2, "X3": 2.3}
    
    
    no_cnot_file = res_path + "_outfix_" + str(out_fix) + "_no_cnot_intime_fix.txt"
    use_cnot_file = res_path + "_outfix_" + str(out_fix) + "_use_cnotintime_fix.txt"
    no_cnot_ene_file = res_path + "_outfix_" + str(out_fix) + "_no_cnot_ene.txt"
    use_cnot_ene_file = res_path + "_outfix_" + str(out_fix) + "_use_cnot_ene.txt"

    for t in range(samp_t):
        print("sample time is: ", t)
        resp_cir = []
        if(in_fix == True):
            resp_cir = get_all_M_circuit(n_q)

        cirq_grp, cirq_std, meas_no_str, meas_cnot_str = get_Hami_meas_cirq(Hami_dic)
        no_cnot_err_vec = []
        use_cnot_err_vec = []
        std_ene_vec = []

        
        print(len(cirq_grp))


        #exit()
        taskid = originq_real.submit_task(resp_cir + cirq_grp, shots = shots, auto_mapping = False, measurement_amend=False)
        taskid = originq_real.get_last_taskid()
        result = originq_real.query_by_taskid_sync(taskid, interval=120.0, # query interval (seconds)
                                              timeout=30000000.0, # max timeout (seconds)
                                              retry=30000000) # max retries for exceptions
        result = originq_real.query_by_taskid(taskid)
        #result_real = qpandalite.convert_originq_result(result, style='keyvalue', prob_or_shots='shots', key_style='bin', reverse_key = True)
        result_ky_lst = result['result']
        result_real = []
        for dict_it in result_ky_lst:
            #print(dict_it)
            key_lst = dict_it['key']
            #print(key_lst)
            val_lst = dict_it['value']
            result_dic = {}
            key_max = 0
            for j in range(len(key_lst)):
            #print(key_lst[j])
                key_max = max(key_max, int(key_lst[j][2:], 16))
            n_q_now = 0
            while(2**n_q_now < key_max):
                n_q_now += 1
            for j in range(len(key_lst)):
                #print(key_lst[j])
                base_id = int(key_lst[j][2:], 16)
                base_bin = format(base_id, '0' + str(n_q)+ 'b')
                base_inv = base_bin[::inv_b]
                base_int = int(base_inv, 2)
                result_dic[base_int] = val_lst[j]
            result_real.append(result_dic)
        #print(result_real)


        print("result_status!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print(result)
        taskid_std = originq_virt.submit_task(cirq_std, shots = shots, task_name='measure energy')
        result_std = originq_virt.query_by_taskid_sync(taskid_std, interval=2.0, # query interval (seconds)
                                              timeout=30000.0, # max timeout (seconds)
                                              retry=30000) # max retries for exceptions
        task_status_and_result_std = originq_virt.query_by_taskid(taskid_std)

        
        if task_status_and_result_std['status'] == 'success':
            result_std = task_status_and_result_std['result']
        
        res_ix = 0 
        if(in_fix == True):
            resp_inv_M = get_all_inverse_matrix(n_q, result_real)
            res_ix = len(resp_cir)
            print("use fix")
        
        no_cnot_ene = 0.0
        use_cnot_ene = 0.0
        std_ene = 0.0
        n_it = 0
        for pstr, val in Hami_dic.items():
            if(len(pstr) < 2):
                no_cnot_ene += val
                use_cnot_ene += val
                std_ene += val
                continue
            #print("pstr: ", pstr)
        #    print("h_val: ", val)
            #print("result: ", result[n_it])
            #print("result: ", result[n_it+1])
         #   print("expec: ", trans_result_expec(result[n_it]))
         #   print("\n")
            std_ene += val * std_trans_result_expec(result_std[n_it//2])
            #no_cnot_ene += val * trans_result_expec(result_real[res_ix + n_it], resp_M_inv[meas_no_str[pstr]])
            #use_cnot_ene += val * trans_result_expec(result_real[res_ix + n_it+1], resp_M_inv[meas_cnot_str[pstr]])
            #no_cnot_err_vec.append(abs(val * trans_result_expec(result_real[res_ix + n_it], resp_M_inv[meas_no_str[pstr]]) - val * std_trans_result_expec(result_std[n_it//2])))
            #use_cnot_err_vec.append(abs(val * trans_result_expec(result_real[res_ix + n_it+1], resp_M_inv[meas_cnot_str[pstr]]) - val * std_trans_result_expec(result_std[n_it//2])))
            if(resp_inv_M == []):
                #print("no fix")
                no_cnot_ene += val * trans_result_expec(result_real[res_ix + n_it], [])
                use_cnot_ene += val * trans_result_expec(result_real[res_ix + n_it+1], [])
                no_cnot_err_vec.append(abs(val * trans_result_expec(result_real[res_ix + n_it], []) - val * std_trans_result_expec(result_std[n_it//2])))
                use_cnot_err_vec.append(abs(val * trans_result_expec(result_real[res_ix + n_it+1], []) - val * std_trans_result_expec(result_std[n_it//2])))
                std_ene_vec.append(val * std_trans_result_expec(result_std[n_it//2]))
            else:
                #print("use fix")
                no_cnot_ene += val * trans_result_expec(result_real[res_ix + n_it], resp_inv_M[meas_no_str[pstr]])
                use_cnot_ene += val * trans_result_expec(result_real[res_ix + n_it+1], resp_inv_M[meas_cnot_str[pstr]])
                no_cnot_err_vec.append(abs(val * trans_result_expec(result_real[res_ix + n_it], resp_inv_M[meas_no_str[pstr]]) - val * std_trans_result_expec(result_std[n_it//2])))
                use_cnot_err_vec.append(abs(val * trans_result_expec(result_real[res_ix + n_it+1], resp_inv_M[meas_cnot_str[pstr]]) - val * std_trans_result_expec(result_std[n_it//2])))
                std_ene_vec.append(val * std_trans_result_expec(result_std[n_it//2]))
            n_it += 2
        print("no cnot energy vec: ", no_cnot_err_vec)
        print("use cnot energy vec: ", use_cnot_err_vec)
        print("std energy vec: ", std_ene_vec)

        print("no cnot energy: ", no_cnot_ene)
        print("use cnot energy: ", use_cnot_ene)
        print("std energy: ", std_ene)
        print("no cnot energy error: ", abs(no_cnot_ene - std_ene))
        print("use cnot energy error: ", abs(use_cnot_ene - std_ene))
        #aver_no_cnot += abs((no_cnot_ene-std_ene) / std_ene + 0.00001))
        #aver_use_cnot += abs((use_cnot_ene-std_ene) / std_ene + 0.00001)
        #write error of each state to file
        with open(no_cnot_file, 'a') as nf:
            nf.write("mapping: " + fq_map + '\n')
            #nf.write(str(abs((no_cnot_ene-std_ene) / std_ene)) + '\n')
            nf.write("no_cnot_err_vec sample time " + str(t) + '\n')
            for re in no_cnot_err_vec:
                nf.write(str(re) + '\n') 
        nf.close()
        with open(no_cnot_ene_file, 'a') as nf:
            nf.write(str(abs((no_cnot_ene-std_ene) / std_ene)) + '\n')
        nf.close()

        
        with open(use_cnot_file, 'a') as uf:
            uf.write("mapping: " + fq_map + '\n')
            uf.write(str(abs((use_cnot_ene-std_ene) / std_ene)) + '\n')
            uf.write("use_cnot_err_vec sample time " + str(t) + '\n')
            for re in use_cnot_err_vec:
                uf.write(str(re) + '\n')
        uf.close
        with open(use_cnot_ene_file, 'a') as uf:
            uf.write(str(abs((use_cnot_ene-std_ene) / std_ene)) + '\n')
        uf.close
        
        print("no cnot average error: ", sum(no_cnot_err_vec) / len(no_cnot_err_vec))
        print("use cnot average error: ", sum(use_cnot_err_vec) / len(use_cnot_err_vec))
        time.sleep(3)
    

    

if __name__ == '__main__':
    time.sleep(10)
    qubit_map = {0 : 20, 1: 26, 2: 32, 3: 33, 4: 39, 5: 38, 6: 44, 7:50}

    run_now = datetime.now()
    run_time = str(run_now.month) + "_" + str(run_now.day) + "_" + str(run_now.hour) + "_" + str(run_now.minute)
    #work_path = "C:/Users/52824/Desktop/Work/"
    #res_path = "C:/Users/52824/Desktop/Work/Qpanda_Experiment/Result/" + run_time
    work_path = "/home/lqs/Work/"
    res_path = "/home/lqs/Work/Qpanda_Experiment/Result/" + run_time
    #set molecular
    Hami_type = "mol" #rand, mol
    mole = "H2"
    n_it = 60############################################################################
    n_q = 4#quit number####################################################################
    n_e = 2 #elec number
    #set_fermion to qubit mapping
    fq_map = "JW"
    # ready for mapping the circuit
    
    shots = 1000######################################################################
    samp_t = 5
    stat_type = "HF" #HF  any_rand sig_rand
   

    out_fix = False
    in_fix = False##################################################################################

    inv_b = 1##################################################

#    get_all_M_circuit(n_q)
#    get_all_inverse_matrix(n_q, {})
    print("Hami is: ", Hami_type) 
    print("mol is: ", mole)
    print("n_q is: ", n_q)
    print("stat_type is: ", stat_type)
    print("out_fix is: ", out_fix)
    if(out_fix == True):
        resp_inv_M = write_all_inv_M(n_q)
        #resp_inv_M = write_part_inv_M(n_q)
    else:
        resp_inv_M =[]
    get_expectation(n_it, resp_inv_M)

