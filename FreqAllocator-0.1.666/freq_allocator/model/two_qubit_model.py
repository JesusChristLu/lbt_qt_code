from .formula import lorentzain, freq2amp_formula
from .single_qubit_model import singq_T1_err, singq_T2_err
import networkx as nx

def is_xtalk(chip, qcq1, qcq2):
    distance = []
    for i in qcq1:
        for j in qcq2:
            if nx.has_path(chip, i, j):
                distance.append(nx.shortest_path_length(chip, i, j))
            else:
                distance.append(100000)
    if 1 in distance and not (0 in distance):
        return True
    else:
        return False

def edge_distance(chip, qcq1, qcq2):
    distance = []
    for i in qcq1:
        for j in qcq2:
            if nx.has_path(chip, i, j):
                distance.append(nx.shortest_path_length(chip, i, j))
            else:
                distance.append(100000)
    return min(distance)


def twoq_T1_err(pulse, a, tq, t1_spectrum):
    return singq_T1_err(a, tq, pulse, t1_spectrum)


def twoq_T2_err(
    pulse, a, tq, t2_spectrum: dict = None, ac_spectrum_paras: list = None, step: int = 50
):
    return singq_T2_err(a, tq, pulse, t2_spectrum, ac_spectrum_paras)


def twoq_xtalk_err(
    pulse1, pulse2, a, anharm1, anharm2
):
    return (
        (lorentzain(pulse1, pulse2, a[0], a[1])) +
        (lorentzain(pulse1 + anharm1, pulse2, a[2], a[3])) +
        (lorentzain(pulse1, pulse2 + anharm2, a[2], a[3]))
    )

def inner_leakage(pulse1, pulse2, a):
    # if min(pulse1) > max(pulse2) and max(pulse1) > min(pulse2) or \
    #    min(pulse2) > max(pulse1) and max(pulse2) > min(pulse1):
    #     return a[0]
    # else:
    #     return a[1]
    return 0


def twoq_pulse_distort_err(fi, fj, a, ac_spectrum_paras1, ac_spectrum_paras2):
    vi0 = freq2amp_formula(fi[0], *ac_spectrum_paras1)
    vi1 = freq2amp_formula(fi[1], *ac_spectrum_paras1)
    vj0 = freq2amp_formula(fj[0], *ac_spectrum_paras2)
    vj1 = freq2amp_formula(fj[1], *ac_spectrum_paras2)
    return a * (abs(vi0 - vi1) + abs(vj0 - vj1))


def twoQ_err_model(frequencys, chip, maxParallelCZ, reOptimizeQCQs, a):
    for qcq in reOptimizeQCQs:
        chip.edges[qcq]['frequency'] = frequencys[reOptimizeQCQs.index(qcq)]
    cost = 0
    for qcq in reOptimizeQCQs:
        if chip.nodes[qcq[0]]['frequency'] > chip.nodes[qcq[1]]['frequency']:
            qh, ql = qcq[0], qcq[1]
        else:
            qh, ql = qcq[1], qcq[0]

        if chip.nodes[qh]['freq_max'] + chip.nodes[qh]['anharm'] < chip.nodes[ql]['freq_min']:
            qh, ql = ql, qh

        fWork = chip.edges[qcq]['frequency']
        pulseql = fWork
        pulseqh = fWork - chip.nodes[qh]['anharm']
        cost += twoq_T1_err(
            pulseql,
            a[0],
            chip.edges[qcq]['two tq'],
            chip.nodes[ql]['T1 spectra']
        )
        cost += twoq_T1_err(
            pulseqh,
            a[0],
            chip.edges[qcq]['two tq'],
            chip.nodes[qh]['T1 spectra'],
        )
        cost += twoq_T2_err(
            pulseql,
            a[1],
            chip.edges[qcq]['two tq'],
            ac_spectrum_paras=chip.nodes[ql]['ac_spectrum'],
        )
        cost += twoq_T2_err(
            pulseqh,
            a[1],
            chip.edges[qcq]['two tq'],
            ac_spectrum_paras=chip.nodes[qh]['ac_spectrum'],
        )
        cost += twoq_pulse_distort_err(
            [pulseqh, chip.nodes[qh]['frequency']],
            [pulseql, chip.nodes[ql]['frequency']],
            a[2],
            ac_spectrum_paras1=chip.nodes[qh]['ac_spectrum'],
            ac_spectrum_paras2=chip.nodes[ql]['ac_spectrum'],
        )
        cost += inner_leakage(
            [pulseqh, chip.nodes[qh]['frequency']],
            [pulseql, chip.nodes[ql]['frequency']],
            a[3 : 5])

        for q in qcq:
            if q == ql:
                pulse = pulseql
            else:
                pulse = pulseqh
            # 计算周围pair中的一个比特周围距离1的所有比特的idle freq对这个比特产生的误差
            for neighbor in chip[q]:
                if neighbor in qcq:
                    continue
                cost += twoq_xtalk_err(
                    pulse,
                    chip.nodes[neighbor]['frequency'],
                    a[5:],
                    chip.nodes[q]['anharm'],
                    chip.nodes[neighbor]['anharm']
                )

        for neighbor in chip.edges:
            if chip.edges[neighbor].get('frequency', False) and neighbor in maxParallelCZ and \
                is_xtalk(chip, qcq, neighbor):
                for q0 in qcq:
                    for q1 in neighbor:
                        # 判定两个pair之间是qh-qh、qh-ql、ql-qh、ql-ql
                        if (q0, q1) in chip.edges:
                            if q0 == ql:
                                pulse = pulseql
                            else:
                                pulse = pulseqh

                            if (
                                chip.nodes[neighbor[0]]['frequency']
                                < chip.nodes[neighbor[1]]['frequency']
                            ):
                                q1l, q1h = neighbor[0], neighbor[1]
                            else:
                                q1l, q1h = neighbor[1], neighbor[0]

                            if chip.nodes[q1h]['freq_max'] + chip.nodes[q1h]['anharm'] < chip.nodes[q1l]['freq_min']:
                                q1l, q1h = q1h, q1l

                            if q1 == q1l:
                                nPulse = chip.edges[neighbor]['frequency']
                            else:
                                nPulse = (
                                    chip.edges[neighbor]['frequency']
                                    - chip.nodes[q1]['anharm']
                                )

                            cost += twoq_xtalk_err(
                                pulse,
                                nPulse,
                                a[5:],
                                chip.nodes[q0]['anharm'],
                                chip.nodes[q1]['anharm']
                            )
    return cost / len(reOptimizeQCQs)
