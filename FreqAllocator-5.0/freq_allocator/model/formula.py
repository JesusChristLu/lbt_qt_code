import numpy as np
from typing import Union
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

rho_qc = 0.031
rho_qq = 0.0034
fc = 7000

def eff_g(f1, f2):
    f1 = f1 * 1e-3
    f2 = f2 * 1e-3
    ffc = fc * 1e-3
    return np.abs(rho_qq * np.sqrt(f1 * f2) + 0.5 * (rho_qc ** 2 * ffc * np.sqrt(f1 * f2)) * (1 / (f1 - ffc) + 1 / (f2 - ffc) - 1 / (f1 + ffc) - 1 / (f2 + ffc)))

def gen_pos(chip):
    pos = dict()
    for qubit in chip:
        pos[qubit] = [chip.nodes[qubit]['coord'][1], -chip.nodes[qubit]['coord'][0]]
    return pos


def amp2freq_formula(
    x: Union[float, np.ndarray],
    fq_max: float,
    detune: float,
    M: float,
    d: float,
    w: float = None,
    g: float = None,
    tans2phi: bool = False  # 判定传入的是amp还是phi
):
    r"""Calculate frequency from AC.

    .. math::
        phi = \pi \ast M \ast (x - offset)

    .. math::
        fq = (fq\_max + detune) \times \sqrt{\sqrt{1 + d^2 (\tan (phi))^2} \times \left | \cos (phi) \right | }
    """
    if tans2phi:
        phi = x
    else:
        phi = np.pi * M * x
    fq = (fq_max + detune) * np.sqrt(
        np.sqrt(1 + d**2 * np.tan(phi) ** 2) * np.abs(np.cos(phi))
    ) - detune
    if w:
        fg = np.sqrt((w - fq) ** 2 + 4 * g ** 2)
        fq = (w + fq + fg) / 2
    return fq


def freq2amp_formula(
    x: float,
    fq_max: float,
    detune: float,
    M: float,
    d: float,
    w: float = None,
    g: float = None,
    tans2phi: bool = False,
):
    r"""Calculate AC based on frequency

    .. math::
        \alpha = \frac{x + detune }{detune + fq_{max}}

    .. math::
        \beta = \frac{{\alpha}^4 - d^2}{1 - d^2}

    .. math::
        amp = \left | \frac{\arccos \beta}{M\cdot \pi}  \right | + offset
    """
    if w:
        x = x - g ** 2 / (x - w)
    else:
        x = x
    alpha = (x + detune) / (detune + fq_max)
    belta = (alpha ** 4 - d ** 2) / (1 - d ** 2)

    if belta < 0 or belta > 1:
        if np.abs(belta) < 1e-3:
            belta = 0.0
        elif np.abs(belta) - 1 < 1e-3:
            belta = 1.0
        else:
            print('???', belta, x, fq_max, detune, M, d, w, g)

    # assert belta >= 0
    # assert belta <= 1
    phi = np.abs(np.arccos(np.sqrt(belta)))
    amp = phi / (M * np.pi)

    if tans2phi:
        return phi
    else:
        return amp

def lorentzain(fi, fj, a, gamma):
    wave = (1 / np.pi) * (gamma / ((fi - fj) ** 2 + (gamma) ** 2))
    return a * wave


def freq_var_map(f, allowFreq):
    rangeLen = sum([af[1] - af[0] for af in allowFreq])
    percent = []
    startPercent = 0
    retF = 0
    for fr in allowFreq:
        percent.append([startPercent, startPercent + (fr[1] - fr[0]) / rangeLen])
        startPercent = startPercent + (fr[1] - fr[0]) / rangeLen
    percent[-1][-1] = 1.0
    for pc in percent:
        if f >= pc[0] and f <= pc[1]:
            pcindex = percent.index(pc)
            retF = allowFreq[pcindex][0] + (
                allowFreq[pcindex][1] - allowFreq[pcindex][0]
            ) * (f - pc[0]) / (pc[1] - pc[0])
    if retF == 0:
        print(f, percent, allowFreq)
    return retF

def draw_chip(chip, name='', qubit_err=None, qcq_err=None, qubit_freq=None, qcq_freq=None):
    pos = gen_pos(chip)
    nodeLabelDict = dict([(i, chip.nodes[i]['name']) for i in chip.nodes])
    if not(qubit_err is None):
        qubitErrDict = dict([(i, str(round(qubit_err[list(chip.nodes).index(i)], 4))) for i in chip.nodes])
    if not(qcq_err is None):
        qcqErrDict = dict([(i, str(round(qcq_err[list(chip.edges).index(i)], 4))) for i in chip.edges])
    if not(qubit_freq is None):
        qubitFreqDict = dict([(i, str(round(qubit_freq[list(chip.nodes).index(i)]))) for i in chip.nodes])
    if not(qcq_freq is None):
        qcqFreqDict = dict([(i, str(round(qcq_freq[list(chip.edges).index(i)]))) for i in chip.edges])
        
    plt.figure(figsize=(7, 10))
    if not(qubit_err is None):
        nx.draw_networkx_labels(chip, pos, labels=qubitErrDict, font_size=6, font_color="red")
        nx.draw_networkx_nodes(chip, pos, nodelist=chip.nodes, node_color=qubit_err, cmap=plt.cm.plasma)
        
        # # 创建Colorbar并嵌入到图中
        # norm = plt.Normalize(vmin=min(qubit_err), vmax=max(qubit_err))
        # sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
        # sm.set_array([])
        # cbar = plt.colorbar(sm, ax=plt.gca())
        # cbar.set_label('Qubit Error Rate')
    elif not(qubit_freq is None):
        nx.draw_networkx_labels(chip, pos, qubitFreqDict, font_size=6, font_color="red")
        nx.draw_networkx_nodes(chip, pos, nodelist=chip.nodes, node_color=qubit_freq, cmap=plt.cm.plasma)
    else:
        nx.draw_networkx_labels(chip, pos, labels=nodeLabelDict, font_size=6, font_color="red")
        nx.draw_networkx_nodes(chip, pos, nodelist=chip.nodes)

    if not(qcq_err is None):
        nx.draw_networkx_edges(chip, pos, edgelist=chip.edges, edge_color=qcq_err, edge_cmap=plt.cm.plasma, width=8)
        nx.draw_networkx_edge_labels(chip, pos, edge_labels=qcqErrDict, font_size=6, font_color="red", bbox=dict(facecolor='none', edgecolor='none', pad=1))
        
        # 创建Colorbar并嵌入到图中
        # norm = plt.Normalize(vmin=min(qcq_err), vmax=max(qcq_err))
        # sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
        # sm.set_array([])
        # cbar = plt.colorbar(sm, ax=plt.gca())
        # cbar.set_label('qcq Error Rate')
    elif not(qcq_freq is None):
        nx.draw_networkx_edges(chip, pos, edgelist=chip.edges, edge_color=qcq_freq, edge_cmap=plt.cm.plasma, width=8)
        nx.draw_networkx_edge_labels(chip, pos, edge_labels=qcqFreqDict, font_size=6, font_color="red", bbox=dict(facecolor='none', edgecolor='none', pad=1))
    else:
        nx.draw_networkx_edges(chip, pos, edgelist=chip.edges, width=8)
        

    plt.axis('off')
    plt.savefig(name + '.pdf', dpi=300)
    plt.close()

def scatter_err(labelList, errList, name):
    plt.scatter([range(len(labelList))], errList, color='blue', alpha=0.5, s=100)
    plt.axhline(y=0.01, color='red', linestyle='--')
    # plt.semilogy()
    plt.savefig(name + '.pdf', dpi=300)
    plt.close()