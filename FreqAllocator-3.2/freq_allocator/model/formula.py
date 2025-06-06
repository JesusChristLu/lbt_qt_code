import numpy as np
from typing import Union
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def draw_chip(chip, name='', err=None, freq=None, centerNode=None, minMaxErr=None, minMaxFreq=None, bar=False):
    pos = gen_pos(chip)
    nodeLabelDict = dict([(i, chip.nodes[i]['name']) for i in chip.nodes])
    if not(err is None):
        qubitErrDict = dict([(i, err[list(chip.nodes).index(i)]) for i in chip.nodes])
        qcqErrDict = dict([(i, err[len(chip.nodes) + list(chip.edges).index(i)]) for i in chip.edges])
        qubitErrLabelDict = dict([(i, str(round(err[list(chip.nodes).index(i)], 5))) for i in chip.nodes])
        qcqErrLabelDict = dict([(i, str(round(err[len(chip.nodes) + list(chip.edges).index(i)], 5))) for i in chip.edges])
    if not(freq is None):
        qubitFreqDict = dict([(i, freq[list(chip.nodes).index(i)]) for i in chip.nodes])
        qcqFreqDict = dict([(i, freq[len(chip.nodes) + list(chip.edges).index(i)]) for i in chip.edges])
        qubitFreqLabelDict = dict([(i, str(freq[list(chip.nodes).index(i)])) for i in chip.nodes])
        qcqFreqLabelDict = dict([(i, str(freq[len(chip.nodes) + list(chip.edges).index(i)])) for i in chip.edges])
        
    plt.figure(figsize=(5, 9))

    if not(err is None):
        cmap = plt.cm.coolwarm
        # norm = plt.Normalize(vmin=minMaxErr[0], vmax=minMaxErr[1])
        norm = plt.Normalize(vmin=minMaxErr[0], vmax=0.02)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        node_colors = [sm.to_rgba(qubitErrDict[node]) for node in chip.nodes]
        edge_colors = [sm.to_rgba(qcqErrDict[edge]) for edge in chip.edges]
        # nx.draw_networkx_labels(chip, pos, labels=qubitErrLabelDict, font_size=6, font_color="red")
        nx.draw_networkx_nodes(chip, pos, nodelist=chip.nodes, node_size=500, node_color=node_colors, cmap=cmap)
        nx.draw_networkx_edges(chip, pos, edgelist=chip.edges, edge_color=edge_colors, edge_cmap=cmap, edge_vmin=norm.vmin, edge_vmax=norm.vmax, width=16)
        # nx.draw_networkx_edge_labels(chip, pos, edge_labels=qcqErrLabelDict, font_size=6, font_color="red", bbox=dict(facecolor='none', edgecolor='none', pad=1))
        
        if bar:
            # 创建Colorbar并嵌入到图中
            cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.02, pad=0.000000001, orientation='horizontal')
            cbar.set_label('error Rate')
            
    elif not(freq is None):
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=minMaxFreq[0], vmax=minMaxFreq[1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        node_colors = [sm.to_rgba(qubitFreqDict[node]) for node in chip.nodes]
        edge_colors = [sm.to_rgba(qcqFreqDict[edge]) for edge in chip.edges]
        # nx.draw_networkx_labels(chip, pos, qubitFreqLabelDict, font_size=6, font_color="red")
        nx.draw_networkx_nodes(chip, pos, nodelist=chip.nodes, node_size=500, node_color=node_colors, cmap=cmap)
        nx.draw_networkx_edges(chip, pos, edgelist=chip.edges, edge_color=edge_colors, edge_cmap=cmap, edge_vmin=norm.vmin, edge_vmax=norm.vmax,  width=16)
        # nx.draw_networkx_edge_labels(chip, pos, edge_labels=qcqFreqLabelDict, font_size=6, font_color="red", bbox=dict(facecolor='none', edgecolor='none', pad=1))

        if bar:
            # 创建Colorbar并嵌入到图中
            cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.02, pad=0.000000001, orientation='horizontal')
            cbar.set_label('freq(MHz)')
    
    x, y = pos[centerNode]
    diamond = plt.Polygon(((x, y + 2.3), (x + 2.3, y), (x, y - 2.3), (x - 2.3, y)), edgecolor='red', fill=None)
    plt.gca().add_patch(diamond)

    # 调整主图和 colorbar 之间的布局
    plt.axis('off')
    plt.savefig(name + '.pdf', dpi=300)
    plt.close()

def scatter_err(labelList, errList, name):
    plt.scatter([range(len(labelList))], errList, color='blue', alpha=0.5, s=100)
    plt.axhline(y=0.01, color='red', linestyle='--')
    # plt.semilogy()
    plt.savefig(name + '.pdf', dpi=300)
    plt.close()