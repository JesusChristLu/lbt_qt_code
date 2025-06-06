from copy import deepcopy
import networkx as nx
import numpy as np
from networkx.algorithms.planarity import check_planarity
from networkx.algorithms.isomorphism import is_isomorphic
from networkx.algorithms.shortest_paths.generic import shortest_path
import matplotlib.pyplot as plt
from prune import Prune

import os #


def search_media_structure():
    mediaStructureN = []
    k_dict = {}
    path = 'F:\\vs experiment\\chipdesignv9\\chipdesign\\media structures ' + str(Prune.degLimit) + '\\' #
    path_list = os.listdir(path) #
    ms = [] #
    for file in path_list: #
        if int(file.split(' ')[1]) == 94 and int(file.split(' ')[0]) == 25: #
            with open(os.path.join(path, file), 'r') as fp: #
                data = fp.read() #
                data = data.split('\n')[:-1] #
                mediaStructure = np.zeros((len(data), len(data))) #
                for raw in range(len(data)): #
                    data_raw = data[raw].split(' ')[:-1] #
                    for column in range(len(data)): #
                        mediaStructure[raw, column] = int(data_raw[column]) #
                ms.append(mediaStructure) #

    test = True #
    for n in range(25, 26):
        if test: #
            id = 55
            mediaStructureN = ms #
        else: #
            print('finding', n, 'bit structures')
            simpliestStructure = nx.Graph()
            simpliestStructure.add_weighted_edges_from([(i, i + 1, 1) for i in range(n - 1)])
            k = calculate_k(nx.to_numpy_array(simpliestStructure))
            mediaStructureN = [nx.to_numpy_array(simpliestStructure)]
            id = 0
            store_media_structure(nx.to_numpy_array(simpliestStructure), n, k, id)
            k_dict[k] = calculate_path_length(simpliestStructure)
        while not(mediaStructureN == []):
            oldStructureN = mediaStructureN
            mediaStructureN = []
            for struct in oldStructureN:
                struct = nx.from_numpy_array(struct)
                print('searching more complicated structures on', len(struct.edges), 'edges.')
                for node1 in struct.nodes:
                    for node2 in struct.nodes:
                        print(node1, node2)
                        if not node1 == node2 and not (node1, node2) in struct.edges:
                            structure = deepcopy(struct)
                            structure.add_edge(node1, node2)
                            valid = True
                            for s in mediaStructureN + oldStructureN:
                                s = nx.from_numpy_array(s)
                                is_planar = check_planarity(s)[0]
                                if is_isomorphic(s, structure) or not is_planar:
                                    valid = False
                                    break
                            if valid:
                                if not(calculate_k(nx.to_numpy_array(structure)) in k_dict.keys()):
                                    k_dict[calculate_k(nx.to_numpy_array(structure))] = calculate_path_length(structure)
                                    mediaStructureN.append(nx.to_numpy_array(structure))
                                    id += 1
                                elif k_dict[calculate_k(nx.to_numpy_array(structure))] > calculate_path_length(structure):
                                    k = calculate_k(nx.to_numpy_array(structure))
                                    mediaStructureN.append(nx.to_numpy_array(structure))
                                    store_media_structure(mediaStructureN[-1], n, k, id)
                                    k_dict[k] = calculate_path_length(structure)
                                    id += 1
            print('we have try all the structures with', len(struct.edges) + 1, 'edges.')
def store_media_structure(mediaStructure, n, k, id):
    # plt.title(k)
    # nx.draw(nx.from_numpy_array(mediaStructure))
    # plt.show()
    file_name = 'F:\\vs experiment\\chipdesignv9\\chipdesign\\media structures ' + str(Prune.degLimit) + '\\' + \
    str(n) + ' ' + str(k) + ' ' + str(id) + '.txt'
    with open(file_name, 'w') as fp:
        for ii in range(len(mediaStructure)):
            for jj in range(len(mediaStructure[0])):
                fp.write(str(int(mediaStructure[ii, jj])) + ' ')
            fp.write('\n')

def calculate_k(mediaStructure):
    mediaStructure = nx.from_numpy_array(mediaStructure)
    node_in_mediaStructure = list(mediaStructure.nodes)
    k = 0
    for node in node_in_mediaStructure:
        while mediaStructure.degree(node) < Prune.degLimit:
            add_node = len(mediaStructure.nodes)
            mediaStructure.add_edge(node, add_node)
            if not check_planarity(mediaStructure)[0]:
                mediaStructure.remove_node(add_node)
                break
            else:
                k += 1
    return k

def calculate_path_length(mediaStructure):
    length = 0
    path_list = shortest_path(mediaStructure)
    for source in path_list:
        for target in path_list[source]:
            if length < len(path_list[source][target]):
                length = len(path_list[source][target])

    return length - 1
search_media_structure()