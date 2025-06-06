import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.optimize import minimize
import random
from copy import deepcopy

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class LineSegment:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

def on_segment(p, q, r):
    """Check if point q lies on line segment pr"""
    if (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
        q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y)):
        return True
    return False

def orientation(p, q, r, s):
    """
    Find the orientation of the ordered triplet (p, q, r).
    """
    
    v_left = np.array((q.x - r.x, q.y - r.y))
    v_right = np.array((p.x - r.x, p.y - r.y))
    v_vert = np.array((s.x - r.x, s.y - r.y))
    
    val1 = np.cross(v_vert, v_left) * np.cross(v_vert, v_right)
    
    v_left = np.array((r.x - p.x, r.y - p.y))
    v_right = np.array((s.x - p.x, s.y - p.y))
    v_vert = np.array((q.x - p.x, q.y - p.y))
    
    val2 = np.cross(v_vert, v_left) * np.cross(v_vert, v_right)
    
    if val1 < 0 and val2 < 0:
        return -1
    elif val1 == 0 and not(val2 == 0):
        return 0
    elif val2 == 0 and not(val1 == 0):
        return 1
    elif val2 == 0 and val1 == 0:
        return 2
    else:
        return 3

def do_intersect(s1, s2):
    """Check if line segment 's1' and 's2' intersect."""
    p1, q1 = s1.p1, s1.p2
    p2, q2 = s2.p1, s2.p2

    # Find the four orientations needed for general and special cases
    o = orientation(p1, q1, p2, q2)

    # General case
    if o < 0:
        return True

    # Special Cases
    # s1 and s2 are collinear and s2.p1 lies on segment s1
    if o == 0 and \
        (on_segment(p2, p1, q2) or 
        on_segment(p2, q1, q2)):
        return True
    if o == 1 and \
        (on_segment(p1, p2, q1) or
        on_segment(p1, q2, q1)):
        return True
    if o == 2:
        if (on_segment(p1, p2, q1)) or (on_segment(p1, q2, q1)) or \
            (on_segment(p2, p1, q2)) or (on_segment(p2, q1, q2)):
            return True
        else:
            return False
        
    if o == 3:
        return False
    
# 定义目标函数
def objective(coords, edges, nodes):
    coords = coords.reshape(-1, 2)
    total_distance_variance = 0
    distances = []
    
    for edge in edges:
        p1 = coords[nodes.index(edge[0])]
        p2 = coords[nodes.index(edge[1])]
        dist = np.linalg.norm(p1 - p2)
        distances.append(dist)

    distances = np.array(distances) / np.min(distances)
        
    mean_distance = np.mean(distances)
    total_distance_variance = np.sum((distances - mean_distance) ** 2)
    
    # 检查边是否交叉
    for i in range(len(edges) - 1):
        for j in range(i + 1, len(edges)):
            if len(set(edges[i]).intersection(set(edges[j]))) == 0:
                s1 = LineSegment(Point(coords[nodes.index(edges[i][0])][0], coords[nodes.index(edges[i][0])][1]),
                                Point(coords[nodes.index(edges[i][1])][0], coords[nodes.index(edges[i][1])][1]))
                s2 = LineSegment(Point(coords[nodes.index(edges[j][0])][0], coords[nodes.index(edges[j][0])][1]),
                                Point(coords[nodes.index(edges[j][1])][0], coords[nodes.index(edges[j][1])][1]))
                if do_intersect(s1, s2):
                    total_distance_variance += 1000  # 惩罚交叉的边

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if not((nodes[i], nodes[j]) in edges) and not((nodes[j], nodes[i]) in edges):
                p1 = coords[i]
                p2 = coords[j]
                total_distance_variance += 1 / np.linalg.norm(p1 - p2)

    return total_distance_variance

# 定义生成随机平面图的函数
def generate_random_planar_graph(G, num_edges):
    for _ in range(num_edges):
        while 1:
            tempG = deepcopy(G)
            while 1:
                nodeIndeces = np.random.choice(len(G.nodes), 2)
                if not(nodeIndeces[0] == nodeIndeces[1]) and not((list(tempG.nodes)[nodeIndeces[0]], list(tempG.nodes)[nodeIndeces[1]]) in tempG.edges):
                    break
            tempG.add_edge(list(tempG.nodes)[nodeIndeces[0]], list(tempG.nodes)[nodeIndeces[1]])
            if nx.check_planarity(tempG)[0]:
                G = tempG
                break
    print(len(G.edges))
    return G

if __name__ == '__main__':

    # 初始化顶点位置
    G = nx.grid_2d_graph(3, 3)
    initial_pos = nx.planar_layout(G)

    # 初始化随机平面图
    num_edges = 9
    G = generate_random_planar_graph(G, num_edges)
    initial_pos = nx.planar_layout(G)

    # 将顶点坐标转换为一维数组
    initial_coords = np.array(list(initial_pos.values())).flatten()

    # 定义边
    edges = list(G.edges())

    # 绘制图形
    plt.figure(figsize=(8, 8))
    nx.draw(G, initial_pos, with_labels=True, node_color='lightblue', node_size=700, font_size=10)

    # 输出各个顶点的坐标
    for node, coord in initial_pos.items():
        print(f"顶点 {node} 的坐标是: {coord}")

    plt.show()

    # 使用模拟退火算法优化顶点位置
    result = minimize(objective, initial_coords, method='Powell', args=(list(G.edges), list(G.nodes)), options={'maxiter': 10000, 'disp': True})
    optimized_coords = result.x.reshape(-1, 2)

    # 创建新的位置字典
    optimized_pos = {list(G.nodes)[i]: optimized_coords[i] for i in range(len(optimized_coords))}

    distances = []
    for edge in edges:
        p1 = optimized_coords[list(G.nodes).index(edge[0])]
        p2 = optimized_coords[list(G.nodes).index(edge[1])]
        dist = np.linalg.norm(p1 - p2)
        distances.append(dist)
    print(distances)

    # 绘制优化后的图形
    plt.figure(figsize=(8, 8))
    nx.draw(G, optimized_pos, with_labels=True, node_color='lightblue', node_size=700, font_size=10)

    # 输出各个顶点的坐标
    for node, coord in optimized_pos.items():
        print(f"顶点 {node} 的坐标是: {coord}")

    plt.show()