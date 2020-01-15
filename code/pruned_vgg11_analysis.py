import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from scipy.sparse import csr_matrix

sys.path.append('../prototype')
from blelloch_sim import blelloch, linear, Identity
from utils import load_jcbT_chain
from symbolic import symbolic_csrgemm


class Node(object):
    """ Tag the jacobian matrices for building the dependency graph. """

    def __init__(self, ID, jcbT):
        self.left_parent = None
        self.right_parent = None
        self.child = []
        self.id = ID
        self.jcbT = jcbT
        self.flop = None
        # Determine shape.
        if isinstance(self.jcbT, csr_matrix):
            self.shape = self.jcbT.shape
        else:
            assert isinstance(self.jcbT, list)
            self.shape = self.jcbT[0].shape
        # Determine sparsity.
        if isinstance(self.jcbT, csr_matrix):
            self.sparsity = 1 - (self.jcbT.nnz /
                                 (self.jcbT.shape[0] * self.jcbT.shape[1]))
        else:
            assert isinstance(self.jcbT, list)
            self.sparsity = np.mean(
                [1 - m.nnz / (m.shape[0] * m.shape[1]) for m in self.jcbT])

    def __repr__(self):
        return '{} {} & {} {} -> {} [{}, {}] -> {}'.format(
            self.left_parent.id if self.left_parent is not None else None,
            self.left_parent.shape if self.left_parent is not None else None,
            self.right_parent.id if self.right_parent is not None else None,
            self.right_parent.shape if self.right_parent is not None else None,
            self.id,
            self.shape,
            self.sparsity,
            [child.id for child in self.child],
        )


class IdIssuer(object):

    def __init__(self):
        self.counter = 0

    def issue(self):
        new_id = self.counter
        self.counter += 1
        return new_id


def save_node_map(node_map, path):
    for ID, node in node_map.items():
        node.jcbT = None
    pickle.dump(node_map, open(path, 'wb'))


def load_node_map(path):
    return pickle.load(open(path, 'rb'))


def flipgemm(b, a, node_map, id_issuer, levels):
    if isinstance(a, Identity):
        return b
    if isinstance(b, Identity):
        return a

    a_node, b_node = a, b
    a, b = a_node.jcbT, b_node.jcbT
    assert a_node.id in node_map and b_node.id in node_map

    measure_csrgemm = symbolic_csrgemm

    c = None
    flop = None
    if isinstance(a, list) and isinstance(b, list):
        c = []
        flop = []
        for i in range(len(a)):
            f, p = measure_csrgemm(a[i], b[i])
            c.append(p)
            flop.append(f)
        flop = np.mean(flop)
    elif isinstance(a, csr_matrix) and isinstance(b, csr_matrix):
        flop, c = measure_csrgemm(a, b)
    else:
        c = []
        flop = []
        if isinstance(b, list):
            for b_per_sample in b:
                f, p = measure_csrgemm(a, b_per_sample)
                c.append(p)
                flop.append(f)
        else:
            for a_per_sample in a:
                f, p = measure_csrgemm(a_per_sample, b)
                c.append(p)
                flop.append(f)
        flop = np.mean(flop)

    c_node = Node(id_issuer.issue(), c)
    c_node.left_parent = a_node
    c_node.right_parent = b_node
    a_node.child.append(c_node)
    b_node.child.append(c_node)
    c_node.flop = flop
    node_map[c_node.id] = c_node
    levels.add(c_node.id)
    return c_node

class Levels(object):

    def __init__(self):
        self._levels = []
        self.freeze = True

    def add(self, node):
        if self.freeze:
            return
        self._levels[-1].append(node)

    def level_start(self):
        self.freeze = False
        self._levels.append([])

    def level_end(self):
        self.freeze = True

    def get_levels(self):
        return self._levels

def bppsa():
    jcbTs = load_jcbT_chain('./jcbTs_prune/retrain_99_jcb_list/')
    id_issuer = IdIssuer()
    jcbTs = [Node(id_issuer.issue(), jcbT) for jcbT in jcbTs]
    node_map = {node.id: node for node in jcbTs}

    levels = Levels()

    def prelevel_callback():
        levels.level_start()

    def postlevel_callback():
        levels.level_end()

    blelloch(
        list(reversed(jcbTs)),
        lambda a, b: flipgemm(a, b, node_map, id_issuer, levels),
        Identity(),
        L_m=2,
        prelevel_callback=prelevel_callback,
        postlevel_callback=postlevel_callback,
    )
    return node_map, levels.get_levels()

def baseline():
    jcbTs = load_jcbT_chain('./jcbTs/iter_4999/')
    mnk = []
    flops = []
    for j in jcbTs[:-1]:
        if isinstance(j, csr_matrix):
            # Convolution
            mnk.append(j.shape[0] * j.shape[1])
            flops.append(j.nnz * 2)
        else:
            mnk.append(j[0].shape[0] * j[0].shape[1])
            # ReLU/MaxPool
            if j[0].shape[0] == j[0].shape[1]: #ReLU
                # Init to 0 and pass certain gradients back.
                flops.append(j[0].shape[0] + j[0].nnz)
            else: # MaxPool
                assert j[0].shape[0] > j[0].shape[1]
                flops.append(j[0].shape[0] + j[0].shape[1])
    return flops, mnk


def get_critical_path(node_map, levels):
    levels_set = set([])
    critical_path = set([])
    for l in levels:
        max_flop = 0
        max_flop_node_id = None
        for node_id in l:
            if node_map[node_id].flop > max_flop:
                max_flop = node_map[node_id].flop
                max_flop_node_id = node_id
            levels_set.add(node_id)
        critical_path.add(max_flop_node_id)

    for node_id, node in node_map.items():
        if node_id in levels_set or (node.left_parent is None and node.right_parent is None):
            continue
        critical_path.add(node_id) # The linear scan in the middle.

    return critical_path


def plot(node_map, baseline_flops, baseline_mnk, critical_path):
    x_axis_mm = []
    x_axis_mv = []
    x_axis_mm_c = []
    x_axis_mv_c = []
    y_axis_mm = []
    y_axis_mv = []
    y_axis_mm_c = []
    y_axis_mv_c = []
    id_list_mm = []
    id_list_mv = []
    id_list_mm_c = []
    id_list_mv_c = []

    for k in node_map:
        node = node_map[k]
        if node.left_parent is None and node.right_parent is None:
            continue
        size = (node.left_parent.shape[0] * node.left_parent.shape[1] *
                node.right_parent.shape[1])
        # matrix-vector multiplication
        if node.right_parent.shape[1] == 1:
            if node.id not in critical_path:
                x_axis_mv.append(size)
                y_axis_mv.append(node.flop)
                id_list_mv.append(node.id)
            else:
                x_axis_mv_c.append(size)
                y_axis_mv_c.append(node.flop)
                id_list_mv_c.append(node.id)
            # matrix-matrix multiplication
        else:
            if node.id not in critical_path:
                x_axis_mm.append(size)
                y_axis_mm.append(node.flop)
                id_list_mm.append(node.id)
            else:
                x_axis_mm_c.append(size)
                y_axis_mm_c.append(node.flop)
                id_list_mm_c.append(node.id)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('m x n x k')
    plt.ylabel('FLOP')

    plt.scatter(x_axis_mv, y_axis_mv, facecolors='none', edgecolors='#f48642',
                label='mv')
    plt.scatter(x_axis_mm, y_axis_mm, facecolors='none', edgecolors='#5A9BD5',
                label='mm')
    plt.scatter(x_axis_mv_c, y_axis_mv_c, color='#f48642', label='mv, critical')
    plt.scatter(x_axis_mm_c, y_axis_mm_c, color='#5A9BD5', label='mm, critical')

    x_axis_baseline = baseline_mnk
    y_axis_baseline = baseline_flops

    plt.scatter(x_axis_baseline, y_axis_baseline, facecolors='none',
                edgecolor='green', label='baseline (BP)')

    plt.legend()
    plt.savefig('./fig_13.png', bbox_inches='tight', pad_inches=0.0)


def main():
    node_map, levels = bppsa()
    critical_path = get_critical_path(node_map, levels)
    baseline_flops, baseline_mnk = baseline()
    plot(node_map, baseline_flops, baseline_mnk, critical_path)


if __name__ == '__main__':
    main()
