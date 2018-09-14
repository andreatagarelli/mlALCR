import csv
import numpy as np
from scipy.sparse import dok_matrix
import os
import os.path as osp
import glob
from functions import LayerSplitter


__author__ = 'DIMES, University of Calabria, 2015'


# Field separator character in the input graph file. Values on each line of the file are separated by this character
SEPARATOR = ";"
# (Over)size of the node set 
DIM_MAX = 2000000
# Maximum number of iterations 
MAX_ITERS = 150
# Convergence threshold
CONV_THRESHOLD = 1e-8


def parser_file(file_in, need_mapping, header=True):
    """
    Parses an input digraph file, in NCOL format <source target weight>. 
    The edge meaning is that "source" node is influenced by (e.g., follows, likes) "target" node.
    If "weight" is not specified, edge weights are assumed to be 1.
    
    :param file_in: the name of the file which the data are to be read from.
    :param need_mapping: a logical value indicating whether nodes are numbered using a non-progressive order.
    :param header: a logical value indicating whether the file contains the names of the variables as its first line.
    :return: the transpose of the adjacency matrix in CSR format.
    """
    mapping = {} # dict for the mapping of node ids
    last_id = 0  # largest id of node 

    with open(file_in) as in_file:
        reader = csv.reader(in_file, delimiter=SEPARATOR)

        iter_reader = iter(reader)
        if header:
            # Skip header
            next(iter_reader)

        dim = DIM_MAX
        P = dok_matrix((dim, dim))

        dim = 0

        for row in iter_reader:
            if row:
                try:
                    source, target = map(int, row[:2])
                except:
                    pass
                if need_mapping:
                    if source not in mapping:
                        mapping[source] = last_id
                        last_id += 1

                    if target not in mapping:
                        mapping[target] = last_id
                        last_id += 1

                    source = mapping[source]
                    target = mapping[target]
                dim = max(dim, source, target)

                if len(row) > 2:  # true if weights are available
                    weight = float(row[2])
                    P[source, target] = weight
                else:
                    P[source, target] = 1.

        # conversion in CSR matrix
        dim += 1
        P = P.tocsr()[:dim, :dim]
    return P, mapping


def graph_to_matrix(G, need_mapping):
    """
    It parses an input DiGraph object of networkx. 
    If "weight" is not specified, edge weights are assumed to be 1.

    :param G: the name of the DiGraph object.
    :param need_mapping: a logical value indicating whether nodes are numbered using a non-progressive order.
    :return: the transpose of the adjacency matrix in CSR format.
    """
    
    mapping = {} # dict for the mapping of node ids
    last_id = 0  # largest id of node

    dim = DIM_MAX
    P = dok_matrix((dim, dim))

    dim = 0
    for edge in G.edges(data=True):
            source,target,w = edge
            if need_mapping:
                if source not in mapping:
                    mapping[source] = last_id
                    last_id += 1

                if target not in mapping:
                    mapping[target] = last_id
                    last_id += 1

                source = mapping[source]
                target = mapping[target]
            dim = max(dim, source, target)
            if len(w.keys()) > 0:  # true if weights are available
                for k in w.keys():    # if the graph is weighted then the list should contain a single attribute (e.g., 'influence' or 'weight')
                    weight = float(w[k])
                    P[target, source] = weight
                    break
            else:
                P[target, source] = 1.
    # conversion in CSR matrix
    dim += 1
    P = P.tocsr()[:dim, :dim]
    return P, mapping


def compute_vector(P):
    """
    It computes the in-degree vector and the out-degree vector 

    :param P: the transpose of the adjacency matrix
    :return: I: the in-degree vector, O: the out-degree vector
    """
    I = P.sum(axis=1).A.ravel() + 1 #each column
    O = P.sum(axis=0).A.ravel() + 1 #each row

    return I, O


"""
All LurkerRank functions require in input: 
P: transpose of the adjacency matrix
I: in-degree vector
O: out-degree vector
alpha: damping factor
Each of the functions terminates after MAX_ITERS iterations or when the error is not greater than CONV_THRESHOLD, 
and returns a unit-norm vector storing the LurkerRank solution. 
"""

def compute_LRin(P, I, O, alpha):
    n = len(I)
    r = np.ones(n) / n
    ratio = O / I
    const = (1 - alpha) / n
    for _ in range(MAX_ITERS):
        r_pre = r
        s = P.dot(ratio * r_pre) / O
        r = alpha * s + const
        if np.allclose(r, r_pre, atol=CONV_THRESHOLD, rtol=0):
            break
    return r / r.sum()


def compute_LRin_out(P, I, O, alpha):
    n = len(I)
    r = np.ones(n) / n
    ratio = O / I
    const = (1 - alpha) / n

    sum_in = P.T.sign().dot(I)
    sum_in[sum_in == 0] = 1
    I_norm = (I / sum_in)
    for _ in range(MAX_ITERS):
        r_pre = r
        lr_in = P.dot(r_pre * ratio) / O
        lr_out = 1 + I_norm * P.T.dot(r_pre / ratio)

        r = alpha * lr_in * lr_out + const
        r = r / r.sum()
        if np.allclose(r, r_pre, atol=CONV_THRESHOLD, rtol=0):
            break
    return r


def compute_alpha_LRin(P, I, O, alpha):
    n = len(I)
    r = np.ones(n) / n
    ratio = O / I
    for _ in range(MAX_ITERS):
        r_pre = r
        s = P.dot(ratio * r_pre) / O
        r = alpha * s + 1
        if np.allclose(r, r_pre, atol=CONV_THRESHOLD, rtol=0):
            break
    return r / r.sum()


def compute_LRout(P, I, O, alpha):
    n = len(I)
    r = np.ones(n) / n
    const = (1 - alpha) / n
    sum_in = P.T.sign().dot(I)
    sum_in[sum_in == 0] = 1
    ratio = I / O
    I_norm = (I / sum_in)
    for _ in range(MAX_ITERS):
        r_pre = r
        r = alpha * I_norm * P.T.dot(r_pre * ratio) + const
        if np.allclose(r, r_pre, atol=CONV_THRESHOLD, rtol=0):
            break
    return r / r.sum()


def compute_alpha_LRout(P, I, O, alpha):
    n = len(I)
    r = np.ones(n) / n
    sum_in = P.T.sign().dot(I)
    sum_in[sum_in == 0] = 1
    ratio = I / O
    I_norm = (I / sum_in)
    for _ in range(MAX_ITERS):
        r_pre = r
        r = alpha * I_norm * P.T.dot(r_pre * ratio) + 1
        if np.allclose(r, r_pre, atol=CONV_THRESHOLD, rtol=0):
            break
    return r / r.sum()


def compute_alpha_LRin_out(P, I, O, alpha):
    n = len(I)
    r = np.ones(n) / n
    ratio = O / I

    sum_in = P.T.sign().dot(I)
    sum_in[sum_in == 0] = 1
    I_norm = (I / sum_in)
    for _ in range(MAX_ITERS):
        r_pre = r
        lr_in = P.dot(r_pre * ratio) / O
        lr_out = 1 + I_norm * P.T.dot(r_pre / ratio)

        r = alpha * lr_in * lr_out + 1
        r = r / r.sum()
        if np.allclose(r, r_pre, atol=CONV_THRESHOLD, rtol=0):
            break
    return r


def print_result(res, mapping, out_file_path):
    """
    :param res: result to be printed out
    :param mapping: IDs mapping
    :param out_file_path
    """

    flatten_result = []
    for n in mapping:
        flatten_result.append((n, res[mapping[n]]))

    flatten_result.sort(key=lambda x: x[0], reverse=False)  #ordino in funzione del nodo

    with open(out_file_path, "w") as out_file:
        print("node;score", file=out_file, sep=SEPARATOR)
        for i in flatten_result:
            print(i[0], i[1], file=out_file, sep=SEPARATOR)

ALL_RANKS = {#'LRin': compute_LRin,
             'LRinout': compute_LRin_out,
             #'LRout': compute_LRout,
             #'acLRin': compute_alpha_LRin,
             'acLRinout': compute_alpha_LRin_out,
             #'acLRout': compute_alpha_LRout
             }


def lr_dict(res, mapping):
    lr_dict = {}
    nodes = np.arange(0, len(res))
    nodes = np.lexsort((nodes, -res))
    res = res[nodes]

    flat_mapp = []
    if mapping:
        while mapping:
            flat_mapp.append(mapping.popitem())
        flat_mapp.sort(key=lambda t: t[1])
    for user_id, val in zip(nodes, res):
            if flat_mapp:
                user_id = flat_mapp[user_id][0]
            lr_dict[int(user_id)] = float(val)
    return lr_dict


"""
Invokes a particular LurkerRank function. 
:param func: the name of a LurkerRank function selected from the enumeration LR_METHODS
:param file_in: the name of the file which the data are to be read from.
:param file_out: the name of the file where the results are to be printed out.
:param need_mapping: a logical value indicating whether nodes are numbered using a non-progressive order.
:param return_dict: if True, the function returns a dict <id,score>, otherwise results are printed out to file_out.
:param input_graph: True if file_in corresponds to a DiGraph object of networkx.
"""
def computeLR(func, file_in, file_out, alpha,need_mapping=True, return_dict=False, input_graph=False):
    if input_graph==True:
        P, mapping = graph_to_matrix(file_in, need_mapping)
    else:
        P, mapping = parser_file(file_in, need_mapping, header=True)

    I, O = compute_vector(P)

    res = func(P, I, O, alpha)
    if return_dict==False:
        print_result(res, mapping, file_out)
    else:
        return lr_dict(res,mapping)



def main(dir, alpha):
    graphs = glob.glob(dir + "*.ncol")
    for in_file in graphs:

        fname = in_file.split('/')[len(in_file.split('/')) - 1].split('.')[0]

        subdirLR = dir + fname +'/LR/input/'
        outdir = dir + fname +'/LR/'
        if not os.path.exists(subdirLR):
            os.makedirs(subdirLR)

        # split layers
        foutLS = LayerSplitter.main(in_file, subdirLR,
                                    score=False)  # create a dir for LR and split the ncol layer by layer

        foutLR = dict()
        for k, v in foutLS.items():
            in_file = v
            current_file = os.path.abspath(os.path.dirname(in_file))

            LR = list()
            fname = in_file.split('/')[len(in_file.split('/')) - 1].split('.')[0]
            need_mapping = True
            for (name, func) in ALL_RANKS.items():
                out_file = outdir + in_file.split('/')[len(in_file.split('/')) - 1] + "_" + name + ".txt"
                print("Computing", name, "on", in_file)

                computeLR(func, in_file, out_file, alpha, need_mapping)
                LR.append(out_file)
            foutLR[k] = LR
        print("=" * 100)
        return foutLR



if __name__ == "__main__":
    main()