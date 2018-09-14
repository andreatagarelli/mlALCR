import csv
import numpy as np
from scipy.sparse import dok_matrix
import os
import os.path as osp
import glob

__author__ = 'DIMES, University of Calabria, 2017'


# Field separator character in the input graph file. Values on each line of the file are separated by this character
SEPARATOR = ";"
# (Over)size of the node set 
DIM_MAX = 2000000
# (Over)size of the layer set 
LAYER_MAX = 1000
# Maximum number of iterations 
MAX_ITERS = 150
# Convergence threshold
CONV_THRESHOLD = 1e-08


def main(dir, alpha):
    """

    :param dir: directory containing the input files with ".ncol" extension and edgelist file format
    :return: dict of files
    """
    files = glob.glob(dir + "*.ncol")
    header = True
    need_mapping = True
    fout = dict()
    for in_file in files:
        fname = in_file.split('/')[len(in_file.split('/')) - 1].split('.')[0]
        out = dir + fname + '/mlLR/'
        if not os.path.exists(out):
                os.makedirs(out)
        print("Computing mlLR on", in_file)
        P, mapping, size = parser_file(in_file, need_mapping, header)

        out_file = out + in_file.split('/')[len(in_file.split('/')) - 1] + "_mlLR.txt"
        Il, Ol = compute_vector(P, size)
        res = compute_mlLR(P, Il, Ol, alpha)
        print_result(res, mapping, out_file)
        fout[fname] = out_file
    print("=" * 100)
    return fout


def parser_file(file_in, need_mapping, header=False):
    """
    It parses the input file containing data in the format "source target layer".

    :param file_in: .ncol input file
    :param need_mapping: boolean: if IDs are not consecutive, it is necessary to map the IDs present in the input file
    :param header: true if the input file contains a header (one row to jump)
    :return: dict of dok matrices, where each matrix is the transpose of the adj matrix of a single layer
    """
    mapping = {}   # dict for the mappings of IDs 
    last_id = 0   
    with open(file_in) as in_file:
        reader = csv.reader(in_file, delimiter=SEPARATOR)

        iter_reader = iter(reader)
        if header:
            next(iter_reader)
        F = {}
        layers = {}
        dim = 0    # no. of nonzero entries
        l = 0      # no. of layers
        for row in iter_reader:
            if row:
                source, target = map(int, row[:2])
                layer = row[2:3][0]
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
                if layer not in layers:
                    layers[layer] = l
                    F[layers[layer]] = dok_matrix((DIM_MAX, DIM_MAX))
                    l += 1

                if len(row) > 3:
                    print(row)
                    weight = float(row[3])
                    F[layers[layer]][source, target] = weight
                else:
                    F[layers[layer]][source, target] = 1


        dim += 1
        P = {}
        for i in F:
            P[i] = F[i].tocsr()[:dim, :dim]
        size = (len(P), len(mapping))

    return P, mapping, size


def compute_vector(P, size):
    """
    It computes global and local measures of indegree and outdegree

    :param P: dict of dok matrices, where each matrix is the transpose of the adj matrix of a single layer
    :param size: (no. of layers x no. of nodes)
    :return:
        Il: bidimensional array of local indegrees: for each node and layer, it stores the node's indegree in that layer
        Ol: bidimensional array of local outdegrees: for each node and layer, it stores the node's outdegree in that layer
    """
    Il = np.zeros(size)
    Ol = np.zeros(size)

    for i in P:        # layer by layer
        Il[i] = P[i].sum(axis=1).A.ravel() # each column
        Ol[i] = P[i].sum(axis=0).A.ravel() # each row

    Il += 1
    Ol += 1

    return Il, Ol


def compute_mlLR(P, Il, Ol, alpha):
    """
    It executes the Multigraph LurkerRank method 

    :param P: dict of dok matrices, where each matrix is the transpose of the adj matrix of a single layer
    :param Il: bidimensional array of local indegrees: for each node and layer, it stores the node's indegree in that layer
    :param Ol: bidimensional array of local outdegrees: for each node and layer, it stores the node's outdegree in that layer
    :return: result array
    """
    n = len(Il[0])          # number of nodes
    r = np.ones(n) / n      # LR scores initialized to 1/n
    ratio = Ol / Il         # Outdegree/Indegree
    const = (1 - alpha) / n # p(V)
    I_norm = np.zeros(n)
    w = np.ones(len(P)) / len(P)     # Wt weights

    for l in P:
        temp = P[l].T.sign().dot(Il[l])
        temp[temp == 0] = 1
        temp = w[l] * Il[l] / temp
        I_norm += temp

    for _ in range(MAX_ITERS):
        lr_out = np.zeros(n)
        lr_in = np.zeros(n)

        r_pre = r

        for l in P:

            lr_in += w[l] * P[l].dot(r_pre * ratio[l]) / Ol[l]
            lr_out += I_norm * P[l].T.dot(r_pre / ratio[l])

        r = alpha * lr_in * (1 + lr_out) + const
        r = r / r.sum()

        if np.allclose(r, r_pre, atol=CONV_THRESHOLD, rtol=0):
            break
    return r


def print_result(res, mapping, out_file_path, ToOrder=True):
    """

    :param res: result to be printed out
    :param out_file_path: output file path
    :param mapping: dict: node mapping original id - consecutive id
    :param ToOrder: boolean: if true then results are ordered by score
    :return: Null

    """
    if ToOrder:
        flatten_result = []
        for n in mapping:
            flatten_result.append((n, res[mapping[n]]))

        flatten_result.sort(key=lambda x: x[0], reverse=False)

        with open(out_file_path, "w") as out_file:
            print("node;score", file=out_file, sep=SEPARATOR)
            for i in flatten_result:
                print(i[0], i[1], file=out_file, sep=SEPARATOR)
    else:
        with open(out_file_path, "w") as out_file:
            print("node;score", file=out_file, sep=SEPARATOR)
            for n in mapping:
                print(mapping[n], res[mapping[n]], file=out_file, sep=SEPARATOR)





if __name__ == "__main__":
    main()