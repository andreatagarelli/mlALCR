import csv
import numpy as np
from scipy.sparse import dok_matrix
import os
import glob
import scipy.stats as ss

__author__ = 'DIMES, University of Calabria, 2017'


# Field separator character in the input graph file. Values on each line of the file are separated by this character
SEPARATOR = ";"
# (Over)size of the node set 
DIM_MAX = 50000000
# (Over)size of the layer set 
LAYER_MAX = 1000
# Maximum number of iterations
MAX_ITERS = 500
# Convergence threshold
CONV_THRESHOLD = 1e-08
 


def main(directory, alpha):
    """
    It processes  each ".ncol" stored into folder "directory":

        Baseline 1, generating:
            input file name + "_BS1_LURK.txt"
            input file name + "_BS1_ACTIVE.txt"

        Baseline 2, generating:
            input file name + "_BS2_VAR.txt"


    :param directory: folder containing ".ncol" files  
    :return: Null but it will generate 3 files for each ".ncol", one for each baseline
    """
    graphs = glob.glob(directory + "*.ncol")
    header = True
    fout = dict()
    foutAgg = dict()
    for in_file in graphs:
        LRs = list()
        LRsAgg = list()
        fname = in_file.split('/')[len(in_file.split('/')) - 1].split('.')[0]
        out = directory + fname
        if not os.path.exists(out):
            os.makedirs(out)
        out = directory + fname + '/BS/'
        if not os.path.exists(out):
            os.makedirs(out)

        P, mapping, size, layers = parser_file(in_file, header)
        Ig, Og, Ie, Oe, Il, Ol = compute_vector(P, size)



        # baseline1
        print("=" * 100)
        print("Computing LRa1 on", in_file)
        R_lurker = compute_baseline1(P, Ig, Og, Ie, Oe, Il, Ol, alpha)  # , R_active

        # lurk score
        out_file = out + in_file.split('/')[len(in_file.split('/')) - 1] + "_LRa1_LURK.txt"
        print_result(R_lurker, out_file, layers, mapping, ToOrder=False)
        LRs.append(out_file)


        #  baseline2
        print("=" * 100)
        print("Computing LRa2 on", in_file)
        R_var = compute_baseline2(P, Ig, Og, Ie, Oe, Il, Ol)
        out_file = out + in_file.split('/')[len(in_file.split('/')) - 1] + "_LRa2_VAR.txt"
        print_result(R_var, out_file, layers, mapping, ToOrder=False)
        LRs.append(out_file)

        # output
        fout[fname] = LRs

    print("=" * 100)
    return fout



def parser_file(file_in, header=False, separator=SEPARATOR):
    """
    It parses the input file containing data in the format "source target layer".

    :param file_in: .ncol input file
    :param need_mapping: true if IDs in the input file are not consecutives
    :param header: true if the first row of the input file contains a header (one row to jump)
    :return: dict of dok matrices, where each matrix is the transpose of the adj matrix of a single layer
    """
    mapping = {}   # dict with mapping of the IDs 
    last_id = 0   
    with open(file_in) as in_file:
        reader = csv.reader(in_file, delimiter=separator)

        iter_reader = iter(reader)
        if header:
            next(iter_reader)
        F = {}     # dict of adj matrices
        layers = {}  # mapping layer names
        dim = 0    # no. of nonzero entries
        l = 0      # no. of layers
        for row in iter_reader:
            if row:
                source, target = map(int, row[:2])
                layer = row[2:3][0]

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
                    weight = float(row[3])
                    F[layers[layer]][source, target] = weight
                else:
                    F[layers[layer]][source, target] = 1
        dim += 1

        P = {}
        for i in F:
            P[i] = F[i].tocsr()[:dim, :dim]
        size = (len(P), dim)

    return P, mapping, size, layers




def compute_vector(P, size):
    """
    It computes global and local measures of indegree and outdegree

    :param P: dict of dok matrices, where each matrix is the transpose of the adj matrix of a single layer
    :param size: (no. of layers x no. of nodes)
    :return:
        Ig: array of global indegrees: for each node, it stores the sum of the node's indegree over all layers - array of size equal to the no. of nodes
        Og: array of global outdegrees: for each node, it stores the sum of the node's outdegree over all layers - array of size equal to the no. of nodes
        Ie: bidimensional array of global-one-leave-out indegrees: for each node and layer, it stores the sum of the node's indegree over all layers except for that layer
        Oe: bidimensional array of global-one-leave-out outdegrees: for each node and layer, it stores the sum of the node's outdegree over all layers except for that layer
        Il: bidimensional array of local indegrees: for each node and layer, it stores the node's indegree in that layer
        Ol: bidimensional array of local outdegrees: for each node and layer, it stores the node's outdegree in that layer
    """


    Il = np.zeros(size)
    Ol = np.zeros(size)
    Ie = np.zeros(size)
    Oe = np.zeros(size)
    for i in P:        # layer per layer
        Il[i] = P[i].sum(axis=1).A.ravel()
        Ol[i] = P[i].sum(axis=0).A.ravel()

    Ig = Il.sum(axis=0)
    Og = Ol.sum(axis=0)

    for r in range(size[1]):  # node by node
        Ie[:, r] = Ig[r] - Il[:, r] + 1
        Oe[:, r] = Og[r] - Ol[:, r] + 1

    Ig += 1
    Og += 1
    Il += 1
    Ol += 1

    return Ig, Og, Ie, Oe, Il, Ol




def compute_baseline1(P, Ig, Og, Ie, Oe, Il, Ol, alpha):
    """
    It executes baseline1 method

    :param P: dict of dok matrix, where each matrix is the transpose of the adj matrix of a single layer
    :param Ig: array of global indegrees: for each node, it stores the sum of the node's indegree over all layers - array of size equal to the no. of nodes
    :param Og: array of global outdegrees: for each node, it stores the sum of the node's outdegree over all layers - array of size equal to the no. of nodes
    :param Ie: bidimensional array of global-one-leave-out indegrees: for each node and layer, it stores the sum of the node's indegree over all layers except for that layer
    :param Oe: bidimensional array of global-one-leave-out outdegrees: for each node and layer, it stores the sum of the node's outdegree over all layers except for that layer
    :param Il: bidimensional array of local indegrees: for each node and layer, it stores the node's indegree in that layer
    :param Ol: bidimensional array of local outdegrees: for each node and layer, it stores the node's outdegree in that layer
    :param alpha
    :return: R_lurk: array to store the nodes' scores as lurker

    """
    n = len(Ig)
    r = np.ones((len(P), n), dtype=float) / n

    # how many nodes in each layer
    Ni = np.zeros(len(P))  # numbers of nodes in each layer
    d = Il + Ol - 2  # degree

    # run LR on each layer in P
    for i in P:
        r[i] = compute_LRin_out(P[i], Il[i], Ol[i], alpha)
        Ni[i] = len(np.nonzero(d[i])[0])

    # from score to rank
    rank = np.zeros((len(P), n), dtype=float)

    for i in P:
        rank[i] = ss.rankdata(r[i], method='dense')


    # compute lurking score
    R_lurk = np.zeros((len(P), n), dtype=float)
    temp = np.zeros((len(P), n), dtype=float)
    for i in P:
        for j in P:
            if i != j:
                temp[i] += rank[j] / Ni[j]
        R_lurk[i] = (Ni[i] - rank[i] + 1) / Ni[i] + temp[i]
        R_lurk[i] = R_lurk[i] / R_lurk[i].sum()



    return R_lurk




def compute_baseline2(P, Ig, Og, Ie, Oe, Il, Ol, alpha):
    """
    It executes baseline2 method

    :param P: dict of dok matrix, where each matrix is the transpose of the adj matrix of a single layer
    :param Ig: array of global indegrees: for each node, it stores the sum of the node's indegree over all layers - array of size equal to the no. of nodes
    :param Og: array of global outdegrees: for each node, it stores the sum of the node's outdegree over all layers - array of size equal to the no. of nodes
    :param Ie: bidimensional array of global-one-leave-out indegrees: for each node and layer, it stores the sum of the node's indegree over all layers except for that layer
    :param Oe: bidimensional array of global-one-leave-out outdegrees: for each node and layer, it stores the sum of the node's outdegree over all layers except for that layer
    :param Il: bidimensional array of local indegrees: for each node and layer, it stores the node's indegree in that layer
    :param Ol: bidimensional array of local outdegrees: for each node and layer, it stores the node's outdegree in that layer
    :param alpha
    :return: res: array of resulting scores of nodes
    """

    n = len(Ig)
    r = np.ones((len(P), n), dtype=float) / n

    # run LR on each layer in P
    Ni = np.zeros(len(P))  # numbers of nodes in each layer

    d = Il + Ol - 2  # degree

    for i in P:
        r[i] = compute_LRin_out(P[i], Il[i], Ol[i], alpha)
        Ni[i] = len(np.nonzero(d[i])[0])

    # from score to rank
    mod_rank = np.zeros((len(P), n), dtype=float)

    for i in P:
        mod_rank[i] = (ss.rankdata(r[i], method='dense') - (Ni[i] / 2)) / Ni[i]  # descending

    # compute variance for each node

    res = np.zeros(n)

    for i in range(n):  # compute for each node
        res[i] = np.var(mod_rank[:, i])


    return res



def compute_LRin_out(P, I, O, alpha):
    """

    :param P: dict of dok matrices, where each matrix is the transpose of the adj matrix of a single layer
    :param I: bidimensional array of local indegrees: for each node and layer, it stores the node's indegree in that layer
    :param O: bidimensional array of local outdegrees: for each node and layer, it stores the node's outdegree in that layer
    :return: r: array of resulting scores of nodes
    """
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


def compute_variance(ranks, average):
    """
    It computes variance of the ranks in input

    :param ranks: array of nodes' ranks 
    :param average: pre-computed average used in the variance computation
    :return: variance
    """
    variance = 0
    for rank in ranks:
        variance += (average - rank) ** 2
    return variance / len(ranks)



def print_result(res, out_file_path, layers, mapping, ToOrder=True):
    """

    :param res: result to be printed out
    :param out_file_path: output file path
    :param layers: dict of layer, id layer, orginal label of the layer present in the input file
    :param mapping: dict: node mapping original id - consecutive id
    :param ToOrder: boolean: if true then results are ordered by score
    :return:
    """

    if len(res.shape) == 1:
        if ToOrder:
            flatten_result = []
            for n in mapping:
                flatten_result.append((n, res[mapping[n]]))

            flatten_result.sort(key=lambda x: x[1], reverse=True)  # ordino in funzione dello score

            with open(out_file_path, "w") as out_file:
                print("node", "score", file=out_file, sep=SEPARATOR)
                for i in flatten_result:
                    print(i[0], i[1], file=out_file, sep=SEPARATOR)  # i[3]
        else:
            with open(out_file_path, "w") as out_file:
                print("node", "score", file=out_file, sep=SEPARATOR)
                for n in mapping:
                    print(n, res[mapping[n]], file=out_file, sep=SEPARATOR)





    else:

        if ToOrder:
            flatten_result = []
            for l in layers:
                for n in mapping:
                    flatten_result.append((n, l, res[layers[l], mapping[n]]))

            flatten_result.sort(key=lambda x: x[2], reverse=True)

            with open(out_file_path, "w") as out_file:
                print("node", "layer", "score", file=out_file, sep=SEPARATOR)
                for i in flatten_result:
                    print(i[0], i[1], i[2], file=out_file, sep=SEPARATOR)
        else:
            with open(out_file_path, "w") as out_file:
                print("node", "layer", "score", file=out_file, sep=SEPARATOR)
                for l in layers:
                    for n in mapping:
                        print(n, l, res[layers[l], mapping[n]], file=out_file, sep=SEPARATOR)







if __name__ == "__main__":
    main()