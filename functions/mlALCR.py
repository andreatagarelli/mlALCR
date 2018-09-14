import csv
import numpy as np
from scipy.sparse import dok_matrix
import scipy.stats as ss
import os
import glob
import networkx as nx
from functions import LayerSplitter
import pandas as pd

__author__ = 'DIMES, University of Calabria, 2017'


# Field separator character in the input graph file. Values on each line of the file are separated by this character
SEPARATOR = ";"
# (Over)size of the node set 
DIM_MAX = 2000000
# (Over)size of the layer set 
LAYER_MAX = 1000
# Maximum number of iterations
MAX_ITERS = 50000
# Convergence threshold
CONV_THRESHOLD = 1e-05
 

# (alpha1 , alpha2) values
#alphas = [0.5, 0.5, 0.85, 0.85]

START_COUNTING = 0


def main(directory, alphas):
    """

    On each input file stored into "directory", it computes mlALCR and aggregates the result (MIN, MAX and Median) for these combinations:

        weighted and unweighted layers

        different values of alpha1 and alpha2 as specified in the vector "alphas", where odd positions correspond to values of alpha1 and the others to values of alpha2



    :param directory: folder containing one or more input files (edgelist ".ncol")
    :return: two dict of files, id input file - path and filename of output files
    """
    graphs = glob.glob(directory + "*.ncol")
    header = True
    fout = dict()
    foutAgg = dict()
    for in_file in graphs:
        LRs = list()
        LRsAgg = list()
        print("=" * 100)
        print("Computing mlALCR on ", in_file)
        fname = in_file.split('/')[len(in_file.split('/')) - 1].split('.')[0]
        out = directory + fname
        if not os.path.exists(out):
            os.makedirs(out)
        out = directory + fname + '/mlALCR/'
        if not os.path.exists(out):
            os.makedirs(out)
        stats = out + 'stats/'
        if not os.path.exists(stats):
            os.makedirs(stats)
        outagg = out + 'aggr/'
        if not os.path.exists(outagg):
            os.makedirs(outagg)
        outagg += fname

        P, mapping, size, layers = parser_file(in_file, header)
        Ig, Og, Ie, Oe, Il, Ol, missing_nodes = compute_vector(P, size)

        # define weights
        not_wi = np.ones(LAYER_MAX)
        wi = compute_weights(in_file, layers, directory)

        for w in range(0, 2, 1):
            # define weights

            for i in range(0, len(alphas), 2):


                ALPHA1 = alphas[i]
                ALPHA2 = alphas[i + 1]

                pedice = ''

                pesi = not_wi
                if bool(w):
                    pedice += '_W'
                    pesi = wi

                pedice = 'a1_' + str(ALPHA1) + '_a2_' + str(ALPHA2) + pedice

                # compute ML_LR
                R_lurker, R_active, lurker_var_perc_dict, lurker_var_indexs_dict, active_var_perc_dict, active_var_indexs_dict \
                    = compute_ML_ALCR(P, Ig, Og, Ie, Oe, Il, Ol, missing_nodes, ALPHA1, ALPHA2, pesi, bool(w))

                # lurking score
                out_file = out + in_file.split('/')[len(in_file.split('/')) - 1] + "_mlALCR_LURK_" + pedice + ".txt"
                print_result(R_lurker, out_file, layers, mapping, missing_nodes, ToOrder=False)
                LRs.append(out_file)
                print_agg_result(R_lurker, mapping, outagg + '_mlALCR_LURK_' + pedice)
                LRsAgg.append(outagg + '_mlALCR_LURK_' + pedice + '_min.txt')
                LRsAgg.append(outagg + '_mlALCR_LURK_' + pedice + '_med.txt')
                LRsAgg.append(outagg + '_mlALCR_LURK_' + pedice + '_max.txt')

                # active score
                out_file = out + in_file.split('/')[len(in_file.split('/')) - 1] + "_mlALCR_ACTIVE_" + pedice + ".txt"
                print_result(R_active, out_file, layers, mapping, missing_nodes, ToOrder=False)
                LRs.append(out_file)
                print_agg_result(R_active, mapping, outagg + '_mlALCR_ACTIVE_' + pedice)
                LRsAgg.append(outagg + '_mlALCR_ACTIVE_' + pedice + '_min.txt')
                LRsAgg.append(outagg + '_mlALCR_ACTIVE_' + pedice + '_med.txt')
                LRsAgg.append(outagg + '_mlALCR_ACTIVE_' + pedice + '_max.txt')

        # output
        fout[fname] = LRs
        foutAgg[fname] = LRsAgg
    print("=" * 100)
    return fout, foutAgg


def compute_weights(in_file, layers, directory):
    """

    It computes the weight for each layer defined as
        number of edges in layer i / number of edges in the multilayer graph

    :param in_file: input file
    :param layers: dict of layers
    :return: array of layer weights 
    """
    N_edges = {}
    # split each layer
    subdir = directory + 'tmp/'
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    splitted_files = LayerSplitter.main(in_file, subdir, score=False)
    #for fname, fnames in splitted_files.items():
    for k, v in splitted_files.items():
        # compute layer number of edges
        D = nx.DiGraph()
        G = nx.read_edgelist(v, delimiter=SEPARATOR)
        D.add_edges_from(G.edges_iter())
        N_edges[k] = G.number_of_edges()
        del D
        del G

    # compute weights
    N_edges_tot = 0
    for k, v in N_edges.items():
        N_edges_tot += v

    weights = np.ones(LAYER_MAX)

    for k, v in layers.items():
        weights[v] = N_edges[k] / N_edges_tot

    return weights


def parser_file(file_in, header=False, separator=SEPARATOR):
    """
    It parses the input file containing data in the format "source target layer".

    :param file_in: .ncol input file
    :param header: true if the first row of the input file contains a header (one row to jump)
    :param separator: values separator
    :return: dict of dok matrices, where each matrix is the transpose of the adj matrix of a single layer
    """
    mapping = {}   # dict with mapping of the IDs
    last_id = 0  
    with open(file_in) as in_file:
        reader = csv.reader(in_file, delimiter=separator)

        iter_reader = iter(reader)
        if header:
            next(iter_reader)
        F = {}
        layers = {}
        dim = 0
        l = 0
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
        # conversion in CSR matrix representation
        dim += 1

        P = {}
        for i in F:
            P[i] = F[i].tocsr()[:dim, :dim]
        size = (len(P), dim)

        print("Parsing completed.")
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
    missing_nodes = np.ones(size)


    for i in P:        # layer by layer
        Il[i] = P[i].sum(axis=1).A.ravel()
        Ol[i] = P[i].sum(axis=0).A.ravel()
        missing_nodes[i] = np.sign(Il[i] + Ol[i])

    Ig = Il.sum(axis=0)
    Og = Ol.sum(axis=0)

    for r in range(size[1]):
        Ie[:, r] = Ig[r] - Il[:, r] + 1
        Oe[:, r] = Og[r] - Ol[:, r] + 1

    Ig += 1
    Og += 1
    Il += 1
    Ol += 1

    print("Indegree, Outdegree computed.")

    return Ig, Og, Ie, Oe, Il, Ol, missing_nodes



def compute_ML_ALCR(P, Ig, Og, Ie, Oe, Il, Ol, missing_nodes, a1, a2, wi, pesi=False):
    """
	It executes the mlALCR method

    :param P: dict of dok matrix, where each matrix is the transpose of the adj matrix of a single layer
    :param Ig: array of global indegrees: for each node, it stores the sum of the node's indegree over all layers - array of size equal to the no. of nodes
    :param Og: array of global outdegrees: for each node, it stores the sum of the node's outdegree over all layers - array of size equal to the no. of nodes
    :param Ie: bidimensional array of global-one-leave-out indegrees: for each node and layer, it stores the sum of the node's indegree over all layers except for that layer
    :param Oe: bidimensional array of global-one-leave-out outdegrees: for each node and layer, it stores the sum of the node's outdegree over all layers except for that layer
    :param Il: bidimensional array of local indegrees: for each node and layer, it stores the node's indegree in that layer
    :param Ol: bidimensional array of local outdegrees: for each node and layer, it stores the node's outdegree in that layer
    :param a1: alpha1
    :param a2: alpha2
    :param wi: array of layer weights 
    :param pesi: boolean: if true then compute the weighted version
    :return: R_lurk, R_active: arrays to store the nodes' scores as lurker and contributor, respectively

    """
    w_str = ' not weighted.'
    if pesi:
        w_str = ' weighted.'

    # variation measurement dicts
    lurker_var_perc_dict = dict()
    lurker_var_indexs_dict = dict()
    active_var_perc_dict = dict()
    active_var_indexs_dict = dict()


    # HITS based
    # NB: P = A^T
    N = len(Ig)  # numbers of nodes
    R_lurker = np.ones((len(P), N), dtype=float) / N
    R_active = np.ones((len(P), N), dtype=float) / N
    for _ in range(MAX_ITERS):
        # copy previous solutions
        R_lurker_pre = R_lurker.copy()
        R_active_pre = R_active.copy()
        for i in P:
            # P[i].dot --> in-neighboors --> followees
            # P[i].T.dot --> out-neighboors --> followers
            # HITS like

            # compute in layer
            lurker_in = missing_nodes[i] * wi[i] * P[i].dot(R_active_pre[i] / Oe[i]) / Ol[i]
            contributor_in = missing_nodes[i] * wi[i] * P[i].T.dot(R_lurker_pre[i] / Ie[i]) / Il[i]

            # initialize to zero
            lurker_out = np.zeros(N)
            contributor_out = np.zeros(N)

            # compute out layer
            for j in P:
                if i != j:
                    lurker_out += missing_nodes[j] * wi[j] * P[j].dot(R_active_pre[j] / Ol[i]) / (Ol[j])
                    contributor_out += missing_nodes[j] * wi[j] * P[j].T.dot(R_lurker_pre[j] / Il[i]) / (Il[j])

            # add 2 contributes
            R_lurker[i] = a1 * lurker_in + (1 - a1) * contributor_out
            R_active[i] = a2 * contributor_in + (1 - a2) * lurker_out

            # reset missing nodes
            R_lurker[i] = R_lurker[i] * missing_nodes[i]
            R_active[i] = R_active[i] * missing_nodes[i]


        # normalization over flattened solution
        flat_R_lurker_sum = R_lurker.flatten().sum()
        flat_R_active_sum = R_active.flatten().sum()

        for i in P:
            R_lurker[i] = R_lurker[i] / flat_R_lurker_sum
            R_active[i] = R_active[i] / flat_R_active_sum


        # evaluate error
        R_lurker_err = np.absolute(R_lurker - R_lurker_pre).sum()
        R_active_err = np.absolute(R_active - R_active_pre).sum()



        if (R_lurker_err < CONV_THRESHOLD) & (R_active_err < CONV_THRESHOLD):
            print("mlALCR: Convergence achieved in: ", _, " iterations, with alpha1 = ", a1, " and alpha2 = ", a2, w_str)
            return R_lurker, R_active, lurker_var_perc_dict, lurker_var_indexs_dict, active_var_perc_dict, active_var_indexs_dict
    print("mlALCR: Convergence not achieved. Number of iterations ", _,". alpha1 = ", a1, " and alpha2 = ", a2, w_str)
    return R_lurker, R_active, lurker_var_perc_dict, lurker_var_indexs_dict, active_var_perc_dict, active_var_indexs_dict


def write_rank_variations_to_file(res, variation_percentage, indexs, out_dir, pedice, layers, mapping):
    if variation_percentage:
        out_file = out_dir + '_Perc_rank_var_' + pedice + '.txt'

        d = pd.DataFrame()
        d = d.from_dict(variation_percentage, orient='index')
        d = d[(d.T != 0).any()]
        if not d.empty:
            d.to_csv(out_file, sep=' ', index=True, index_label=False)
        del d


def rbo(l1, l2, p=0.98):
    """
        It computea Ranked Biased Overlap (RBO) score.
        l1 -- ranked list
        l2 -- ranked list
    """
    if not l1:
        l1 = []
    if not l2:
        l2 = []

    sl, ll = sorted([(len(l1), l1), (len(l2), l2)])
    s, S = sl
    l, L = ll
    if s == 0:
        return 0

    # Calculate the overlaps at ranks 1 through l
    # (the longer of the two lists)
    ss = set([])  # contains elements from the smaller list till depth i
    ls = set([])  # contains elements from the longer list till depth i
    x_d = {0: 0}
    sum1 = 0.0
    for i in range(l):
        x = L[i]
        y = S[i] if i < s else None
        d = i + 1

        # if two elements are identical then
        # we don't need to add to either of the set
        if x == y:
            x_d[d] = x_d[d - 1] + 1.0
        # else add items to respective list
        # and calculate overlap
        else:
            ls.add(x)
            if y != None:
                ss.add(y)
            x_d[d] = x_d[d - 1] + (1.0 if x in ss else 0.0) + (1.0 if y in ls else 0.0)

        #calculate average overlap
        sum1 += x_d[d] / d * pow(p, d)

    sum2 = 0.0
    for i in range(l-s):
        d = s + i + 1
        sum2 += x_d[d] *(d - s) / (d * s) * pow(p, d)

    sum3 = ((x_d[l] - x_d[s]) / l + x_d[s] / s) * pow(p, l)

    # Equation 32
    rbo_ext = (1 - p) / p * (sum1 + sum2) + sum3
    return rbo_ext


def measure_variations(ranks, ranks_prev):
    """
    :returns variation_percentage
             intersection: a list of indexes of the nodes that changed position in the last iteration
    """
    intersection = list()
    for i in range(len(ranks)):
        if ranks[i] != ranks_prev[i]:
            intersection.append(i)

    variation_percentage = rbo(ranks, ranks_prev)

    return variation_percentage, intersection



def print_result(res, out_file_path, layers, mapping, missing_nodes, ToOrder=True):
    """

    :param res: result to be printed out
    :param out_file_path: output file path
    :param layers: dict of layer, id layer, orginal label of the layer present in the input file
    :param mapping: dict: node mapping original id - consecutive id
    :param ToOrder: boolean: if true then results are ordered by score
    :return: Null

    """
    if ToOrder:
        flatten_result = []
        for l in layers:
            for n in mapping:
                if missing_nodes[layers[l]][mapping[n]]:
                    flatten_result.append((n, l, res[layers[l], mapping[n]]))
                else:
                    pass

        flatten_result.sort(key=lambda x: x[2], reverse=True)

        with open(out_file_path, "w") as out_file:
            print("node", "layer", "score", file=out_file, sep=SEPARATOR)
            for i in flatten_result:
                print(i[0], i[1], i[2], file=out_file, sep=SEPARATOR)  # i[3]
    else:
        with open(out_file_path, "w") as out_file:
            print("node", "layer", "score", file=out_file, sep=SEPARATOR)
            for l in layers:
                for n in mapping:
                    if missing_nodes[layers[l]][mapping[n]]:
                        print(n, l, res[layers[l], mapping[n]], file=out_file, sep=SEPARATOR)
                    else:
                        pass





def print_agg_result(res, mapping, out_file_path):
    """

    :param res: result to be printed out
    :param mapping: dict: node mapping original id - consecutive id
    :param out_file_path: output file path
    :return: Null

    """

    flatten_result_max = []
    flatten_result_min = []
    flatten_result_med = []
    res_max = np.max(res, axis=0)
    res_min = np.min(res, axis=0)
    res_med = np.median(res, axis=0)

    for n in mapping:
        flatten_result_max.append((n, res_max[mapping[n]]))
        flatten_result_min.append((n, res_min[mapping[n]]))
        flatten_result_med.append((n, res_med[mapping[n]]))

    flatten_result_max.sort(key=lambda x: x[0], reverse=False)
    flatten_result_min.sort(key=lambda x: x[0], reverse=False)
    flatten_result_med.sort(key=lambda x: x[0], reverse=False)

    with open(out_file_path + "_max.txt", "w") as out_file:
        print("node;score", file=out_file, sep=SEPARATOR)
        for i in flatten_result_max:
            print(i[0], i[1], file=out_file, sep=SEPARATOR)

    with open(out_file_path + "_min.txt", "w") as out_file:
        print("node;score", file=out_file, sep=SEPARATOR)
        for i in flatten_result_min:
            print(i[0], i[1], file=out_file, sep=SEPARATOR)

    with open(out_file_path + "_med.txt", "w") as out_file:
        print("node;score", file=out_file, sep=SEPARATOR)
        for i in flatten_result_med:
            print(i[0], i[1], file=out_file, sep=SEPARATOR)




if __name__ == "__main__":
    main()