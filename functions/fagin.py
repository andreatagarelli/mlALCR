import csv
from scipy import stats
import pandas as pd
import itertools
import ntpath
import numpy as np

__author__ = 'DIMES, University of Calabria, 2017'


# Field separator character in the input graph file. Values on each line of the file are separated by this character
SEPARATOR = ";"
# (Over)size of the node set 
DIM_MAX = 2000000
# no. of top-ranked elemtns to consider 
Fagin_Top = 1000


def main(list1, k, out_file, list2=[]):

    if not list2:
        # only one list of files
        res_corr = np.ones((len(list1), len(list1)))
        for x, y, in itertools.combinations(list1, 2):
            temp_corr = computefagin(x, y, k)
            res_corr[list1.index(x), list1.index(y)] = temp_corr
            res_corr[list1.index(y), list1.index(x)] = temp_corr


        df = pd.DataFrame(res_corr)
        labels = {}
        for i in range(len(list1)):
            labels[i] = ntpath.basename(list1[i])
        df.rename(columns=labels, inplace=True)
        df.to_csv(out_file, sep=';', index=False)

    else:
        # two lists to be combined
        # res_corr = np.ones((len(list1), len(list2)))
        res_corr = {}
        for x in list1:
            name1 = x.split('/')[len(x.split('/')) - 1]
            res_corr[name1] = {}
            for y in list2:
                name2 = y.split('/')[len(y.split('/')) - 1]
                temp_corr = computefagin(x, y, k)
                res_corr[name1][name2] = temp_corr

        df = pd.DataFrame(res_corr).fillna(1)
        df.to_csv(out_file, sep=';')




def parser_file(file_in, header=False):
    """

    It parses the input file containing data in the format "NODE,LAYER;SCORE" (Note the comma, the actual separator is ';')

    :param file_in: input file (it can handle single and multilayer input)
    :param header: true if the first row of the input file contains a header (one row to jump)
    :return: IDs sorted by scores

    """
    df = pd.read_csv(file_in, sep=SEPARATOR)
    try:
        df = df.sort_values(by=['score'], ascending=False)

    except Exception as e:

        print('cannot sort ', file_in)



    try:
        ids = df['node,layer'].values
    except:
        #print('WARNING: cannot select \"node,layer\" perform a replace operation if needed')
        ids = df['node'].values

    return ids

def computefagin(file1, file2, top):
    """
    It computes Fagin's metric between different result files comparing node IDs (IDs are pre-ordered by decreasing score)

    :param file1: input file
    :param file2: input file
    :param out_file: output file
    :param top: number of top nodes to compare
    """

    R1 = parser_file(file1, True)
    R2 = parser_file(file2, True)

    lmax = min(len(R1), len(R2))
    k = min(top, lmax)
    fag = 0

    for d in range(k):

        current_intersection = 0

        for i in range(d + 1):
            for j in range(d + 1):
                if R1[i] == R2[j]:
                    current_intersection += 1
        fag += current_intersection / (d + 1)

    fag = fag / k

    return fag

if __name__ == "__main__":
    main()