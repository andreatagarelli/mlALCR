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



def main(list1, out_file, list2=[]):


    if not list2:
        # only one list of files
        res_corr = np.ones((len(list1), len(list1)))
        for x, y, in itertools.combinations(list1, 2):
            temp_corr, temp_pval = computeKendall(x, y)
            res_corr[list1.index(x), list1.index(y)] = temp_corr
            res_corr[list1.index(y), list1.index(x)] = temp_corr


        df = pd.DataFrame(res_corr)
        labels = {}
        for i in range(len(list1)):
            labels[i] = ntpath.basename(list1[i])
        #df.column = labels
        df.rename(columns=labels, inplace=True)
        df.to_csv(out_file, sep=';', index=False)

    else:
        # two lists to be combined
        res_corr = {}
        for x in list1:
            name1 = x.split('/')[len(x.split('/')) - 1]
            res_corr[name1] = {}
            for y in list2:
                name2 = y.split('/')[len(y.split('/')) - 1]
                temp_corr, temp_pval = computeKendall(x, y)
                res_corr[name1][name2] = temp_corr

        df = pd.DataFrame(res_corr).fillna(1)
        df.to_csv(out_file, sep=';')








def parser_file(file_in, header=True):
    """

    It parses the input file containing data in the format "NODE;LAYER;SCORE"

    :param file_in: input file
    :param header: true if the first row of the input file contains a header
    :return: ranked data by score

    """
    R = []
    with open(file_in) as in_file:
        reader = csv.reader(in_file, delimiter=SEPARATOR)

        iter_reader = iter(reader)
        if header:
            next(iter_reader)

        for row in iter_reader:
            if row:
                if len(row) > 2:
                    score = float(row[2])
                else:
                    score = float(row[1])

                R.append(score)

    return stats.rankdata(R)





def computeKendall(file1, file2):
    """
        It computes Kendall Tau B (Scipy.stats lib) between file1 and file2. (It handles ties.)

        :param file1: input file
        :param file2: input file
        :return: corr: kendall tau correlation
                 p_value: kendall tau p-value
        """
    R1 = parser_file(file1, True)
    R2 = parser_file(file2, True)

    try:
        corr, p_value = stats.kendalltau(R1, R2)
    except:
        corr = 0
        p_value = 0

    return corr, p_value

if __name__ == "__main__":
    main()

