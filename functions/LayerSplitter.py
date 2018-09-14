import csv
import os

__author__ = 'DIMES, University of Calabria, 2017'


# Field separator character in the input graph file. Values on each line of the file are separated by this character
SEPARATOR = ";"


def main(infile, odir, score=False):
    """
    It splits a multilayer input file in L (number of layers) single-layer files

    :param infile: input file 
    :param odir: output directory
    :param score:  boolean: true if the input file is a result and not an input ".ncol" file
    :return:
    """

    #files = glob.glob(idir + "*.ncol")
    if not os.path.exists(odir):
            os.makedirs(odir)
    fout = dict()
    #for file in files:
    fname = infile.split('/')[len(infile.split('/')) - 1].split('.')[0]

    return layersplit(infile, odir, header=True, score=score)


def layersplit(file_in, folder, header=True, score=False):
    """
    It splits a multilayer input file in L (number of layers) single-layer files 

    :param file_in: input file 
    :param folder: output folder
    :param header: boolean: true if the first row in the input file contains a header
    :param score: boolean: true if the input file is a result and not an input ".ncol" file
    :return: dict of files - input files - output files
    """

    with open(file_in) as file:
        F = {}
        layers = {}
        l = 0
        reader = csv.reader(file, delimiter=SEPARATOR)
        iter_reader = iter(reader)
        if header:
            # skip the header row
            next(iter_reader)
        for row in iter_reader:
            if row:
                if score:
                    layer = row[1:2][0]  # 2:3 input   1:2 output
                else:
                    layer = row[2:3][0]  # 2:3 input   1:2 output
                if layer not in layers:
                    l += 1
                    layers[layer] = l
                    F[layers[layer]] = []
                F[layers[layer]].append(row)
    ls = dict()
    for layer in layers:
        out_file = folder + file_in.split('/')[len(file_in.split('/')) - 1] + "_" + layer + ".txt"
        ls[layer] = (out_file)
        L = F[layers[layer]][:]
        with open(out_file, "a") as out_file:
            if score:
                print("node;score", file=out_file, sep=";")
                for i, value in enumerate(L):
                    #if L[i][2] != '0.0':
                    print(L[i][0], L[i][2], file=out_file, sep=";")  #L[i][1],
                    #else:
                        #pass
            else:
                print("source;target", file=out_file, sep=";")
                for i, value in enumerate(L):
                    print(L[i][0], L[i][1], file=out_file, sep=";")
    return ls





if __name__ == "__main__":
    main()