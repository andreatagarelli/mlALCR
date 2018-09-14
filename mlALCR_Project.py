from functions import mlALCR,lurkerRank, mlLR, kendallTau, fagin, baseline
import argparse
import os



__author__ = 'DIMES, University of Calabria, 2017'


def parse_args():
    parser = argparse.ArgumentParser(description="Run mlALCR, mlLR, LR, and baselines LRa1 and LRa2.")

    parser.add_argument('-m', '--method', default='None', choices=['mlALCR', 'mlLR', 'LR', 'baselines'],
                        help='select the method to run among mlALCR, mlLR, LR, and baselines.')

    parser.add_argument('-aa','--alphas', type=pair, default=[0.5, 0.5, 0.85, 0.85],
                        help='list of alpha1 and alpha2 values for mlALCR (i.e., alpha1,alpha2)')

    parser.add_argument('-a','--alpha', type=float, default=0.85,
                        help='alpha value for mlLR, LR, or the baselines')

    parser.add_argument('-e', '--eval', choices=['kendall', 'fagin'], default='',
                        help='select the evaluation measure to run')

    parser.add_argument('-k', type=int, default=0,
                        help="top-k users to be considered in fagin's intersection metric (i.e., 10, 100, 1000, ...).")

    parser.add_argument('--f1', '-file1', dest='file1', default=None, help='file1 to be compared with the evaluation measure selected.')

    parser.add_argument('--f2', '-file2', dest='file2', default=None, help='file2 to be compared with the evaluation measure selected.')

    args = parser.parse_args()

    if args.eval != '':
        if args.file1 is None and args.file2 is None:
            parser.error("--eval requires --file1 and --file2.")
        if args.file1 is None:
            parser.error("--eval requires --file1.")
        if args.file2 is None:
            parser.error("--eval requires --file2.")

    if args.eval == 'fagin' and args.k == 0:
        parser.error("Fagin requires k.")

    return args

def pair(arg):
    return [float(x) for x in arg.split(',')]

def main(args):

    dir = './data/'

    if not os.path.exists(dir):
        os.makedirs(dir)

    if args.method == 'mlALCR':
        fout_mlALCR, fout_mlALCR_Agg = mlALCR.main(dir, args.alphas)
        print('mlALCR result files:\n', fout_mlALCR)
        print('mlALCR_aggregated result files:\n', fout_mlALCR_Agg)
    elif args.method == 'mlLR':
        foutMLLR = mlLR.main(dir, args.alpha)
        print('mlLR result files:\n', foutMLLR)
    elif args.method == 'LR':
        foutLR = lurkerRank.main(dir, args.alpha)
        print('LR result files:\n', foutLR)
    elif args.method == 'baselines':
        foutBS = baseline.main(dir, args.alpha)
        print('Baselines result files:\n', foutBS)


    if args.eval != '':

        if args.eval == 'kendall':
            corr, p_value = kendallTau.computeKendall(args.file1, args.file2)
            print('Kendall Tau B:\nCorrelation = %s \np-value = %s' % (corr, p_value))
        if args.eval == 'fagin':
            fagin_value = fagin.computefagin(args.file1, args.file2, args.k)
            print('Fagin-%s = %s' % (args.k,fagin_value ))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)

