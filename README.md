# MULTILAYER ALTERNATE LURKER-CONTRIBUTOR RANKING (mlALCR)


![Example multilayer social network, which illustrates eighteen users located over three layer networks.](https://github.com/andreatagarelli/mlalcr/blob/master/ml_network.png "Multilayer Network")

**Copyright 2017-2018 Diego Perna, Andrea Tagarelli, DIMES Dept.,  University of Calabria, Italy**


**_mlALCR_** is a Python software for identifying and ranking users who alternately behave as contributors and as lurkers over multiple layers of a multilayer social network. 
  The software, in addition to an implementation of the **_mlALCR_** method, contains the implementation of two baselines and an extension of **[LurkerRank](https://github.com/andreatagarelli/lurkerrank)** to multilayer networks. The implemented methods were originally defined and described in the peer-reviewed scientific publication reported below.
___ 

**TERMS OF USAGE:**
The following paper should be cited in any research product whose findings are based on the code here distributed:


[Diego Perna, Roberto Interdonato, Andrea Tagarelli.<br>
Identifying Users With Alternate Behaviors of Lurking and Active Participation in Multilayer Social Networks.<br> 
IEEE Trans. Comput. Social Systems 5(1): 46-63 (2018).](https://ieeexplore.ieee.org/document/8101313/)

___

**Instructions**
 
The relative path of the working directory is assumed to be `./data/` which must contain the input multilayer network(s). Each multilayer network is stored into a text file with extension `.ncol` modeling the edge lists over the layers.  If the working directory contains more than one multilayer network, the input files of the networks will be processed sequentially, according to lexicographic order of the file names.

***
The format of the input `.ncol` files:

Each row in a  `.ncol` file contains a triple of values (integer, integer, string)  corresponding to the IDs of source-node, target-node, and layer, respectively, separated by `;`. Moreover, the `.ncol` file is required to have the first row corresponding to a header (e.g., `source;target;layer`). 


**Options**

The software runs one of the following ranking methods, by specifying the option `--method $method_name`:
1. `mlALCR`
2. `mlLR`
3. `LR` (LurkerRank)
4. `Baselines` (LRa1, LRa2)

The software also evaluates the correlation between two ranking solutions, by specifying the option `--eval $method_name`:

1. `Kendall` (Kendall Tau rank correlation coefficient)
2. `Fagin` (Fagin's intersection metric --- it requires to specify the number `k` of top users to be selected)


 
***
 
**Usage Examples**

1. Ranking methods:

 - Run **mlALCR** on all the input files in the working directory, with `alpha1=0.5` and `alpha2=0.95`:<br> 
`python3 mlALCR_Project.py --method mlALCR --alphas 0.5,0.95`

- Run LurkerRank on all the input files in the working directory, with `alpha=0.75`:<br>
`python3 mlALCR_Project.py --method LR --alpha 0.75`


2. Evaluation methods:

 - Compute the Kendall Tau rank correlation coefficient between `ranking1.txt` and `ranking2.txt`:<br> 
`python3 mlALCR_Project.py --eval kendall --f1 ranking1.txt -f2 ranking2.txt`

 - Compute the Fagin's intersection metric between `ranking1.txt` and `ranking2.txt`, with `k=100`:<br> 
`python3 mlALCR_Project.py --eval fagin -k 100 --f1 ranking1.txt -f2 ranking2.txt`

