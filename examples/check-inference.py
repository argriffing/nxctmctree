"""
This script checks inference of branch lengths and HKY parameter values.

No part of this script uses eigendecomposition or matrix exponentials,
or even numpy or scipy linear algebra functions.
The sampling of joint nucleotide states at leaves uses
unconditional Gillespie sampling, and the inference of branch lengths
and parameter values given these samples uses Monte Carlo
expectation-maximization with non-CTBN Rao-Teh sampling of substitution
trajectories.

"""
from __future__ import division, print_function, absolute_import

import math
import random

import networkx as nx

import nxctmctree
from nxctmctree.sampling import sample_histories

def main():
    pass

if __name__ == '__main__':
    main()

