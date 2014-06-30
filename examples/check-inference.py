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

from collections import defaultdict
from itertools import permutations
import math
import random

import numpy as np
import networkx as nx

import nxctmctree
from nxctmctree.gillespie import (
        get_gillespie_trajectory, get_incomplete_gillespie_sample)


def expand_Q(Q):
    state_to_rate = Q.degree(weight='weight')
    state_to_distn = dict()
    for sa in Q:
        rate = state_to_rate[sa]
        distn = dict((sb, Q[sa][sb]['weight'] / rate) for sb in Q[sa])
        state_to_distn[sa] = distn
    return state_to_rate, state_to_distn


def create_rate_matrix(nt_probs, kappa):
    """
    Create an HKY rate matrix normalized to expected rate of 1.0.

    """
    nts = 'ACGT'
    nt_distn = dict(zip(nts, nt_probs))
    Q = nx.DiGraph()
    for sa, sb in permutations(nts, 2):
        rate = nt_distn[sb]
        if {sa, sb} in ({'A', 'G'}, {'C', 'T'}):
            rate *= kappa
        Q.add_edge(sa, sb, weight=rate)
    state_to_rate = Q.degree(weight='weight')
    expected_rate = sum(nt_distn[s] * state_to_rate[s] for s in nts)
    for sa in Q:
        for sb in Q[sa]:
            Q[sa][sb]['weight'] /= expected_rate
    return Q, nt_distn


def pack_params(edges, edge_rates, nt_probs, kappa):
    params = np.concatenate([edge_rates, nt_probs, [kappa]])
    log_params = np.log(params)
    return log_params


def unpack_params(edges, log_params):
    params = np.exp(log_params)
    nedges = len(edges)
    edge_rates = params[:nedges]
    nt_probs = params[nedges:nedges+4]
    penalty = np.square(np.log(np.sum(nt_probs)))
    nt_probs = nt_probs / nt_probs.sum()
    kappa = params[-1]
    Q, nt_distn = create_rate_matrix(nt_probs, kappa)
    return edge_rates, Q, nt_distn, penalty


def main():

    # Define an edge ordering.
    edges = [
            ('N0', 'N1'),
            ('N0', 'N2'),
            ('N0', 'N3'),
            ('N1', 'N4'),
            ('N1', 'N5'),
            ]

    # Define a rooted tree shape.
    T = nx.DiGraph()
    T.add_edges_from(edges)
    root = 'N0'
    leaves = ('N2', 'N3', 'N4', 'N5')

    edges = T.edges()

    # Define edge-specific rate scaling factors.
    edge_rates = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # Define HKY parameter values.
    nt_probs = np.array([0.1, 0.2, 0.3, 0.4])

    # Define kappa.
    kappa = 2.4

    # Initialize some more stuff before getting the gillespie samples.
    edge_to_rate = dict(zip(edges, edge_rates))
    edge_to_blen = dict((e, 1) for e in edges)
    Q, nt_distn = create_rate_matrix(nt_probs, kappa)
    root_prior_distn = nt_distn
    state_to_rate, state_to_distn = expand_Q(Q)
    edge_to_state_to_rate = dict((e, state_to_rate) for e in edges)
    edge_to_state_to_distn = dict((e, state_to_distn) for e in edges)

    # Get some gillespie samples.
    # Pick out the leaf states, and get a sample distribution over
    # leaf state patterns.
    pattern_to_count = defaultdict(int)
    nsamples_gillespie = 10000
    for i in range(nsamples_gillespie):
        track = get_gillespie_trajectory(T, root, root_prior_distn,
                edge_to_rate, edge_to_blen,
                edge_to_state_to_rate, edge_to_state_to_distn)
        #node_to_state = get_incomplete_gillespie_sample(
                #T, root, root_prior_distn,
                #edge_to_rate, edge_to_blen,
                #edge_to_state_to_rate, edge_to_state_to_distn)
        pattern = tuple(track.history[v] for v in leaves)
        pattern_to_count[pattern] += 1


    # Report the patterns.
    for pattern, count in sorted(pattern_to_count.items()):
        print(pattern, ':', count)

    # TODO compute max likelihood estimates
    # using the actual gillespie sampled trajectories

    # TODO compute max likelihood estimates
    # using EM with conditionally sampled histories using Rao-Teh.



if __name__ == '__main__':
    main()

