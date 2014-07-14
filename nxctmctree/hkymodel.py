"""
This is a utility module for the example HKY model.

This module is pattered after npctmctree/hkymodel.py which is for
dense numpy data arrays, whereas the nxctmctree/hkymodel.py uses
sparse dicts and sparse networkx.DiGraph data types.

"""
from __future__ import division, print_function, absolute_import

import math
from itertools import permutations

import networkx as nx


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
    state_to_rate = Q.out_degree(weight='weight')
    expected_rate = sum(nt_distn[s] * state_to_rate[s] for s in nts)
    for sa in Q:
        for sb in Q[sa]:
            Q[sa][sb]['weight'] /= expected_rate
    return Q, nt_distn


def pack_params(edge_rates, nt_probs, kappa):
    params = []
    params.extend(edge_rates)
    params.extend(nt_probs)
    params.append(kappa)
    log_params = [math.log(p) for p in params]
    return log_params


def unpack_params(edges, log_params):
    params = [math.exp(p) for p in log_params]
    nedges = len(edges)
    edge_rates = params[:nedges]
    nt_probs = params[nedges:nedges+4]
    nt_probs_sum = sum(nt_probs)
    sqrt_penalty = math.log(nt_probs_sum)
    penalty = sqrt_penalty * sqrt_penalty
    nt_probs = [p / nt_probs_sum for p in nt_probs]
    kappa = params[-1]
    Q, nt_distn = create_rate_matrix(nt_probs, kappa)
    edge_to_rate = dict(zip(edges, edge_rates))
    return edge_to_rate, Q, nt_distn, kappa, penalty

