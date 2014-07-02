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

from scipy.optimize import minimize

import nxctmctree
from nxctmctree.trajectory import get_node_to_tm, FullTrackSummary
from nxctmctree.gillespie import (
        get_gillespie_trajectory, gen_gillespie_trajectories,
        get_incomplete_gillespie_sample)


def expand_Q(Q):
    state_to_rate = Q.out_degree(weight='weight')
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
    state_to_rate = Q.out_degree(weight='weight')
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
    penalty = np.square(np.log(nt_probs.sum()))
    nt_probs = nt_probs / nt_probs.sum()
    kappa = params[-1]
    Q, nt_distn = create_rate_matrix(nt_probs, kappa)
    edge_to_rate = dict(zip(edges, edge_rates))
    return edge_to_rate, Q, nt_distn, kappa, penalty


def get_trajectory_log_likelihood(T, root,
        edge_to_Q, edge_to_rate, root_prior_distn, full_track_summary):
    """
    """
    root_ll = 0
    for root_state, count in full_track_summary.root_state_to_count.items():
        if count:
            p = root_prior_distn[root_state]
            root_ll += count * math.log(p)
    trans_ll = 0
    dwell_ll = 0
    for edge in nx.bfs_edges(T, root):
        edge_rate = edge_to_rate[edge]
        Q = edge_to_Q[edge]

        # transition contribution
        info = full_track_summary.edge_to_transition_to_count.get(edge, None)
        if info is None:
            raise Exception('found an edge with no observed transitions')
        for (sa, sb), count in info.items():
            if count:
                rate = edge_rate * Q[sa][sb]['weight']
                trans_ll += count * math.log(rate)

        # dwell time contribution
        info = full_track_summary.edge_to_state_to_time.get(edge, None)
        if info is None:
            raise Exception('found an edge with no observed dwell times')
        for state, duration in info.items():
            if duration:
                rate = edge_rate * Q.out_degree(state, weight='weight')
                dwell_ll -= rate * duration
    #print(root_ll, trans_ll, dwell_ll)
    log_likelihood = root_ll + trans_ll + dwell_ll
    return log_likelihood


def main():

    random.seed(23456)

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

    # Define edge-specific rate scaling factors.
    #edge_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    #edge_rates = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    #edge_rates = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    edge_rates = np.array([1, 2, 3, 4, 5])

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
    edge_to_Q = dict((e, Q) for e in edges)
    edge_to_state_to_rate = dict((e, state_to_rate) for e in edges)
    edge_to_state_to_distn = dict((e, state_to_distn) for e in edges)
    node_to_tm = get_node_to_tm(T, root, edge_to_blen)
    bfs_edges = list(nx.bfs_edges(T, root))

    print('state_to_rate:')
    print(state_to_rate)
    print('state_to_distn:')
    print(state_to_distn)
    print()

    # Get some gillespie samples.
    # Pick out the leaf states, and get a sample distribution over
    # leaf state patterns.
    full_track_summary = FullTrackSummary()
    pattern_to_count = defaultdict(int)
    nsamples_gillespie = 10000
    node_to_state_to_count = dict((v, defaultdict(int)) for v in T)
    for track in gen_gillespie_trajectories(T, root, root_prior_distn,
            edge_to_rate, edge_to_blen, edge_to_Q, nsamples_gillespie):
        full_track_summary.on_track(T, root, node_to_tm, bfs_edges, track)
        for v, state in track.history.items():
            node_to_state_to_count[v][state] += 1
        pattern = tuple(track.history[v] for v in leaves)
        pattern_to_count[pattern] += 1

    # Report the patterns.
    print('sampled patterns:')
    for pattern, count in sorted(pattern_to_count.items()):
        print(pattern, ':', count)
    print()

    # Report state counts at nodes.
    print('state counts at nodes:')
    for v in T:
        state_to_count = node_to_state_to_count[v]
        print('node:', v)
        for state in 'ACGT':
            print('  state:', state, 'count:', state_to_count[state])
    print()

    # Report some summary of the trajectories.
    print('full track summary:')
    print('root state counts:', full_track_summary.root_state_to_count)
    print()

    # Report parameter values used for sampling.
    print('parameter values used for sampling:')
    print('edge to rate:', edge_to_rate)
    print('nt distn:', nt_distn)
    print('kappa:', kappa)
    print()

    # TODO compute max likelihood estimates
    # using the actual gillespie sampled trajectories
    #
    # Define some initial guesses for the parameters.
    x0_edge_rates = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    x0_nt_probs = np.array([0.25, 0.25, 0.25, 0.25])
    x0_kappa = 3.0
    x0 = pack_params(edges, x0_edge_rates, x0_nt_probs, x0_kappa)

    def objective(log_params):
        # edges, T, root, full_track_summary are available within this function
        unpacked = unpack_params(edges, log_params)
        edge_to_rate, Q, nt_distn, kappa, penalty = unpacked
        edge_to_Q = dict((e, Q) for e in edges)
        root_prior_distn = nt_distn
        log_likelihood = get_trajectory_log_likelihood(T, root,
                edge_to_Q, edge_to_rate, root_prior_distn, full_track_summary)
        return -log_likelihood + penalty

    x_sim = pack_params(edges, edge_rates, nt_probs, kappa)
    print('objective function value using the parameters used for sampling:')
    print(objective(x_sim))
    print()

    result = minimize(
            objective, x0, method='L-BFGS-B',
            #options=dict(pgtol=1e-8),
            )

    print(result)
    log_params = result.x
    unpacked = unpack_params(edges, log_params)
    edge_to_rate, Q, nt_distn, kappa, penalty = unpacked
    print('max likelihood estimates from sampled trajectories:')
    print('edge to rate:', edge_to_rate)
    print('nt distn:', nt_distn)
    print('kappa:', kappa)
    print('penalty:', penalty)
    print()


    # TODO compute max likelihood estimates
    # using EM with conditionally sampled histories using Rao-Teh.



if __name__ == '__main__':
    main()

