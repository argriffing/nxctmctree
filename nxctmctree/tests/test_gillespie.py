"""
Test Monte Carlo inference using unconditional Gillespie forward samples.

This test module has been converted from an example application.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict
from itertools import permutations
import math
import random

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose

from scipy.optimize import minimize

import nxctmctree
from nxctmctree.trajectory import get_node_to_tm, FullTrackSummary
from nxctmctree.gillespie import gen_gillespie_trajectories


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


def test_gillespie():

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
    node_to_tm = get_node_to_tm(T, root, edge_to_blen)
    bfs_edges = list(nx.bfs_edges(T, root))

    # Get some gillespie samples.
    # Pick out the leaf states, and get a sample distribution over
    # leaf state patterns.
    full_track_summary = FullTrackSummary(T, root, edge_to_blen)
    pattern_to_count = defaultdict(int)
    nsamples_gillespie = 10000
    for track in gen_gillespie_trajectories(T, root, root_prior_distn,
            edge_to_rate, edge_to_blen, edge_to_Q, nsamples_gillespie):
        full_track_summary.on_track(track)

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

    # Compute max likelihood estimates
    # using the actual Gillespie sampled trajectories.
    result = minimize(objective, x0, method='L-BFGS-B')
    log_params = result.x
    unpacked = unpack_params(edges, log_params)
    opt_edge_to_rate, opt_Q, opt_nt_distn, opt_kappa, penalty = unpacked

    # Check that the max likelihood estimates
    # are somewhat near the parameter values used for sampling.
    # Require relative tolerance of 1e-1 which is not too precise,
    # but because we want the test to run in a reasonable amount of time
    # we are not using so many samples so we cannot expect too much precision.
    rtol = 1e-1
    for edge in edges:
        assert_allclose(opt_edge_to_rate[edge], edge_to_rate[edge], rtol=rtol)
    nts = 'ACGT'
    for nt in nts:
        assert_allclose(opt_nt_distn[nt], nt_distn[nt], rtol=rtol)
    assert_allclose(opt_kappa, kappa, rtol=rtol)

