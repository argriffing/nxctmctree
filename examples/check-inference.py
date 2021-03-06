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

import itertools
import math
import random
from collections import defaultdict
from itertools import permutations
from functools import partial

import numpy as np
import networkx as nx

from scipy.optimize import minimize

import nxctmctree
from nxctmctree import gillespie, raoteh, hkymodel
from nxctmctree.likelihood import get_trajectory_log_likelihood
from nxctmctree.trajectory import get_node_to_tm, FullTrackSummary


def objective(T, root, edges, full_track_summary, log_params):
    unpacked = hkymodel.unpack_params(edges, log_params)
    edge_to_rate, Q, nt_distn, kappa, penalty = unpacked
    edge_to_Q = dict((e, Q) for e in edges)
    root_prior_distn = nt_distn
    log_likelihood = get_trajectory_log_likelihood(T, root,
            edge_to_Q, edge_to_rate, root_prior_distn, full_track_summary)
    return -log_likelihood + penalty


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
    edge_rates = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    #edge_rates = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    #edge_rates = np.array([1, 2, 3, 4, 5])

    # Define HKY parameter values.
    nt_probs = np.array([0.1, 0.2, 0.3, 0.4])

    # Define kappa.
    kappa = 2.4

    # Initialize some more stuff before getting the gillespie samples.
    edge_to_rate = dict(zip(edges, edge_rates))
    edge_to_blen = dict((e, 1) for e in edges)
    Q, nt_distn = hkymodel.create_rate_matrix(nt_probs, kappa)
    root_prior_distn = nt_distn
    state_to_rate, state_to_distn = gillespie.expand_Q(Q)
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
    full_track_summary = FullTrackSummary(T, root, edge_to_blen)
    pattern_to_count = defaultdict(int)
    nsamples_gillespie = 10000
    node_to_state_to_count = dict((v, defaultdict(int)) for v in T)
    for track in gillespie.gen_trajectories(T, root, root_prior_distn,
            edge_to_rate, edge_to_blen, edge_to_Q, nsamples_gillespie):
        full_track_summary.on_track(track)
        for v, state in track.history.items():
            node_to_state_to_count[v][state] += 1
        pattern = tuple(track.history[v] for v in leaves)
        pattern_to_count[pattern] += 1

    # Count the number of patterns.
    npatterns = len(pattern_to_count)

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
    x0 = hkymodel.pack_params(x0_edge_rates, x0_nt_probs, x0_kappa)
    x0 = np.array(x0)

    x_sim = hkymodel.pack_params(edge_rates, nt_probs, kappa)
    x_sum = np.array(x_sim)
    print('objective function value using the parameters used for sampling:')
    print(objective(T, root, edges, full_track_summary, x_sim))
    print()

    f = partial(objective, T, root, edges, full_track_summary)
    result = minimize(f, x0, method='L-BFGS-B')

    print(result)
    log_params = result.x
    unpacked = hkymodel.unpack_params(edges, log_params)
    edge_to_rate, Q, nt_distn, kappa, penalty = unpacked
    print('max likelihood estimates from sampled trajectories:')
    print('edge to rate:', edge_to_rate)
    print('nt distn:', nt_distn)
    print('kappa:', kappa)
    print('penalty:', penalty)
    print()


    # TODO compute max likelihood estimates
    # using EM with conditionally sampled histories using Rao-Teh.

    # Initialize a blank track for each leaf pattern, for EM with Rao-Teh.
    #tracks = []
    #for i in range(npatterns):
        #track = raoteh.get_feasible_blank_trajectory(
                #T, root, root_prior_distn, edge_to_Q, node_to_tm)
        #tracks.append(track)

    # Get the leaf states.
    # This sampled data is in pattern_to_count.
    
    # and use Rao-Teh to sample a bunch of trajectories
    # given these leaf states.

    # Define initial parameter values for the expectation maximization.
    x0_edge_rates = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    x0_nt_probs = np.array([0.25, 0.25, 0.25, 0.25])
    x0_kappa = 3.0
    x0 = hkymodel.pack_params(x0_edge_rates, x0_nt_probs, x0_kappa)
    x0 = np.array(x0)
    packed = x0
    unpacked = hkymodel.unpack_params(edges, x0)
    edge_to_rate, Q, nt_distn, kappa, penalty = unpacked
    edge_to_Q = dict((e, Q) for e in edges)
    root_prior_distn = nt_distn

    # Do some burn-in samples for each pattern,
    # using the initial parameter values.
    # Do not store summaries of these sampled trajectories.
    nburn = 10
    pattern_to_track = {}
    pattern_to_data = {}
    set_of_all_states = set('ACGT')
    for idx, pattern in enumerate(pattern_to_count):
        #print('burning in the trajectory',
                #'for pattern', idx+1, 'of', npatterns, '...')
        
        # Create the data representation.
        leaf_to_state = dict(zip(leaves, pattern))
        node_to_data_fset = {}
        for node in T:
            if node in leaves:
                fset = {leaf_to_state[node]}
            else:
                fset = set_of_all_states
            node_to_data_fset[node] = fset

        # Save the data representation constructed for each pattern.
        pattern_to_data[pattern] = node_to_data_fset

        # Add the track.
        pattern_to_track[pattern] = None

    # Do some EM iterations.
    for em_iteration_index in itertools.count():
        print('starting EM iteration', em_iteration_index+1, '...')

        # Each EM iteration gets its own summary object.
        full_track_summary = FullTrackSummary(T, root, edge_to_blen)

        # Do a few Rao-Teh samples for each pattern within each EM iteration.
        for idx, (pattern, track) in enumerate(pattern_to_track.items()):
            #print('sampling Rao-Teh trajectories for pattern', idx+1, '...')
            count = pattern_to_count[pattern]
            node_to_data_fset = pattern_to_data[pattern]

            # Note that the track is actually updated in-place
            # even though the track object is yielded at each iteration.
            nburnin = 0 if idx else nburn
            for updated_track in raoteh.gen_raoteh_trajectories(
                    T, edge_to_Q, root, root_prior_distn, node_to_data_fset,
                    edge_to_blen, edge_to_rate, set_of_all_states,
                    initial_track=track, nburnin=nburn, nsamples=count):
                full_track_summary.on_track(updated_track)

        # This is the M step of EM.
        f = partial(objective, T, root, edges, full_track_summary)
        result = minimize(f, packed, method='L-BFGS-B')
        #print(result)
        packed = result.x
        unpacked = hkymodel.unpack_params(edges, packed)
        edge_to_rate, Q, nt_distn, kappa, penalty = unpacked
        print('max likelihood estimates from sampled trajectories:')
        print('penalized negative log likelihood:', result.fun)
        print('edge to rate:', edge_to_rate)
        print('nt distn:', nt_distn)
        print('kappa:', kappa)
        print('penalty:', penalty)
        print()


if __name__ == '__main__':
    main()

