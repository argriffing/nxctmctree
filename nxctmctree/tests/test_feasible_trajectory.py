"""
Test the initialization of a feasible trajectory.

The tree shape is known, the edge rate and length scaling factors are known,
the equilibrium state at the root is known, and the edge-specific
instantaneous transition rates (before scaling) are known.

The initialization can be subtle if a large region of the tree
is forced to not have any state changes; this forcing can happen
if some edge-specific rate or length scaling factors are zero.

"""
from __future__ import division, print_function, absolute_import

from itertools import permutations

import networkx as nx
from numpy.testing import assert_equal

import nxctmctree.raoteh


def test_feasible_trajectory():
    T = nx.DiGraph()
    edge = ('a', 'b')
    T.add_edge(*edge)
    root = edge[0]
    edge_to_rate = {edge : 0}
    edge_to_blen = {edge : 1}
    n = 4
    all_states = set(range(n))
    Q = nx.DiGraph()
    for sa, sb in permutations(range(n), 2):
        Q.add_edge(sa, sb, weight=1)
    edge_to_Q = {edge : Q}
    distn = dict((i, 1/n) for i in range(n))
    node_to_data_fset = dict((v, all_states) for v in T)

    # Keep track of the (initial, final) state pairs.
    state_pairs = []
    for i in range(100):
        for track in nxctmctree.raoteh.gen_raoteh_trajectories(
                T, edge_to_Q, root, distn, node_to_data_fset,
                edge_to_blen, edge_to_rate, all_states,
                None, nburnin=0, nsamples=1):
            pair = tuple(track.history[v] for v in edge)
            state_pairs.append(pair)

    # Check the sampled histories.
    # No history should have initial and final states that are different.
    # All n of the states should be represented.
    assert_equal(set(state_pairs), set((s, s) for s in all_states))

