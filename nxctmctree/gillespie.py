"""
Gillespie sampling.

This is unconditional forward sampling on a tree.

Full Gillespie sampling will sample an entire state trajectory on a tree.
In other words, it includes the state at all nodes of the tree
and it also includes the time and nature of each substitution event.

Incomplete Gillespie sampling stores only the sampled states at nodes.
Although the time and nature of each substitution event is generated
during the sampling procedure, this extra information is not recorded.

"""
from __future__ import division, print_function, absolute_import

import math
from random import expovariate, uniform

import networkx as nx


#FIXME copypasted from nxmctree.sampling
def dict_random_choice(d):
    total = sum(d.values())
    x = uniform(0, total)
    for i, w in d.items():
        x -= w
        if x < 0:
            return i


def get_incomplete_gillespie_sample(T, root, root_prior_distn,
        edge_to_rate, edge_to_blen,
        edge_to_state_to_rate, edge_to_state_to_distn):
    """
    Return map from node to sampled state.

    Parameters
    ----------
    T : networkx DiGraph
        rooted tree
    root : hashable
        root node of the rooted networkx tree T
    root_prior_distn : dict
        prior distribution over states at the root
    edge_to_rate : dict
        map from edge to rate scaling factor
    edge_to_blen : dict
        map from edge to branch length
    edge_to_state_to_rate : dict
        map from edge to a map from each state to its total exit rate
    edge_to_state_to_distn : dict
        map from edge to a map from each state to its transition distribution

    """
    # Sample the state at the root.
    node_to_state = {root : dict_random_choice(root_prior_distn)}

    # Sample states along edges, recording only the final state.
    for edge in nx.bfs_edges(T, root):

        # Unpack edge-specific info.
        rate = edge_to_rate[edge]
        blen = edge_to_blen[edge]
        state_to_rate = edge_to_state_to_rate[edge]
        state_to_distn = edge_to_state_to_distn[edge]

        # Initialize the process at the upstream endpoint of the edge.
        na, nb = edge
        state = node_to_state[na]
        tm = 0

        # Sample transitions along the edge conditionally only on the
        # current state and without regard for downstream states.
        while True:
            tm += expovariate(rate * state_to_rate[state])
            if tm > blen:
                break
            state = dict_random_choice(state_to_distn[state])
        node_to_state[nb] = state

    # Return the map from node to state.
    return node_to_state

