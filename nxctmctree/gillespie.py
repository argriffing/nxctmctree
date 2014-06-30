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

from .trajectory import Event, LightTrajectory


#FIXME copypasted from nxmctree.sampling
def dict_random_choice(d):
    total = sum(d.values())
    x = uniform(0, total)
    for i, w in d.items():
        x -= w
        if x < 0:
            return i


#FIXME copypasted from nxblink.util
def get_node_to_tm(T, root, edge_to_blen):
    """
    Use branch lengths to compute the distance from each node to the root.

    Parameters
    ----------
    T : networkx DiGraph
        the tree
    root : hashable
        the root of the tree
    edge_to_blen : dict
        branch length associated with each directed edge

    Returns
    -------
    node_to_tm : dict
        map from node to distance from the root

    """
    node_to_tm = {root : 0}
    for edge in nx.bfs_edges(T, root):
        va, vb = edge
        node_to_tm[vb] = node_to_tm[va] + edge_to_blen[edge]
    return node_to_tm


def get_gillespie_trajectory(T, root, root_prior_distn,
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
    # Initialize the trajectory with an arbitrary name and no history or events.
    track = LightTrajectory(name='gillespie', history={}, events={})

    # Compute map from nodes to times with respect to the root.
    node_to_tm = get_node_to_tm(T, root, edge_to_blen)

    # Sample the state at the root.
    track.history[root] = dict_random_choice(root_prior_distn)

    # Sample events along edges.
    for edge in nx.bfs_edges(T, root):

        # Unpack edge-specific info.
        rate = edge_to_rate[edge]
        state_to_rate = edge_to_state_to_rate[edge]
        state_to_distn = edge_to_state_to_distn[edge]
        na, nb = edge
        tma = node_to_tm[na]
        tmb = node_to_tm[nb]

        # Initialize the process at the upstream endpoint of the edge.
        state = track.history[na]
        tm = tma

        # Initialize a list of events along the edge.
        track.events[edge] = []

        # Sample transitions along the edge conditionally only on the
        # current state and without regard for downstream states.
        while True:
            tm += expovariate(rate * state_to_rate[state])
            if tm > tmb:
                break
            sb = dict_random_choice(state_to_distn[state])
            ev = Event(track=track, tm=tm, sa=state, sb=sb)
            track.events[edge].append(ev)
            state = sb
        track.history[nb] = state

    # Return the track.
    return track


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

