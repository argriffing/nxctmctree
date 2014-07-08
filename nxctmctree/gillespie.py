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
from random import expovariate, uniform, random

import networkx as nx

from .trajectory import get_node_to_tm, Event, LightTrajectory


def unsafe_dict_random_choice(d):
    x = random()
    for i, w in d.items():
        x -= w
        if x < 0:
            return i


def dict_random_choice(d):
    total = sum(d.values())
    x = uniform(0, total)
    for i, w in d.items():
        x -= w
        if x < 0:
            return i


def expand_Q(Q):
    state_to_rate = Q.out_degree(weight='weight')
    state_to_distn = dict()
    for sa in Q:
        rate = state_to_rate[sa]
        distn = dict((sb, Q[sa][sb]['weight'] / rate) for sb in Q[sa])
        state_to_distn[sa] = distn
    return state_to_rate, state_to_distn


def expand_edge_to_Q(edge_to_Q):
    # Compute expansion of Q on each edge.
    edge_to_state_to_rate = {}
    edge_to_state_to_distn = {}
    for edge, Q in edge_to_Q.items():
        state_to_rate, state_to_distn = expand_Q(Q)
        edge_to_state_to_rate[edge] = state_to_rate
        edge_to_state_to_distn[edge] = state_to_distn
    return edge_to_state_to_rate, edge_to_state_to_distn


def _get_trajectory(T, root, root_prior_distn,
        edge_to_rate, edge_to_blen, node_to_tm, bfs_edges,
        edge_to_state_to_rate, edge_to_state_to_distn,
        ):
    """
    This helper function uses precalculated information.

    """
    # Initialize the trajectory with an arbitrary name and no history or events.
    track = LightTrajectory(name='gillespie', history={}, events={})

    # Sample the state at the root.
    track.history[root] = unsafe_dict_random_choice(root_prior_distn)

    # Sample events along edges.
    for edge in bfs_edges:

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
            sb = unsafe_dict_random_choice(state_to_distn[state])
            ev = Event(track=track, tm=tm, sa=state, sb=sb)
            track.events[edge].append(ev)
            state = sb
        track.history[nb] = state

    # Return the track.
    return track


def get_trajectory(T, root, root_prior_distn,
        edge_to_rate, edge_to_blen, edge_to_Q):
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
    edge_to_Q : dict
        map from edge to networkx DiGraph rate matrix

    """
    # Precalculate some stuff.
    node_to_tm = get_node_to_tm(T, root, edge_to_blen)
    bfs_edges = list(nx.bfs_edges(T, root))
    edge_to_state_to_rate, edge_to_state_to_distn = expand_edge_to_Q(edge_to_Q)

    # Sample a track.
    track = _get_trajectory(T, root, root_prior_distn,
            edge_to_rate, edge_to_blen, node_to_tm, bfs_edges,
            edge_to_state_to_rate, edge_to_state_to_distn)
    return track


def gen_trajectories(T, root, root_prior_distn,
        edge_to_rate, edge_to_blen, edge_to_Q, ntrajectories=None):
    """
    Avoid redundant calculations shared across trajectories.

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
    edge_to_Q : dict
        map from edge to networkx DiGraph rate matrix
    ntrajectories : integer, optional
        optionally limit the number of sampled trajectories

    """
    # Precalculate some stuff.
    node_to_tm = get_node_to_tm(T, root, edge_to_blen)
    bfs_edges = list(nx.bfs_edges(T, root))
    edge_to_state_to_rate, edge_to_state_to_distn = expand_edge_to_Q(edge_to_Q)

    # Sample a bunch of tracks.
    nsampled = 0
    while ntrajectories is None or nsampled < ntrajectories:
        track = _get_trajectory(T, root, root_prior_distn,
                edge_to_rate, edge_to_blen, node_to_tm, bfs_edges,
                edge_to_state_to_rate, edge_to_state_to_distn)
        yield track
        nsampled += 1


def get_incomplete_sample(T, root, root_prior_distn,
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
    node_to_state = {root : unsafe_dict_random_choice(root_prior_distn)}

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
            state = unsafe_dict_random_choice(state_to_distn[state])
        node_to_state[nb] = state

    # Return the map from node to state.
    return node_to_state

