"""
Vanilla CTBN-free Rao-Teh sampling of trajectories on a tree.

Assume a rooted tree with a known prior root distribution
and edge-specific rate matrices, possibly extra edge-specific
rate scaling factors, and data consisting of sets of feasible states at nodes.
Only Python and networkx data structures are used (not numpy).

Rooted trees are represented by networkx DiGraphs, as are rate matrices.
Finite distributions are represented by Python dictionaries.
The component of the sampler that requires dynamic programming on trees
uses the dynamic_fset_lhood function in the nxmctree package.

The trajectory data structures on trees are the same as used for unconditional
forward sampling using the Gillespie algorithm.

"""
from __future__ import division, print_function, absolute_import

import math

import nxmctree
from nxmctree.sampling import sample_history

from .uniformization import get_rates_out, get_omega, get_uniformized_P
from .chunking import ChunkNodeInfo, ChunkTreeInfo, trajectory_to_chunk_tree


def resample_states(
        T, edge_to_P, root, root_prior_distn, node_to_data_fset,
        track, set_of_all_states):
    """
    """
    ct_info = trajectory_to_chunk_tree(T, edge_to_P, root, track)

    # Propagate the data constraints from structural nodes to chunk nodes.
    chunk_node_to_data_fset = {}
    for chunk_node in ct_info.T:
        cn_info = ct_info.node_to_info[chunk_node]
        fset = set(set_of_all_states)
        for sn in cn_info.structural_nodes:
            fset &= node_to_data_fset[sn]
        chunk_node_to_data_fset[chunk_node] = fset

    # Because nxmctree does not have flexible functions,
    # convert the fset constraints to lmap constraints.
    chunk_node_to_data_lmap = {}
    for node, fset in chunk_node_to_data_fset.items():
        lmap = dict((s, 1) for s in fset)
        chunk_node_to_data_lmap[node] = lmap

    # Resample the states at the chunk nodes.
    cn_to_state = sample_history(ct_info.T, ct_info.edge_to_P, ct_info.root,
            root_prior_distn, chunk_node_to_data_lmap)
    
    # Propagate the sampled chunk node states back onto the
    # events and the structural nodes.
    for chunk_node, state in cn_to_state.items():
        cn_info = ct_info.node_to_info[chunk_node]

        # Update the track history at the structural nodes.
        for sn in cn_info.structural_nodes:
            track.history[sn] = state

        # Update the initial state of transitions corresponding to events.
        for ev in cn_info.downstream_events:
            ev.init_sa(state)

        # Update the final state of transitions corresponding to events.
        for ev in cn_info.upstream_events:
            ev.init_sb(state)


def get_poisson_info(T, root, edge_to_Q, edge_to_rate,
        node_to_tm, bfs_edges):
    """

    Parameters
    ----------
    T : directed networkx tree graph
        Edge and node annotations are ignored.
    root : hashable
        This is the root node.
        Following networkx convention, this may be anything hashable.
    edge_to_Q : dict
        map from edge to edge-specific transition rate matrix.
        This does not include consideration of the edge-specific rate
        scaling factor.
    edge_to_rate : dict
        map from edge to edge-specific rate scaling factor.
    node_to_tm : dict
        precalculated node times with respect to root
    bfs_edges : sequence
        precalculated edges in pre-order

    Returns
    -------
    edge_to_P : dict
        map from edge to uniformized transition probability matrix
    edge_to_poisson_rates : dict
        map from edge to map from state to event poisson rate

    """
    uniformization_factor = 2
    rates_out = uniformization.get_rates_out(Q)
    omega = get_omega(rates_out, uniformization_factor)
    edge_to_P = {}
    edge_to_poisson_rates = {}
    for edge, Q in edge_to_Q.items():
        edge_rate = edge_to_rate[edge]
        P = get_uniformized_P(Q, rates_out, omega)
        edge_to_P[edge] = P
        poisson_rates = dict((s, edge_rate * r) for s, r in rates_out.items())
        edge_to_poisson_rates[edge] = poisson_rates
    return edge_to_P, edge_to_poisson_rates


def add_poisson_events(T, root, node_to_tm, edge_to_poisson_rates, track):
    """
    Add poisson events onto edges on a track.

    """
    for edge in T.edges():
        na, nb = edge
        sa = track.history[na]
        sb = track.history[nb]
        tma = node_to_tm[na]
        tmb = node_to_tm[nb]
        poisson_rates = edge_to_poisson_rates[edge]

        # Create triples of (initial time, final time, poisson rate).
        tm = tma
        state = sa
        triples = []
        for ev in sorted(T.events[edge]):
            triples.append((tm, ev.tm, poisson_rates[state]))
            tm = ev.tm
        triples.append((tm, tmb, poisson_rates[state]))

        # For each triple, sample some poisson event times.
        new_events = []
        for ta, tb, poisson_rate in triples:
            t = ta
            while True:
                t += expovariate(poisson_rate)
                if t >= tb:
                    break
                ev = Event(track=track, tm=t, sa=None, sb=None)
                new_events.append(ev)

        # Extend the list of events with the new events.
        track.events[edge].extend(new_events)

