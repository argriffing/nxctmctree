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

from random import expovariate
import math

import networkx as nx

import nxmctree
from nxmctree.sampling import sample_history

from .uniformization import get_rates_out, get_omega, get_uniformized_P
from .chunking import ChunkNodeInfo, ChunkTreeInfo, trajectory_to_chunk_tree
from .trajectory import LightTrajectory, Event, get_node_to_tm


def resample_states(
        T, edge_to_P, root, root_prior_distn, node_to_data_fset,
        track, set_of_all_states):
    """
    """
    ct_info = trajectory_to_chunk_tree(T, edge_to_P, root, track)

    # Propagate the data constraints from structural nodes to chunk nodes.
    print('found', len(ct_info.T), 'chunk nodes')
    chunk_node_to_data_fset = {}
    for cn in ct_info.T:
        cn_info = ct_info.node_to_info[cn]
        fset = set(set_of_all_states)
        #print('fset, before:', fset)
        nstructural = len(cn_info.structural_nodes)
        print('found', nstructural, 'structural nodes in chunk node', cn)
        for sn in cn_info.structural_nodes:
            fset &= node_to_data_fset[sn]
        if not fset:
            raise Exception('chunk node has no feasible state')
        #print('fset, after:', fset)
        forbidden = set(set_of_all_states) - fset
        if forbidden:
            print('forbidden states at chunk node', cn, ':', forbidden)
        chunk_node_to_data_fset[cn] = fset

    # Because nxmctree does not have flexible functions,
    # convert the fset constraints to lmap constraints.
    chunk_node_to_data_lmap = {}
    for node, fset in chunk_node_to_data_fset.items():
        lmap = dict((s, 1) for s in fset)
        chunk_node_to_data_lmap[node] = lmap

    # Resample the states at the chunk nodes.
    cn_to_state = sample_history(ct_info.T, ct_info.edge_to_P, ct_info.root,
            root_prior_distn, chunk_node_to_data_lmap)

    # Check that all chunk nodes have been assigned states.
    missing = set(ct_info.node_to_info) - set(cn_to_state)
    if missing:
        raise Exception('the following chunk nodes were not included '
                'in the history sampled on the chunk tree: %s' % missing)
    
    # Propagate the sampled chunk node states back onto the
    # events and the structural nodes.
    for chunk_node, state in cn_to_state.items():
        if state is None:
            raise Exception('chunk node % has state %s' % (chunk_node, state))

        # Get the history and event information for the chunk node.
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

    # Check that the track history is complete.
    for node, state in track.history.items():
        if state is None:
            raise Exception('failed to set the history '
                    'of structural node %s' % node)


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
    edge_to_P = {}
    edge_to_poisson_rates = {}
    for edge, Q in edge_to_Q.items():
        rates_out = get_rates_out(Q)
        omega = get_omega(rates_out, uniformization_factor)
        edge_rate = edge_to_rate[edge]
        edge_to_P[edge] = get_uniformized_P(Q, rates_out, omega)
        poisson_rates = {}
        for state, rate_out in rates_out.items():
            poisson_rates[state] = (omega - rate_out) * edge_rate
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
        for ev in sorted(track.events[edge]):
            triples.append((tm, ev.tm, poisson_rates[state]))
            tm = ev.tm
            state = ev.sb
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


def get_feasible_blank_trajectory(
        T, root, root_prior_distn, edge_to_Q, node_to_tm):
    """
    Create a blank trajectory that should be feasible.

    The events are placed on the edges of the tree,
    but none of the structural node states are determined,
    and none of the event transition natures are determined.

    """
    history = dict((v, None) for v in T)
    events = dict((e, None) for e in T.edges())
    track = LightTrajectory(name='mytrack', history=history, events=events)
    for edge, Q in edge_to_Q.items():
        na, nb = edge
        tma = node_to_tm[na]
        tmb = node_to_tm[nb]
        track.events[edge] = []

        # Use the diameter of the transition rate matrix graph
        # to decide how many events to place on the edge.
        diameter = nx.diameter(Q)
        dt = (tmb - tma) / diameter
        for i in range(1, diameter):
            tm = tma + dt * i
            ev = Event(track=track, tm=tm, sa=None, sb=None)
            track.events[edge].append(ev)

    # Return the track.
    return track


def gen_raoteh_trajectories(
        T, edge_to_Q, root, root_prior_distn, node_to_data_fset,
        edge_to_blen, edge_to_rate,
        set_of_all_states, initial_track=None, ntrajectories=None):
    """
    Yield non-independently sampled trajectories.

    The first few trajectories may have low probability,
    but the probability should be positive.

    """
    # Extract properties of the tree.
    node_to_tm = get_node_to_tm(T, root, edge_to_blen)
    bfs_edges = list(nx.bfs_edges(T, root))

    # Get structures to help with uniformization.
    edge_to_P, edge_to_poisson_rates = get_poisson_info(
            T, root, edge_to_Q, edge_to_rate,
            node_to_tm, bfs_edges)

    # Initialize a blank track if none has been provided.
    # Otherwise prepare the provided track for state sampling.
    if initial_track is None:
        track = get_feasible_blank_trajectory(
                T, root, root_prior_distn, edge_to_Q, node_to_tm)
    else:
        track = initial_track
        add_poisson_events(T, root, node_to_tm, edge_to_poisson_rates, track)
        track.clear_state_labels()

    # Sample a bunch of tracks.
    nsampled = 0
    while True:

        # Sample states on the track.
        resample_states(
                T, edge_to_P, root, root_prior_distn, node_to_data_fset,
                track, set_of_all_states)

        # Clear the self-transition events.
        track.remove_self_transitions()

        # Yield the track.
        yield track
        nsampled += 1

        # Check if we have finished.
        if ntrajectories is not None and nsampled >= ntrajectories:
            return

        # Add poisson sampled events to the trajectory.
        add_poisson_events(T, root, node_to_tm, edge_to_poisson_rates, track)

        # Clear trajectory state labels.
        track.clear_state_labels()

