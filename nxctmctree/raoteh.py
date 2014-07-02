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


def get_rates_out(Q):
    return Q.degree_out(weight='weight')


def get_omega(rates_out, uniformization_factor):
    """
    Omega is a uniformization rate in the notation of Rao and Teh.

    """
    return uniformization_factor * max(rates_out.values())


def get_uniformized_P(Q, rates_out, omega):
    """

    Parameters
    ----------
    Q : directed weighted networkx graph
        Rate matrix.
    rates_out : dict
        map from state to sum of rates out of the state
    omega : float
        a rate larger than the max rate out

    Returns
    -------
    P : directed weighted networkx graph
        Transition probability matrix.

    """
    P = nx.DiGraph()
    for sa in Q:
        p = 1.0 - rates_out[sa] / omega
        P.add_edge(sa, sa, weight=p)
        for sb in Q[sa]:
            p = Q[sa][sb]['weight'] / omega
            P.add_edge(sa, sb, weight=p)
    return P


class ChunkNodeInfo(object):
    def __init__(self):
        self.structural_nodes = []
        self.upstream_events = []
        self.downstream_events = []

    def on_structural_node(self, node):
        self.structural_nodes.append(node)

    def on_upstream_event(self, event):
        self.upstream_events.append(event)

    def on_downstream_event(self, event):
        self.downstream_events.append(event)


class ChunkTreeInfo(object):
    """
    The on_x() member functions are called during iteration over the trajectory.

    """
    def __init__(self):
        self.T = DiGraph()
        self.edge_to_P = {}
        self.root = None
        self.next_idx = 0
        self.node_to_info = {}

    def _request_idx(self):
        idx = self.next_idx
        self.next_idx += 1
        return idx

    def on_root(self):
        self.root = self._request_idx()
        return self.root

    def on_event(self, event, upstream_cn, P):
        """

        Parameters
        ----------
        event : Event object
            event on the trajectory
        upstream_cn : integer
            upstream chunk node
        P : networkx DiGraph
            uniformized transition probability matrix

        """
        downstream_cn = self._request_idx()
        edge = (upstream_cn, downstream_cn)
        self.edge_to_P[edge] = P
        self.node_to_info[upstream_cn].on_downstream_event(event)
        self.node_to_info[downstream_cn].on_upstream_event(event)
        return downstream_cn

    def on_structural_node(self, chunk_node, structural_node):
        self.node_to_info[chunk_node].on_structural_node(structural_node)


def trajectory_to_chunk_tree(T, edge_to_P, root, track):
    """
    Convert a trajectory to a chunk tree.

    The purpose of the chunk tree is to facilitate resampling of states
    on the segments of the trajectory.
    Note that the edge_to_P input dictionary should have uniformized transition
    probability matrices.

    Parameters
    ----------
    T : directed networkx tree graph
        Edge and node annotations are ignored.
    edge_to_P : dict
        A map from directed edges of the tree graph
        to networkx graphs representing state transition probabilities.
        This is a uniformized transition probability matrix.
    root : hashable
        This is the root node.
        Following networkx convention, this may be anything hashable.
    root_prior_distn : dict
        Prior state distribution at the root.
    track : Trajectory object
        The chunk tree should be constructed from this trajectory object.

    Returns
    -------
    chunk_tree : networkx DiGraph
        rooted directed chunk tree
    chunk_root : hashable
        root node of the chunk tree
    chunk_edge_to_P : dict
        map from chunk tree edges to transition probability matrices

    """
    # Some naming conventions in this function:
    #   ct : chunk tree
    #   cn : chunk node
    #   sn : structural node
    ct_info = ChunkTreeInfo()
    ct_root = chunk_tree_info.on_root(root_prior_distn)
    sn_to_cn = {root : ct_root}

    # Iterate over edges of the structural tree.
    for edge in nx.bfs_edges(T, root):
        na, nb = edge
        P = edge_to_P[edge]
        chunk_node = sn_to_cn[na]

        # Iterate over events on the tree.
        events = sorted(track.events[edge])
        for ev in events:

            # At each event a new chunk node is created.
            chunk_node = ct_info.on_event(ev, chunk_node, P)

        # Associate the chunk node with the structural node
        # at the downstream endpoint of the edge.
        ct_info.on_structural_node(chunk_node, structural_node)
        sn_to_cn[nb] = chunk_node

    # Return the chunk tree info.
    return ct_info


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
    edge_to_rates_out = {}
    #TODO unfinished...


def get_edge_to_state_to_poisson_rate(T, root, edge_to_Q, edge_to_rate):
    """
    Compute poisson rates.

    This depends on the 

    """
    for 
def get_rates_out(Q):
    return Q.degree_out(weight='weight')


def get_omega(rates_out, uniformization_factor):
    """
    Omega is a uniformization rate in the notation of Rao and Teh.

    """
    return uniformization_factor * max(rates_out.values())


def get_uniformized_P(Q, rates_out, omega):


def add_poisson_events(T, root,
        node_to_tm, bfs_edges,
        edge_to_state_to_poisson_rate, track):
    """
    Add poisson events onto a track.

    """
    pass

