"""
Break a tree into chunks within which the state cannot change.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx


__all__ = ['ChunkNodeInfo', 'ChunkTreeInfo', 'trajectory_to_chunk_tree']


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
    The member functions are called during iteration over the trajectory.

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

