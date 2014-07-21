"""
A data structure for a state trajectory on a fixed tree.

In the notation of this module, the 'history' refers to the joint
states at nodes of the fixed tree, whereas 'events' refer to substitutions or
self-substitutions that occur on edges of the fixed tree.
Event times are with respect to the root of the fixed tree,
not with respect to an edge endpoint.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict
import warnings

import networkx as nx


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


class NodeStateSummary(object):
    """
    Record joint states for a subset of nodes in sampled trajectories.

    This is the information that would be available, for example,
    in a sequence alignment for which each aligned site corresponds
    to a sampled trajectory and the list of extant taxa corresponds
    to the subset of nodes for which the state is observable.

    Parameters
    ----------
    observable_nodes : sequence
        The ordered sequence of nodes whose joint states are to be stored.

    """
    def __init__(self, observable_nodes):
        self.observable_nodes = observable_nodes
        self.joint_states_to_count = defaultdict(int)

    def on_track(self, track, track_weight=None):
        if track_weight is not None:
            raise NotImplementedError
        joint_states = tuple(track.history[v] for v in self.observable_nodes)
        self.joint_states_to_count[joint_states] += 1

    def gen_xmaps_with_repetition(self):
        for joint_states, count in self.joint_states_to_count.items():
            xmap = dict(zip(self.observable_nodes, joint_states))
            for i in range(count):
                yield xmap

    def gen_xmap_count_pairs(self):
        # An xmap maps nodes to states and includes only nodes of interest.
        for joint_states, count in self.joint_states_to_count.items():
            xmap = dict(zip(self.observable_nodes, joint_states))
            yield xmap, count


class FullTrackSummary(object):
    """
    Record everything possibly relevant for trajectory likelihood calculation.

    These will be sufficient statistics
    but probably not minimial sufficient statistics.

    The usual usage is to create the object,
    then sample a bunch of tracks, calling on_track for each sample,
    then doing things with the summaries.

    An alternative usage could be to directly call the root state,
    transition, and dwell notification functions directly from a trajectory
    sampler, bypassing the step of actually constructing the trajectory object.

    """
    def __init__(self, T, root, edge_to_blen):
        # Store some input parameters and
        # precompute some properties of the tree.
        self.T = T
        self.root = root
        self.node_to_tm = get_node_to_tm(T, root, edge_to_blen)
        self.bfs_edges = list(nx.bfs_edges(T, root))

        # Initialize trajectory summary.
        self.root_state_to_count = defaultdict(int)
        self.edge_to_transition_to_count = {}
        self.edge_to_state_to_time = {}

    def on_root_state(self, root_state, track_weight=None):
        if track_weight is None:
            track_weight = 1
        self.root_state_to_count[root_state] += track_weight

    def on_transition(self, edge, sa, sb, track_weight=None):
        if track_weight is None:
            track_weight = 1
        transition = (sa, sb)
        if edge not in self.edge_to_transition_to_count:
            self.edge_to_transition_to_count[edge] = defaultdict(int)
        transition_to_count = self.edge_to_transition_to_count[edge]
        transition_to_count[transition] += track_weight

    def on_dwell(self, edge, state, dwell, track_weight=None):
        if track_weight is None:
            track_weight = 1
        if edge not in self.edge_to_state_to_time:
            self.edge_to_state_to_time[edge] = defaultdict(float)
        state_to_time = self.edge_to_state_to_time[edge]
        state_to_time[state] += dwell * track_weight

    def assert_valid_event(self, state, tm, ev):
        if ev.sa == ev.sb:
            raise Exception('self transitions are not allowed')
        if ev.sa != state:
            raise Exception('the initial state of the transition is invalid')
        if ev.tm <= tm:
            raise Exception('the time of the transition is invalid')

    def on_track(self, track, track_weight=None):
        """
        The track weight could be used for multiplicity of site patterns.

        """
        if track_weight is None:
            track_weight = 1
        self.on_root_state(track.history[self.root], track_weight)
        for edge in self.bfs_edges:
            na, nb = edge
            tma = self.node_to_tm[na]
            tmb = self.node_to_tm[nb]
            state = track.history[na]
            tm = tma
            events = sorted(track.events[edge])
            for ev in events:
                self.assert_valid_event(state, tm, ev)
                self.on_dwell(edge, state, ev.tm - tm, track_weight)
                self.on_transition(edge, ev.sa, ev.sb, track_weight)
                tm = ev.tm
                state = ev.sb
            self.on_dwell(edge, state, tmb - tm, track_weight)


class LightTrajectory(object):
    """
    This base class is used by itself for testing.

    Parameters
    ----------
    name : hashable
        name of the trajectory
    history : dict
        map from structural node to current state.
    events : dict
        map from structural edge to list of events on that edge

    """
    def __init__(self, name=None, history=None, events=None):
        self.name = name
        self.history = history
        self.events = events

    def remove_self_transitions(self):
        edges = set(self.events)
        for edge in edges:
            events = self.events[edge]
            self.events[edge] = [ev for ev in events if ev.sb != ev.sa]

    def clear_state_labels(self):
        """
        Clear the state labels but not the event times.

        """
        nodes = set(self.history)
        edges = set(self.events)
        for v in nodes:
            self.history[v] = None
        for edge in edges:
            for ev in self.events[edge]:
                ev.sa = None
                ev.sb = None

    def __str__(self):
        return 'Track(%s)' % self.name


class Event(object):
    def __init__(self, track=None, tm=None, sa=None, sb=None):
        """

        Parameters
        ----------
        track : Trajectory object, optional
            the trajectory object on which the event occurs
        tm : float, optional
            time along the edge at which the event occurs
        sa : hashable, optional
            initial state of the transition
        sb : hashable, optional
            final state of the transition

        """
        self.track = track
        self.tm = tm
        self.sa = sa
        self.sb = sb

    def init_sa(self, state):
        if self.sa is not None:
            raise Exception('the initial state is already set')
        self.sa = state

    def init_sb(self, state):
        if self.sb is not None:
            raise Exception('the final state is already set')
        self.sb = state

    def init_or_confirm_sb(self, state):
        if self.sb is None:
            self.sb = state
        if self.sb != state:
            raise Exception('final state incompatibility')

    def __repr__(self):
        return 'Event(track=%s, tm=%s, sa=%s, sb=%s)' % (
                self.track, self.tm, self.sa, self.sb)

    def __lt__(self, other):
        """
        Give events a partial order.

        """
        if self.tm == other.tm:
            warnings.warn('simultaneous events')
        return self.tm < other.tm

