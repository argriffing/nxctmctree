"""
A data structure for a state trajectory on a fixed tree.

In the notation of this module, the 'history' refers to the joint
states at nodes of the fixed tree, whereas 'events' refer to substitutions or
self-substitutions that occur on edges of the fixed tree.
Event times are with respect to the root of the fixed tree,
not with respect to an edge endpoint.

"""
from __future__ import division, print_function, absolute_import

import warnings

import networkx as nx

#TODO use this module for base classes in nxblink/trajectory.py


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

