"""
Functions related to the poisson step of non-CTBN Rao-Teh sampling.

"""
from __future__ import division, print_function, absolute_import

import random

import numpy as np
import networkx as nx

from .util import get_total_rates, get_omega, get_uniformized_P_nx
from .navigation import gen_context_segments
from .trajectory import Event

#TODO use random instead of np.random to remove dependence on numpy


def poisson_helper(track, rate, tma, tmb):
    """
    Sample poisson events on a segment.

    Parameters
    ----------
    track : Trajectory
        trajectory object for which the poisson events should be sampled
    rate : float
        poisson rate of events
    tma : float
        initial segment time
    tmb : float
        final segment time

    Returns
    -------
    events : list
        list of event objects

    """
    blen = tmb - tma
    nevents = np.random.poisson(rate * blen)
    times = np.random.uniform(low=tma, high=tmb, size=nevents)
    return [Event(track=track, tm=tm) for tm in times]

