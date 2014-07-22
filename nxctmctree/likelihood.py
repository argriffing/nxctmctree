"""
Log likelihood computed using a summary of multiple tracks.

"""
from __future__ import division, print_function, absolute_import

import math


def get_trajectory_log_likelihood(T, root,
        edge_to_Q, edge_to_rate, root_prior_distn, full_track_summary):
    """

    Parameters
    ----------
    T : networkx DiGraph
        rooted tree
    root : hashable
        root node of the rooted networkx tree T
    edge_to_Q : dict
        map from edge to networkx DiGraph rate matrix
    edge_to_rate : dict
        map from edge to rate scaling factor
    root_prior_distn : dict
        prior distribution over states at the root
    full_track_summary : FullTrackSummary object from the trajectory module
        an object that summarizes the trajectories

    Returns
    -------
    log_likelihood : float
        the log likelihood

    """
    root_ll = 0
    trans_ll = 0
    dwell_ll = 0
    for root_state, count in full_track_summary.root_state_to_count.items():
        if count:
            p = root_prior_distn[root_state]
            root_ll += count * math.log(p)
    for edge in T.edges():
        edge_rate = edge_to_rate[edge]
        Q = edge_to_Q[edge]

        # transition contribution
        info = full_track_summary.edge_to_transition_to_count.get(edge, None)
        if info is None:
            # found an edge with no observed transitions
            pass
        else:
            for (sa, sb), count in info.items():
                if count:
                    rate = edge_rate * Q[sa][sb]['weight']
                    trans_ll += count * math.log(rate)

        # dwell time contribution
        info = full_track_summary.edge_to_state_to_time.get(edge, None)
        if info is None:
            raise Exception('found an edge with no observed dwell times')
        for state, duration in info.items():
            if duration:
                rate = edge_rate * Q.out_degree(state, weight='weight')
                dwell_ll -= rate * duration

    # Return log likelihood.
    log_likelihood = root_ll + trans_ll + dwell_ll
    return log_likelihood

