"""
Convert rate matrices to uniformized transition probability matrices.

This does not use the matrix exponential,
and it does not compute transition probability matrices that
account for integration over unobserved events.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx


__all__ = ['get_rates_out', 'get_omega', 'get_uniformized_P']


def get_rates_out(Q):
    """
    Get the rates of leaving each state in transition rate matrix Q.

    Parameters
    ----------
    Q : networkx DiGraph
        transition rates

    Returns
    -------
    rates_out : dict
        map from state to the total rate of leaving the state

    """
    return Q.degree_out(weight='weight')


def get_omega(rates_out, uniformization_factor):
    """
    Omega is a uniformization rate in the notation of Rao and Teh.

    Parameters
    ----------
    rates_out : dict
        map from state to the total rate of leaving the state
    uniformization_factor : float
        a factor greater than 1.0

    Returns
    -------
    omega : float
        a uniformized rate

    """
    uf = uniformization_factor
    if uf <= 1:
        raise Exception('invalid uniformization factor %s' % uf)
    return uf * max(rates_out.values())


def get_uniformized_P(Q, rates_out, omega):
    """
    Get the transition probability matrix under uniformization.

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

