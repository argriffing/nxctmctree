"""
Cython implementation of simplified Rao-Teh sampling.

The sampling is simplified in the following ways.
It does not use the continuous-time Bayesian network (CTBN) framework.
In that framework, this simplified scheme would correspond to a network
with a single node.
The second simplification is that it assumes a time-homogeneous
and edge-homogeneous process, with the exception that it allows
edge-specific rate scaling factors.

* original tree
    * global csr tree shape
        * indices
        * indptr
        (column indices are indices[indptr[i] : indptr[i+1])
        (data values are data[indptr[i] : indptr[i+1])
        * per-edge rate scaling factor
        * per-edge event array
            * each event has sink state
            * each event has associated time (wrt. root)

* equilibrium and instantaneous properties of the stochastic process
    * global csr instantaneous state transition sparsity graph
        * indices
        * indptr
        * per-edge rates
    * per-state out-rate (weighted out-degree of the csr matrix)
    * per-state prior probability at the root

* temporary chunk-tree
    * each node points to one or more of the following
        * structural nodes (nodes in the original tree)
        * upstream events (events whose transition leads to the current chunk)
    * each edge is associated with an event on the original tree
        * because of homogeneity, the chunk-tree edges depend on their
          source events only through the edge-rate scaling factor,
          not through other potential complications like edge-specific
          processes or the state of the CTBN Markov blanket at the event.

"""
