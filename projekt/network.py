"""
network.py — shared CAN network instance.

Single source of truth for the topology and per-link / per-object parameters
used by the optimizer (`can_cplex.py`) and the visualizer (`routing_viz.py`).
Edit this file (or replace it) to switch instances; neither consumer needs
any change.

Exports:
    M, N, S, R          dimensions (clients, objects, servers, routers)
    L                   total number of links
    y_min, y_max        per-flow rate bounds
    rho_max             maximum link utilisation
    bn, Cl, tau3        object sizes / link capacities / constant link delays
    router_of           map  server s → router index
    a_us [M+1, S, L]    routing, user↔server↔link
    a    [M+1, N, S, L] same, broadcast over the object axis
"""

import numpy as np


# =============================================================================
# INSTANCE PARAMETERS
# =============================================================================

M = 2    # clients
N = 2    # objects
S = 2    # servers
R = 2    # routers

y_min   = 0.1
y_max   = 10.0
rho_max = 0.95

bn   = np.ones(N)                       # object sizes (paper: 1 MB each)
# Cl / tau3 filled below once L is known.


# =============================================================================
# LINK LAYOUT — multi-router topology, no backbone
# =============================================================================
# Every user has a dedicated access link to EACH router; every server
# attaches to exactly one router (round-robin by default).  Every flow
# takes a 2-hop path:  user → router → server.
#
#   l = u·R + r                 L_cli(u, r)  user u ↔ router r
#   l = (M+1)·R + s             L_srv(s)     router(router_of[s]) ↔ server s

def L_cli(u, r): return u * R + r
def L_srv(s):    return (M + 1) * R + s

L = (M + 1) * R + S

Cl   = np.full(L, 10.0)      # every link 10 MB/s
tau3 = np.zeros(L)           # no constant per-link delay

router_of = np.array([s % R for s in range(S)], dtype=int)


# =============================================================================
# ROUTING TENSOR
# =============================================================================

a_us = np.zeros((M + 1, S, L), dtype=int)
for u in range(M + 1):
    for s in range(S):
        r = router_of[s]
        a_us[u, s, [L_cli(u, r), L_srv(s)]] = 1

a = np.zeros((M + 1, N, S, L), dtype=int)
for n in range(N):
    a[:, n, :, :] = a_us
