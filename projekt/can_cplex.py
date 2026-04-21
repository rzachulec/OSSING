"""
CAN Optimization — Mean Transmission Delay (Gąsior & Drwal 2012)
================================================================
Faithful implementation of the paper's problem:

    minimize   Q(x, y) = Σ_{m,n,s}  x_mns · τ_mns(x, y)

where
    τ_mns(x, y) = τ^(1)_mns(y_mns)  +  Σ_l a_mnsl · ( τ^(2)_l(x, y) + τ^(3)_l )
    τ^(1)(y_mns) = b_n / y_mns                — source pushing time
    τ^(2)_l      = 1 / (C_l - load_l)         — M/M/1 mean sojourn (paper Sec. 2.3)
    τ^(3)_l      = constant per-link delay    — propagation + processing
    load_l       = Σ_{m,n,s} a_mnsl · x_mns · y_mns

Constraints (paper Sec. 2.2):
    (1) ∀m≥1, n:   Σ_s x_mns = 1
    (2) ∀m≥1, n,s: x_mns ≤ x_0ns
    (3) ∀l:        Σ_{m,n,s} a_mnsl · x_mns · y_mns ≤ C_l
    Storage capacity assumed unbounded (paper's simplifying assumption).

CPLEX modelling technique (the paper uses a decomposition heuristic; we solve
the MIP directly for small instances):
    * bilinear  x_mns · y_mns  linearised by forcing  y = 0 iff x = 0  via
      bigM.  Then  load_l = Σ a_mnsl · y_mns  is plain linear.
    * τ^(1)(y) = 1/y            PWL over y ∈ [y_min, y_max] ∪ {0}, log-spaced.
    * τ^(2)_l(load) = 1/(C-load) PWL over load ∈ [0, ρ_max·C_l], dense near C.
    * x_mns · τ^(2)_l           (binary × continuous) linearised via bigM.

Routing matrix  `a_us[u, s, l]` — user→server→link adjacency.  u=0 publisher,
u=1..M clients.  It is the sole topology input; to load from file replace the
hand-written block below with e.g.:

      a_us = np.loadtxt("routing.txt", dtype=int).reshape(M + 1, S, L)

and persist with  np.savetxt("routing.txt", a_us.reshape(M + 1, -1), fmt="%d").
"""

import numpy as np
from docplex.mp.model import Model


# =============================================================================
# NETWORK INSTANCE   — paper Section IV:  two routers, every link 10 MB/s.
# Generalised to (M clients, S servers) — paper's case is M=2, S=2, L=6.
# =============================================================================

M = 2   # clients
N = 2   # objects
S = 2   # servers

L = M * S + 2                      # = 6 for the M=2, S=2 instance
L_pub   = 0
L_back  = M + 1
def L_cli(u):  return u            # u ∈ {1..M} → link u
def L_srv(s):  return M + 2 + s    # s ∈ {0..S-1}

bn   = np.ones(N)                           # object sizes  b_n
Cl   = np.full(L, 10.0)                     # every link at 10 MB/s (paper)
tau3 = np.zeros(L)                          # no constant link delay

y_min   = 0.1
y_max   = 10.0
rho_max = 0.95

# -----------------------------------------------------------------------------
# Routing  a_us[u, s, l]   u=0 publisher, u=1..M clients
# -----------------------------------------------------------------------------
a_us = np.zeros((M + 1, S, L), dtype=int)
for s_idx in range(S):
    # publisher path: L_pub → L_back → L_srv(s)
    a_us[0, s_idx, [L_pub, L_back, L_srv(s_idx)]] = 1
    # each client path: L_cli(u) → L_back → L_srv(s)
    for u in range(1, M + 1):
        a_us[u, s_idx, [L_cli(u), L_back, L_srv(s_idx)]] = 1

# Broadcast across object dimension (routing is independent of the object).
a = np.zeros((M + 1, N, S, L), dtype=int)
for n in range(N):
    a[:, n, :, :] = a_us


# =============================================================================
# PWL HELPERS
# =============================================================================

def pwl_inverse(mdl, lo, hi, n_bps=18, include_zero=True):
    """PWL of  f(y) = 1/y  over [lo, hi].  (0, 0) added so that inactive
    flows (y=0 when x=0) contribute nothing to the pushing-delay sum."""
    bps  = np.logspace(np.log10(lo), np.log10(hi), n_bps)
    vals = 1.0 / bps
    if include_zero:
        bps  = np.concatenate([[0.0], bps])
        vals = np.concatenate([[0.0], vals])
    postslope = (vals[-1] - vals[-2]) / (bps[-1] - bps[-2])
    return mdl.piecewise(0.0, list(zip(bps.tolist(), vals.tolist())),
                         float(postslope))


def pwl_queue(mdl, C, rho_max=0.9, n_bps=20):
    """PWL of  g(load) = 1/(C - load)  over load ∈ [0, ρ_max · C].
    Log-spaced in the residual (C - load) so breakpoints cluster near C,
    where the function is steep."""
    residual = np.geomspace(C, C * (1 - rho_max), n_bps)   # C → (1-ρ)C
    bps  = np.sort(np.unique(C - residual))                # 0 → ρC
    vals = 1.0 / (C - bps)
    postslope = (vals[-1] - vals[-2]) / (bps[-1] - bps[-2])
    return mdl.piecewise(0.0, list(zip(bps.tolist(), vals.tolist())),
                         float(postslope))


# =============================================================================
# MODEL
# =============================================================================

mdl = Model(name="CAN_MeanDelay_paper")

objects, clients, servers, links = range(N), range(M), range(S), range(L)

# ---- decision variables -----------------------------------------------------

x_0ns = mdl.binary_var_dict(
    ((n, s) for n in objects for s in servers),
    name=lambda ns: f"x0_n{ns[0]}_s{ns[1]}"
)
x_mns = mdl.binary_var_dict(
    ((m, n, s) for m in clients for n in objects for s in servers),
    name=lambda mns: f"xc_m{mns[0]}_n{mns[1]}_s{mns[2]}"
)
y_0ns = mdl.continuous_var_dict(
    ((n, s) for n in objects for s in servers),
    lb=0, ub=y_max, name=lambda ns: f"y0_n{ns[0]}_s{ns[1]}"
)
y_mns = mdl.continuous_var_dict(
    ((m, n, s) for m in clients for n in objects for s in servers),
    lb=0, ub=y_max, name=lambda mns: f"yc_m{mns[0]}_n{mns[1]}_s{mns[2]}"
)

# Aggregate link load and queue delay (PWL result).
load  = mdl.continuous_var_dict(
    links, lb=0, ub=[rho_max * Cl[l] for l in links],
    name=lambda l: f"load_l{l}"
)
delay = mdl.continuous_var_dict(
    links, lb=0, name=lambda l: f"delay_l{l}"
)

# ---- constraints ------------------------------------------------------------

# (1) each (client, object) served by exactly one server
for m in clients:
    for n in objects:
        mdl.add_constraint(
            mdl.sum(x_mns[m, n, s] for s in servers) == 1,
            ctname=f"one_server_m{m}_n{n}"
        )

# (2) feasibility: client can only use a server that holds the object
for m in clients:
    for n in objects:
        for s in servers:
            mdl.add_constraint(
                x_mns[m, n, s] <= x_0ns[n, s],
                ctname=f"feasible_m{m}_n{n}_s{s}"
            )

# # Implied (∃ server for each object) — redundant, helps presolve.
# for n in objects:
#     mdl.add_constraint(
#         mdl.sum(x_0ns[n, s] for s in servers) >= 1,
#         ctname=f"min_placement_n{n}"
#     )

# bigM rate–placement coupling: y = 0 iff x = 0, else y ∈ [y_min, y_max]
for n in objects:
    for s in servers:
        mdl.add_constraint(y_0ns[n, s] >= y_min * x_0ns[n, s],
                           ctname=f"pub_lb_n{n}_s{s}")
        mdl.add_constraint(y_0ns[n, s] <= y_max * x_0ns[n, s],
                           ctname=f"pub_ub_n{n}_s{s}")
for m in clients:
    for n in objects:
        for s in servers:
            mdl.add_constraint(y_mns[m, n, s] >= y_min * x_mns[m, n, s],
                               ctname=f"cli_lb_m{m}_n{n}_s{s}")
            mdl.add_constraint(y_mns[m, n, s] <= y_max * x_mns[m, n, s],
                               ctname=f"cli_ub_m{m}_n{n}_s{s}")

# (3) link load definition; capacity enforced via load[l] upper bound.
for l in links:
    mdl.add_constraint(
        load[l] ==
          mdl.sum(a[0, n, s, l] * y_0ns[n, s]
                  for n in objects for s in servers)
        + mdl.sum(a[m + 1, n, s, l] * y_mns[m, n, s]
                  for m in clients for n in objects for s in servers),
        ctname=f"load_def_l{l}"
    )

# Queue delay per link via PWL of 1/(C_l - load_l).
for l in links:
    pwl_q = pwl_queue(mdl, Cl[l], rho_max=rho_max)
    mdl.add_constraint(delay[l] == pwl_q(load[l]),
                       ctname=f"queue_pwl_l{l}")


# ---- objective components ---------------------------------------------------

pwl_inv = pwl_inverse(mdl, y_min, y_max, n_bps=18)

# τ^(1): Σ_{m,n,s} x_mns · b_n / y_mns
# Since y=0 when x=0 and PWL(0)=0, the x multiplication is absorbed by PWL.
push_delay = (
    mdl.sum(bn[n] * pwl_inv(y_0ns[n, s])
            for n in objects for s in servers)
  + mdl.sum(bn[n] * pwl_inv(y_mns[m, n, s])
            for m in clients for n in objects for s in servers)
)

# τ^(2): Σ_{m,n,s,l} x_mns · a_mnsl · delay_l   (binary × continuous → bigM)
delay_max = np.array([1.0 / ((1 - rho_max) * Cl[l]) for l in links])

queue_terms = []

def _link_term(x_bin, l):
    """Linearise  q = x_bin · delay[l]  with bigM = delay_max[l]."""
    q = mdl.continuous_var(lb=0, ub=delay_max[l])
    mdl.add_constraint(q <= delay_max[l] * x_bin)
    mdl.add_constraint(q <= delay[l])
    mdl.add_constraint(q >= delay[l] - delay_max[l] * (1 - x_bin))
    return q

for n in objects:
    for s in servers:
        for l in links:
            if a[0, n, s, l]:
                queue_terms.append(_link_term(x_0ns[n, s], l))

for m in clients:
    for n in objects:
        for s in servers:
            for l in links:
                if a[m + 1, n, s, l]:
                    queue_terms.append(_link_term(x_mns[m, n, s], l))

queue_delay = mdl.sum(queue_terms)

# τ^(3): linear in placement/assignment.
const_delay = (
    mdl.sum(tau3[l] * a[0, n, s, l] * x_0ns[n, s]
            for n in objects for s in servers for l in links)
  + mdl.sum(tau3[l] * a[m + 1, n, s, l] * x_mns[m, n, s]
            for m in clients for n in objects for s in servers for l in links)
)

mdl.minimize(push_delay + queue_delay + const_delay)


# =============================================================================
# SOLVE & REPORT
# =============================================================================

print("=== Instance ===")
print(f"  M={M}  N={N}  S={S}  L={L}")
print(f"  b_n     = {bn.tolist()}")
print(f"  C_l     = {Cl.tolist()}")
print(f"  τ^(3)_l = {tau3.tolist()}")
print(f"  rate bounds y∈[{y_min},{y_max}],  ρ_max={rho_max}")

print("\n=== Model ===")
mdl.print_information()

sol = mdl.solve(log_output=True)
if not sol:
    print(f"No solution found. Status: {mdl.solve_status}")
    raise SystemExit(1)

Q_pwl = mdl.objective_value
print("\n" + "=" * 64)
print(f"  OPTIMAL  total mean delay  (PWL approx)  Q = {Q_pwl:.5f}")
print("=" * 64)

print("\n-- Placement --")
for n in objects:
    placed = [s for s in servers if x_0ns[n, s].solution_value > 0.5]
    print(f"  object {n} on server(s) {placed}")
    for s in placed:
        print(f"     upload rate y0[{n},{s}] = {y_0ns[n,s].solution_value:.4f}")

print("\n-- Client assignments --")
for m in clients:
    for n in objects:
        for s in servers:
            if x_mns[m, n, s].solution_value > 0.5:
                r = y_mns[m, n, s].solution_value
                print(f"  client {m} ← server {s}  for object {n}   rate {r:.4f}")

# y_mn = Σ_s x_mns · y_mns  (paper Fig. 1 middle plot).
# Since exactly one server is assigned per (m,n), this collapses to the
# rate on the assigned server; for the publisher it's summed over placed
# servers (m=0 can place the same object on multiple servers).
print("\n-- Aggregated transmission rates  y_mn = Σ_s x_mns · y_mns --")
print(f"  {'m':>4}  {'n':>3}  {'y_mn':>8}")
for n in objects:
    y0n = sum(y_0ns[n, s].solution_value
              for s in servers
              if x_0ns[n, s].solution_value > 0.5)
    print(f"  {'pub':>4}  {n:>3}  {y0n:>8.4f}")
for m in clients:
    for n in objects:
        ymn = sum(y_mns[m, n, s].solution_value
                  for s in servers
                  if x_mns[m, n, s].solution_value > 0.5)
        print(f"  {m:>4}  {n:>3}  {ymn:>8.4f}")

print("\n-- Links --")
print(f"  {'l':>2} {'load':>8} {'cap':>6} {'util':>7} {'queue':>10} {'τ^(3)':>8}")
for l in links:
    ld, dl = load[l].solution_value, delay[l].solution_value
    print(f"  {l:>2} {ld:>8.3f} {Cl[l]:>6.1f} {100*ld/Cl[l]:>6.1f}% "
          f"{dl:>10.5f} {tau3[l]:>8.4f}")

# Re-evaluate objective using the EXACT delay formulas (not PWL) to audit
# the quality of the approximation.
active_pub = [(n, s) for n in objects for s in servers
              if x_0ns[n, s].solution_value > 0.5]
active_cli = [(m, n, s) for m in clients for n in objects for s in servers
              if x_mns[m, n, s].solution_value > 0.5]

exact_push = (
    sum(bn[n] / y_0ns[n, s].solution_value for (n, s) in active_pub)
  + sum(bn[n] / y_mns[m, n, s].solution_value for (m, n, s) in active_cli)
)
exact_queue = 0.0
exact_const = 0.0
for l in links:
    C_minus_load = Cl[l] - load[l].solution_value
    q_l = 1.0 / C_minus_load if C_minus_load > 1e-9 else float("inf")
    flows_l = sum(a[0, n, s, l] for (n, s) in active_pub) \
            + sum(a[m + 1, n, s, l] for (m, n, s) in active_cli)
    exact_queue += flows_l * q_l
    exact_const += flows_l * tau3[l]
Q_exact = exact_push + exact_queue + exact_const

print("\n-- Delay decomposition (exact, non-PWL) --")
print(f"  push   Σ b/y         = {exact_push:.5f}")
print(f"  queue  Σ f·1/(C-load)= {exact_queue:.5f}")
print(f"  const  Σ f·τ^(3)     = {exact_const:.5f}")
print(f"  total Q (exact)      = {Q_exact:.5f}")
print(f"  total Q (CPLEX PWL)  = {Q_pwl:.5f}")
print(f"  PWL error            = {abs(Q_pwl - Q_exact):.5f}")
