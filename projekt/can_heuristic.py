"""
Two-Level Heuristic Algorithm for CAN Optimization
====================================================
Gąsior & Drwal (2012), IEEE/IPSJ SAINT 2012.

Maximize  Q(x,y) = U(y) - G(x,y) - H(x0)
  Upper level : rate allocation  →  concave max, solved with SLSQP
  Lower level : data placement   →  Lagrangian relaxation + subgradient method

Expected result (paper Table II, iteration 1):
  Placement : Object 1 on Server 1 AND Server 2  (replicated)
              Object 2 on Server 2 only
  Q ≈ 73.7

Install: pip install numpy scipy
Run    : python can_heuristic.py
"""

import numpy as np
import scipy.optimize as opt

# ─────────────────────────────────────────────────────────────────────────────
# NETWORK PARAMETERS  (Section IV computational example)
# ─────────────────────────────────────────────────────────────────────────────

M = 2    # clients  (m = 1..M;  m = 0 = publisher)
N = 2    # data objects
S = 2    # cache servers
L = 2    # network links

Bs  = np.array([1.0, 2.0])             # server storage capacities
bn  = np.array([1.0, 1.0])             # object sizes
Cl  = np.array([10.0, 10.0])           # link capacities (MB/s)
kl  = np.array([5.0,  2.0])            # link unit bandwidth cost
dns = np.array([[0.4, 1.0],            # storage cost  dns[n, s]
                [0.4, 1.0]])
w       = 20.0    # logarithmic utility weight ("willingness-to-pay")
ymn_min =  0.1    # minimum QoS rate
ymn_max = 10.0    # maximum rate cap

# Routing: server s exclusively uses link s
# a[m, n, s, l] = 1  iff  s == l   (0-indexed; m=0 → publisher)
a = np.zeros((M + 1, N, S, L))
for _m in range(M + 1):
    for _n in range(N):
        for _s in range(S):
            a[_m, _n, _s, _s] = 1


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY & COST
# ─────────────────────────────────────────────────────────────────────────────

def U(y):    return w * np.log(max(float(y), 1e-12))
def dU(y):   return w / max(float(y), 1e-12)

def G(x0, xmns, y0, ymn):
    """Link bandwidth cost G(x,y)."""
    cost = 0.0
    for n in range(N):
        for s in range(S):
            if x0[n, s]:
                cost += np.dot(kl, a[0, n, s, :]) * y0[n, s]
    for m in range(M):
        for n in range(N):
            for s in range(S):
                if xmns[m, n, s]:
                    cost += np.dot(kl, a[m+1, n, s, :]) * ymn[m, n]
    return cost

def H(x0):
    """Storage cost H(x0)."""
    return float(np.sum(dns * x0))

def Q_profit(x0, xmns, y0, ymn):
    """Total profit Q = U − G − H."""
    util  = sum(U(y0[n, s]) for n in range(N) for s in range(S) if x0[n, s])
    util += sum(U(ymn[m, n]) for m in range(M) for n in range(N))
    return util - G(x0, xmns, y0, ymn) - H(x0)

def is_feasible(x0, xmns):
    if any(x0[n, :].sum() < 1 for n in range(N)):                 return False
    if any(xmns[m, n, :].sum() != 1
           for m in range(M) for n in range(N)):                   return False
    if np.any(xmns > x0[np.newaxis, :, :]):                       return False
    if any((x0[:, s] * bn).sum() > Bs[s] for s in range(S)):     return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# UPPER LEVEL: Rate Allocation  (Section III-A)
# ─────────────────────────────────────────────────────────────────────────────

def solve_rate_allocation(x0, xmns):
    """
    Given fixed placement x0[N,S] and assignment xmns[M,N,S], solve:
        max_y  U(y) − G(x', y)
    Uses SLSQP with analytic gradient.
    Returns y0[N,S], ymn[M,N], objective_value, converged.
    """
    pub_pairs  = [(n, s) for n in range(N) for s in range(S) if x0[n, s]]
    cli_assign = {(m, n): next(s for s in range(S) if xmns[m, n, s])
                  for m in range(M) for n in range(N)}

    n_pub  = len(pub_pairs)
    n_vars = n_pub + M * N

    pi  = lambda i: i
    ci  = lambda m, n: n_pub + m * N + n

    def link_loads(y):
        ld = np.zeros(L)
        for i, (n, s) in enumerate(pub_pairs):
            ld += a[0, n, s, :] * y[pi(i)]
        for (m, n), s in cli_assign.items():
            ld += a[m+1, n, s, :] * y[ci(m, n)]
        return ld

    def neg_obj(y):
        util  = sum(U(y[pi(i)])    for i in range(n_pub))
        util += sum(U(y[ci(m, n)]) for m in range(M) for n in range(N))
        return -(util - np.dot(kl, link_loads(y)))

    def neg_grad(y):
        g = np.zeros(n_vars)
        for i, (n, s) in enumerate(pub_pairs):
            g[pi(i)]    = -(dU(y[pi(i)])    - np.dot(kl, a[0,   n, s, :]))
        for (m, n), s in cli_assign.items():
            g[ci(m, n)] = -(dU(y[ci(m, n)]) - np.dot(kl, a[m+1, n, s, :]))
        return g

    constraints = [{'type': 'ineq',
                    'fun': lambda y, l=l: Cl[l] - link_loads(y)[l]}
                   for l in range(L)]
    bounds  = [(ymn_min, ymn_max)] * n_vars
    y_init  = np.full(n_vars, ymn_min * 2.0)

    res = opt.minimize(neg_obj, y_init, jac=neg_grad, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'ftol': 1e-12, 'maxiter': 5000})

    y0_out  = np.zeros((N, S))
    ymn_out = np.zeros((M, N))
    for i, (n, s) in enumerate(pub_pairs):
        y0_out[n, s]  = res.x[pi(i)]
    for m in range(M):
        for n in range(N):
            ymn_out[m, n] = res.x[ci(m, n)]

    return y0_out, ymn_out, -res.fun, res.success


# ─────────────────────────────────────────────────────────────────────────────
# LOWER LEVEL: Data Placement  (Section III-B)
# ─────────────────────────────────────────────────────────────────────────────

def _knapsack(values, sizes, capacity):
    """
    0-1 knapsack: maximise sum(values*x) s.t. sum(sizes*x) <= capacity.
    Empty selection has value 0; negative-value items are never chosen.
    """
    cap = int(round(capacity))
    n   = len(values)
    dp  = np.full(cap + 1, -np.inf);  dp[0] = 0.0
    sel = [[] for _ in range(cap + 1)]
    for i in range(n):
        sz = int(round(sizes[i]))
        for c in range(cap, sz - 1, -1):
            v = dp[c - sz] + values[i]
            if v > dp[c]:
                dp[c] = v
                sel[c] = sel[c - sz] + [i]
    best = int(np.argmax(dp))
    x = np.zeros(n, dtype=int)
    for i in sel[best]:
        x[i] = 1
    return x


def _capacity_aware_enforce(x0, y0):
    """
    Enforce constraint (1): every object placed at least once.
    FIXED: respects server storage capacities when choosing placement server.
    """
    used = np.array([(x0[:, s] * bn).sum() for s in range(S)])
    for n in range(N):
        if x0[n, :].sum() == 0:
            candidates = [
                (dns[n, s] + np.dot(kl, a[0, n, s, :]) * y0[n, s], s)
                for s in range(S) if used[s] + bn[n] <= Bs[s]
            ]
            if not candidates:
                candidates = [(dns[n, s], s) for s in range(S)]
            _, best_s = min(candidates)
            x0[n, best_s] = 1
            used[best_s] += bn[n]
    return x0


def _greedy_assign(x0):
    """Assign each client m to the cheapest server that holds object n."""
    xmns = np.zeros((M, N, S), dtype=int)
    for m in range(M):
        for n in range(N):
            available = [s for s in range(S) if x0[n, s]] or list(range(S))
            best_s = min(available, key=lambda s: np.dot(kl, a[m+1, n, s, :]))
            xmns[m, n, best_s] = 1
    return xmns


def solve_data_placement(y0_ref, ymn_ref, max_iter=500, eps=1e-6, kappa=1.5):
    """
    Given fixed rates y0_ref[N,S] and ymn_ref[M,N] from upper level, solve:
        min_{x}  G(x, y') + H(x0)
    via Lagrangian relaxation + subgradient (Algorithm 1 in paper).

    KEY: Each server's knapsack runs independently, so objects CAN be
    selected by multiple servers simultaneously (replication supported).
    """
    y0 = np.where(y0_ref > ymn_min * 0.5, y0_ref, ymn_min)

    alpha = np.zeros((M, N, S))
    beta  = np.zeros(L)

    LB = -np.inf
    UB =  np.inf
    best_x0   = None
    best_xmns = None

    for _t in range(max_iter):

        # ── Knapsack per server: which objects to cache? (eq. 10) ──────────
        x0 = np.zeros((N, S), dtype=int)
        for s in range(S):
            vals = np.array([
                sum(alpha[m, n, s] for m in range(M))
                - np.dot(kl, a[0, n, s, :]) * y0[n, s]
                - dns[n, s]
                - ymn_min * np.dot(beta, a[0, n, s, :])
                for n in range(N)
            ])
            x0[:, s] = _knapsack(vals, bn, Bs[s])

        x0 = _capacity_aware_enforce(x0, y0)   # Bug 1 fix

        # ── Assignment per (m,n) ────────────────────────────────────────────
        xmns = np.zeros((M, N, S), dtype=int)
        for m in range(M):
            for n in range(N):
                available = [s for s in range(S) if x0[n, s]] or list(range(S))
                costs = {
                    s: (np.dot(kl, a[m+1, n, s, :]) * ymn_ref[m, n]
                        + alpha[m, n, s]
                        + ymn_min * np.dot(beta, a[m+1, n, s, :]))
                    for s in available
                }
                xmns[m, n, min(costs, key=costs.get)] = 1

        # ── Lagrangian value L(x̃, α, β) ────────────────────────────────────
        L_val = (H(x0)
                 + sum(np.dot(kl, a[0, n, s, :]) * x0[n, s] * y0[n, s]
                       for n in range(N) for s in range(S))
                 + sum(np.dot(kl, a[m+1, n, s, :]) * xmns[m, n, s] * ymn_ref[m, n]
                       for m in range(M) for n in range(N) for s in range(S))
                 + sum(alpha[m, n, s] * (xmns[m, n, s] - x0[n, s])
                       for m in range(M) for n in range(N) for s in range(S))
                 + sum(beta[l] * (
                       sum(x0[n, s] * a[0, n, s, l] * ymn_min
                           + sum(xmns[m, n, s] * a[m+1, n, s, l] * ymn_min
                                 for m in range(M))
                           for n in range(N) for s in range(S)) - Cl[l])
                   for l in range(L)))

        # ── Update bounds (Algorithm 1, steps 3–4) ─────────────────────────
        feasible = is_feasible(x0, xmns)
        if feasible:
            if L_val > LB:
                LB = L_val
                best_x0   = x0.copy()
                best_xmns = xmns.copy()
            primal = G(x0, xmns, y0, ymn_ref) + H(x0)
            if primal < UB:
                UB = primal

        if UB == np.inf:
            UB = max(abs(L_val) * 1.1, 1.0)

        gap = UB - LB
        if LB > -np.inf and gap < eps * max(1.0, abs(LB)):
            break

        # ── Subgradient ascent on dual (Algorithm 1, step 5) ───────────────
        D_alpha = (xmns - x0[np.newaxis, :, :]).astype(float)
        D_beta  = np.array([
            sum(x0[n, s] * a[0, n, s, l] * ymn_min
                + sum(xmns[m, n, s] * a[m+1, n, s, l] * ymn_min
                      for m in range(M))
                for n in range(N) for s in range(S)) - Cl[l]
            for l in range(L)
        ])

        norm_a2 = np.sum(D_alpha ** 2)
        norm_b2 = np.sum(D_beta  ** 2)
        if norm_a2 > 1e-14:
            alpha = np.maximum(0.0, alpha + kappa * gap / norm_a2 * D_alpha)
        if norm_b2 > 1e-14:
            beta  = np.maximum(0.0, beta  + kappa * gap / norm_b2 * D_beta)

    if best_x0 is None:
        best_x0   = x0
        best_xmns = _greedy_assign(x0)

    return best_x0, best_xmns


# ─────────────────────────────────────────────────────────────────────────────
# INITIALISATION  (Section III-C)
# ─────────────────────────────────────────────────────────────────────────────

def initial_placement():
    """
    Greedy capacity-aware placement at minimum rates (Section III-C).
    Used to obtain a feasible starting point before the first upper-level solve.
    """
    x0   = np.zeros((N, S), dtype=int)
    used = np.zeros(S)
    for n in range(N):
        candidates = [
            (dns[n, s] + np.dot(kl, a[0, n, s, :]) * ymn_min, s)
            for s in range(S) if used[s] + bn[n] <= Bs[s]
        ]
        if not candidates:
            candidates = [(dns[n, s], s) for s in range(S)]
        _, best_s = min(candidates)
        x0[n, best_s] = 1
        used[best_s] += bn[n]
    return x0


# ─────────────────────────────────────────────────────────────────────────────
# TWO-LEVEL HEURISTIC  (Section III)
# ─────────────────────────────────────────────────────────────────────────────

def two_level_heuristic(max_outer_iter=10):
    print("=" * 62)
    print("  Two-Level CAN Heuristic  —  Gąsior & Drwal (2012)")
    print("=" * 62)

    print("\n[Init] Greedy placement at minimum rates...")
    x0   = initial_placement()
    xmns = _greedy_assign(x0)
    print(f"[Init] x0:\n{x0}")
    print(f"[Init] Feasible: {is_feasible(x0, xmns)}")

    best_Q   = -np.inf
    best_sol = None

    for it in range(1, max_outer_iter + 1):
        print(f"\n{'─'*62}\n  Iteration {it}\n{'─'*62}")

        # Upper level
        y0, ymn, obj_val, ok = solve_rate_allocation(x0, xmns)
        if not ok:
            print("  [!] Rate solver may not have fully converged.")

        Q = Q_profit(x0, xmns, y0, ymn)
        print(f"  [Upper] Q = {Q:.4f}   (rate-obj = {obj_val:.4f})")
        print(f"  [Upper] Publisher rates y0[n,s]:")
        for n in range(N):
            for s in range(S):
                if y0[n, s] > ymn_min * 0.5:
                    print(f"           Object {n+1} → Server {s+1}: {y0[n,s]:.4f}")
        print(f"  [Upper] Client rates ŷ[m,n]:")
        for m in range(M):
            for n in range(N):
                print(f"           Client {m+1} Object {n+1}: {ymn[m,n]:.4f}")

        if Q <= best_Q:
            print(f"\n  [Stop] Q did not improve ({Q:.4f} ≤ {best_Q:.4f}). Converged.")
            break

        best_Q   = Q
        best_sol = (x0.copy(), xmns.copy(), y0.copy(), ymn.copy())

        # Lower level
        x0_new, xmns_new = solve_data_placement(y0, ymn)
        replication = [(n+1, s+1) for n in range(N) for s in range(S) if x0_new[n,s]]
        print(f"  [Lower] New x0:\n{x0_new}  →  objects cached: {replication}")
        print(f"  [Lower] Feasible: {is_feasible(x0_new, xmns_new)}")

        if np.array_equal(x0_new, x0) and np.array_equal(xmns_new, xmns):
            print("  [Stop] Placement unchanged. Converged.")
            break

        x0, xmns = x0_new, xmns_new

    if best_sol is None:
        best_sol = (x0, xmns, np.full((N,S), ymn_min), np.full((M,N), ymn_min))

    x0b, xb, y0b, yb = best_sol

    print(f"\n{'='*62}")
    print(f"  FINAL SOLUTION   Q = {best_Q:.4f}")
    print(f"{'='*62}")

    print("\nObject placement (x0):")
    for n in range(N):
        srvs = [f"Server {s+1}" for s in range(S) if x0b[n,s]]
        print(f"  Object {n+1} → {', '.join(srvs)}")

    print("\nClient-server assignments:")
    for m in range(M):
        for n in range(N):
            for s in range(S):
                if xb[m,n,s]:
                    print(f"  Client {m+1} reads Object {n+1} from Server {s+1}")

    print("\nPublisher upload rates:")
    for n in range(N):
        for s in range(S):
            if x0b[n,s]:
                print(f"  y0[{n+1},{s+1}] = {y0b[n,s]:.4f}")

    print("\nClient download rates:")
    for m in range(M):
        for n in range(N):
            print(f"  ŷ[{m+1},{n+1}] = {yb[m,n]:.4f}")

    print("\nLink utilisation:")
    for l in range(L):
        load = (sum(a[0,n,s,l]*x0b[n,s]*y0b[n,s]   for n in range(N) for s in range(S))
              + sum(a[m+1,n,s,l]*xb[m,n,s]*yb[m,n]
                   for m in range(M) for n in range(N) for s in range(S)))
        print(f"  Link {l+1} (cost={kl[l]:.0f}/unit): {load:.4f} / {Cl[l]:.1f}")

    print(f"\nStorage cost H(x0) = {H(x0b):.4f}")
    print(f"\nPaper Table II ref:   Q ≈ 73.7  (optimal, found in iteration 1)")

    return x0b, xb, y0b, yb, best_Q


if __name__ == "__main__":
    two_level_heuristic()
