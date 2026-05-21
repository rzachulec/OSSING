"""
Two-level heuristic for CAN mean-delay minimization based on Section 3 of the
paper. The algorithm alternates between:

1. upper level: rate allocation for fixed placement/assignment,
2. lower level: greedy data placement for fixed rates.

The implementation follows the shared project conventions:
- load instances from JSON via ModelReader
- expose solve_instance() / solve_file()
- keep CLI handling in main()
"""

from __future__ import annotations

import argparse

import numpy as np
import scipy.optimize as opt

try:
    from . import ModelReader
except ImportError:
    import ModelReader


DEFAULT_TAU3 = 0.0
DEFAULT_RATE_EPS = 1e-4
DEFAULT_QUEUE_EPS = 1e-4
DEFAULT_MAX_OUTER_ITER = 10


def route_capacity(route_bits, capacities):
    """Return the tightest link capacity on a route."""
    used_links = [l for l, bit in enumerate(route_bits) if bit]
    if not used_links:
        return 0.0
    return float(min(capacities[l] for l in used_links))


def active_link_loads(instance, x0, xmns, y0, ymn):
    """Compute link loads for the current active flow set."""
    loads = np.zeros(instance.L, dtype=float)
    for n in range(instance.N):
        for s in range(instance.S):
            if x0[n, s]:
                loads += instance.A[0, n, s, :] * y0[n, s]
    for m in range(instance.M):
        for n in range(instance.N):
            for s in range(instance.S):
                if xmns[m, n, s]:
                    loads += instance.A[m + 1, n, s, :] * ymn[m, n]
    return loads


def flow_delay(instance, m, n, s, flow_rate, link_loads, tau3):
    """Evaluate tau_mns for one routed flow under the given link loads."""
    route = instance.A[m, n, s, :]
    push = instance.bn[n] / flow_rate
    queue = sum(
        route[l] * (1.0 / (instance.Cl[l] - link_loads[l]) + tau3)
        for l in range(instance.L)
    )
    return float(push + queue)


def total_delay(instance, x0, xmns, y0, ymn, tau3):
    """Evaluate the paper objective Q(x, y)."""
    loads = active_link_loads(instance, x0, xmns, y0, ymn)
    total = 0.0
    for n in range(instance.N):
        for s in range(instance.S):
            if x0[n, s]:
                total += flow_delay(instance, 0, n, s, y0[n, s], loads, tau3)
    for m in range(instance.M):
        for n in range(instance.N):
            for s in range(instance.S):
                if xmns[m, n, s]:
                    total += flow_delay(instance, m + 1, n, s, ymn[m, n], loads, tau3)
    return float(total)


def build_route_caps(instance):
    """Precompute y_mns,max for all flows."""
    pub_caps = {
        (n, s): route_capacity(instance.A[0, n, s, :], instance.Cl)
        for n in range(instance.N)
        for s in range(instance.S)
    }
    cli_caps = {
        (m, n, s): route_capacity(instance.A[m + 1, n, s, :], instance.Cl)
        for m in range(instance.M)
        for n in range(instance.N)
        for s in range(instance.S)
    }
    return pub_caps, cli_caps


def greedy_assign_from_placement(instance, x0, tau_cost):
    """For each client/object, choose the cheapest open server."""
    xmns = np.zeros((instance.M, instance.N, instance.S), dtype=int)
    for m in range(instance.M):
        for n in range(instance.N):
            open_servers = [s for s in range(instance.S) if x0[n, s]]
            best_s = min(open_servers, key=lambda s: tau_cost[m + 1, n, s])
            xmns[m, n, best_s] = 1
    return xmns


def initial_assignment(instance):
    """Build a simple feasible initial solution."""
    x0 = np.zeros((instance.N, instance.S), dtype=int)
    xmns = np.zeros((instance.M, instance.N, instance.S), dtype=int)

    for n in range(instance.N):
        x0[n, 0] = 1

    for m in range(instance.M):
        for n in range(instance.N):
            xmns[m, n, 0] = 1

    return x0, xmns


def solve_rate_allocation(
    instance,
    x0,
    xmns,
    pub_caps,
    cli_caps,
    tau3=DEFAULT_TAU3,
    rate_eps=DEFAULT_RATE_EPS,
    queue_eps=DEFAULT_QUEUE_EPS,
):
    """Solve the upper-level problem from Section 3.1 for fixed x."""
    pub_pairs = [(n, s) for n in range(instance.N) for s in range(instance.S) if x0[n, s]]
    cli_pairs = [
        (m, n, next(s for s in range(instance.S) if xmns[m, n, s]))
        for m in range(instance.M)
        for n in range(instance.N)
    ]

    n_pub = len(pub_pairs)
    n_cli = len(cli_pairs)

    pub_cap_list = [pub_caps[n, s] for (n, s) in pub_pairs]
    cli_cap_list = [cli_caps[m, n, s] for (m, n, s) in cli_pairs]

    def pub_index(i):
        return i

    def cli_index(i):
        return n_pub + i

    def unpack(y):
        y0 = np.zeros((instance.N, instance.S), dtype=float)
        ymn = np.zeros((instance.M, instance.N), dtype=float)
        for i, (n, s) in enumerate(pub_pairs):
            y0[n, s] = y[pub_index(i)]
        for i, (m, n, _s) in enumerate(cli_pairs):
            ymn[m, n] = y[cli_index(i)]
        return y0, ymn

    def objective(y):
        y0, ymn = unpack(y)
        loads = active_link_loads(instance, x0, xmns, y0, ymn)
        if np.any(loads >= instance.Cl - queue_eps):
            overflow = np.maximum(0.0, loads - (instance.Cl - queue_eps))
            return 1e9 + 1e6 * float(np.sum(overflow))
        return total_delay(instance, x0, xmns, y0, ymn, tau3)

    def link_loads_from_vector(y):
        y0, ymn = unpack(y)
        return active_link_loads(instance, x0, xmns, y0, ymn)

    constraints = [
        {
            "type": "ineq",
            "fun": lambda y, l=l: instance.Cl[l] - queue_eps - link_loads_from_vector(y)[l],
        }
        for l in range(instance.L)
    ]
    bounds = (
        [(rate_eps, cap) for cap in pub_cap_list]
        + [(rate_eps, cap) for cap in cli_cap_list]
    )
    y_upper = np.array([hi for _lo, hi in bounds], dtype=float)
    upper_loads = link_loads_from_vector(y_upper)
    scale = 1.0
    positive_loads = upper_loads > 0.0
    if np.any(positive_loads):
        scale = min(
            1.0,
            float(
                np.min((instance.Cl[positive_loads] - queue_eps) / upper_loads[positive_loads])
            ),
        )
    y_init = np.maximum(
        np.array([lo for lo, _hi in bounds], dtype=float),
        0.9 * scale * y_upper,
    )

    res = opt.minimize(
        objective,
        y_init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 3000},
    )

    y0_out, ymn_out = unpack(res.x)
    return y0_out, ymn_out, float(res.fun), bool(res.success)


def build_tau_cost(instance, x_prev, y0_prev, ymn_prev, pub_caps, cli_caps, tau3):
    """Build the linearized tau-bar costs used by the lower-level problem.

    The previous placement x_prev defines the queueing state. Previously active
    flows use their solved rates; inactive flows use y_max as described in the
    paper.
    """
    x0_prev, xmns_prev = x_prev
    y0_aug = np.zeros((instance.N, instance.S), dtype=float)
    ymn_aug = np.zeros((instance.M, instance.N, instance.S), dtype=float)

    for n in range(instance.N):
        for s in range(instance.S):
            y0_aug[n, s] = y0_prev[n, s] if x0_prev[n, s] else pub_caps[n, s]

    for m in range(instance.M):
        for n in range(instance.N):
            for s in range(instance.S):
                if xmns_prev[m, n, s]:
                    ymn_aug[m, n, s] = ymn_prev[m, n]
                else:
                    ymn_aug[m, n, s] = cli_caps[m, n, s]

    loads = np.zeros(instance.L, dtype=float)
    for n in range(instance.N):
        for s in range(instance.S):
            if x0_prev[n, s]:
                loads += instance.A[0, n, s, :] * y0_aug[n, s]
    for m in range(instance.M):
        for n in range(instance.N):
            for s in range(instance.S):
                if xmns_prev[m, n, s]:
                    loads += instance.A[m + 1, n, s, :] * ymn_aug[m, n, s]

    tau_cost = np.zeros((instance.M + 1, instance.N, instance.S), dtype=float)
    for n in range(instance.N):
        for s in range(instance.S):
            tau_cost[0, n, s] = flow_delay(
                instance,
                0,
                n,
                s,
                max(pub_caps[n, s], DEFAULT_RATE_EPS),
                loads,
                tau3,
            )
    for m in range(instance.M):
        for n in range(instance.N):
            for s in range(instance.S):
                tau_cost[m + 1, n, s] = flow_delay(
                    instance,
                    m + 1,
                    n,
                    s,
                    max(ymn_aug[m, n, s], DEFAULT_RATE_EPS),
                    loads,
                    tau3,
                )
    return tau_cost


def solve_data_placement(instance, tau_cost):
    """Greedy lower-level solver inspired by the facility-location step.

    For each object, repeatedly open the server/cluster with the best
    average cost (publisher placement cost + client connection costs).
    """
    x0 = np.zeros((instance.N, instance.S), dtype=int)
    xmns = np.zeros((instance.M, instance.N, instance.S), dtype=int)

    for n in range(instance.N):
        unassigned = list(range(instance.M))
        opened = set()

        while unassigned:
            best_ratio = float("inf")
            best_server = None
            best_cluster = None

            for s in range(instance.S):
                sorted_clients = sorted(unassigned, key=lambda m: tau_cost[m + 1, n, s])
                placement_cost = 0.0 if s in opened else tau_cost[0, n, s]

                for k in range(1, len(sorted_clients) + 1):
                    cluster = sorted_clients[:k]
                    ratio = (
                        placement_cost
                        + sum(tau_cost[m + 1, n, s] for m in cluster)
                    ) / k
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_server = s
                        best_cluster = cluster

            x0[n, best_server] = 1
            opened.add(best_server)
            for m in best_cluster:
                xmns[m, n, best_server] = 1
                unassigned.remove(m)

    return x0, xmns


def solve_instance(
    instance,
    tau3=DEFAULT_TAU3,
    rate_eps=DEFAULT_RATE_EPS,
    queue_eps=DEFAULT_QUEUE_EPS,
    max_outer_iter=DEFAULT_MAX_OUTER_ITER,
    verbose=False,
):
    """Run the centralized two-level heuristic from Section 3."""
    pub_caps, cli_caps = build_route_caps(instance)
    x0, xmns = initial_assignment(instance)

    best_q = float("inf")
    best_solution = None
    history = []
    seen_states = set()

    for iteration in range(1, max_outer_iter + 1):
        state_key = (tuple(x0.reshape(-1).tolist()), tuple(xmns.reshape(-1).tolist()))
        if state_key in seen_states:
            break
        seen_states.add(state_key)

        y0, ymn, q_val, converged = solve_rate_allocation(
            instance,
            x0,
            xmns,
            pub_caps,
            cli_caps,
            tau3=tau3,
            rate_eps=rate_eps,
            queue_eps=queue_eps,
        )

        history.append(
            {
                "iteration": iteration,
                "objective": q_val,
                "rate_converged": converged,
            }
        )

        if verbose and not converged:
            print(f"[Iteration {iteration}] upper-level solver may not have converged.")

        if q_val < best_q:
            best_q = q_val
            best_solution = (x0.copy(), xmns.copy(), y0.copy(), ymn.copy())

        tau_cost = build_tau_cost(
            instance,
            (x0, xmns),
            y0,
            ymn,
            pub_caps,
            cli_caps,
            tau3,
        )
        x0_new, xmns_new = solve_data_placement(instance, tau_cost)

        if np.array_equal(x0_new, x0) and np.array_equal(xmns_new, xmns):
            break

        x0, xmns = x0_new, xmns_new

    if best_solution is None:
        raise RuntimeError("Heuristic failed to produce a solution.")

    x0_best, xmns_best, y0_best, ymn_best = best_solution
    loads = active_link_loads(instance, x0_best, xmns_best, y0_best, ymn_best)

    placement = {}
    publisher_rates = {}
    assignments = {}
    client_rates = {}
    aggregated_rates = {"pub": {}, "clients": {}}
    link_stats = {}

    for n in range(instance.N):
        placed = [s for s in range(instance.S) if x0_best[n, s]]
        placement[n] = placed
        publisher_rates[n] = {s: float(y0_best[n, s]) for s in placed}
        aggregated_rates["pub"][n] = float(sum(y0_best[n, s] for s in placed))

    for m in range(instance.M):
        aggregated_rates["clients"][m] = {}
        for n in range(instance.N):
            chosen = next(s for s in range(instance.S) if xmns_best[m, n, s])
            assignments[m, n] = chosen
            client_rates[m, n] = float(ymn_best[m, n])
            aggregated_rates["clients"][m][n] = float(ymn_best[m, n])

    for l in range(instance.L):
        link_stats[l] = {
            "load": float(loads[l]),
            "capacity": float(instance.Cl[l]),
            "utilization": float(loads[l] / instance.Cl[l]),
            "queue_delay": float(1.0 / (instance.Cl[l] - loads[l])),
            "tau3": float(tau3),
        }

    return {
        "objective_delay": float(best_q),
        "placement": placement,
        "publisher_rates": publisher_rates,
        "assignments": assignments,
        "client_rates": client_rates,
        "aggregated_rates": aggregated_rates,
        "link_stats": link_stats,
        "iteration_history": history,
        "instance_shape": {
            "M": instance.M,
            "N": instance.N,
            "S": instance.S,
            "L": instance.L,
        },
        "bn": instance.bn.tolist(),
        "Cl": instance.Cl.tolist(),
        "tau3": float(tau3),
    }


def solve_file(
    instance_path,
    tau3=DEFAULT_TAU3,
    rate_eps=DEFAULT_RATE_EPS,
    queue_eps=DEFAULT_QUEUE_EPS,
    max_outer_iter=DEFAULT_MAX_OUTER_ITER,
    verbose=False,
):
    """Load a JSON instance and run the heuristic."""
    instance = ModelReader.load_input_data(instance_path)
    return solve_instance(
        instance,
        tau3=tau3,
        rate_eps=rate_eps,
        queue_eps=queue_eps,
        max_outer_iter=max_outer_iter,
        verbose=verbose,
    )


def print_solution_report(result):
    """Pretty-print the numeric result dictionary returned by solve_file()."""
    shape = result["instance_shape"]
    print("=== Instance ===")
    print(f"  M={shape['M']}  N={shape['N']}  S={shape['S']}  L={shape['L']}")
    print(f"  b_n     = {result['bn']}")
    print(f"  C_l     = {result['Cl']}")
    print(f"  tau3    = {result['tau3']}")

    print("\n" + "=" * 64)
    print(f"  FINAL HEURISTIC DELAY Q = {result['objective_delay']:.5f}")
    print("=" * 64)

    print("\n-- Placement --")
    for n, placed in result["placement"].items():
        print(f"  object {n} on server(s) {placed}")
        for s, rate in result["publisher_rates"][n].items():
            print(f"     upload rate y0[{n},{s}] = {rate:.4f}")

    print("\n-- Client assignments --")
    for (m, n), s in result["assignments"].items():
        rate = result["client_rates"][m, n]
        print(f"  client {m} <- server {s} for object {n} rate {rate:.4f}")

    print("\n-- Aggregated transmission rates y_mn --")
    print(f"  {'m':>4}  {'n':>3}  {'y_mn':>8}")
    for n, rate in result["aggregated_rates"]["pub"].items():
        print(f"  {'pub':>4}  {n:>3}  {rate:>8.4f}")
    for m, by_object in result["aggregated_rates"]["clients"].items():
        for n, rate in by_object.items():
            print(f"  {m:>4}  {n:>3}  {rate:>8.4f}")

    print("\n-- Links --")
    print(f"  {'l':>2} {'load':>8} {'cap':>6} {'util':>7} {'queue':>10} {'tau3':>8}")
    for l, stats in result["link_stats"].items():
        print(
            f"  {l:>2} {stats['load']:>8.3f} {stats['capacity']:>6.1f} "
            f"{100 * stats['utilization']:>6.1f}% {stats['queue_delay']:>10.5f} {stats['tau3']:>8.4f}"
        )

    print("\n-- Iterations --")
    for item in result["iteration_history"]:
        print(
            f"  iter {item['iteration']:>2}: objective={item['objective']:.5f}, "
            f"rate_ok={item['rate_converged']}"
        )


def build_arg_parser():
    """Create the CLI parser used by main()."""
    parser = argparse.ArgumentParser(
        description="Run the Section-3 CAN heuristic for a JSON instance file.",
        epilog="Example: uv run python projekt/can_heuristic.py projekt/data/data1.json",
    )
    parser.add_argument("instance_path", help="Path to the input JSON instance file.")
    parser.add_argument(
        "--tau3",
        type=float,
        default=DEFAULT_TAU3,
        help=f"Constant per-link delay term. Default: {DEFAULT_TAU3}.",
    )
    parser.add_argument(
        "--rate-eps",
        type=float,
        default=DEFAULT_RATE_EPS,
        help=f"Minimum active-flow rate. Default: {DEFAULT_RATE_EPS}.",
    )
    parser.add_argument(
        "--queue-eps",
        type=float,
        default=DEFAULT_QUEUE_EPS,
        help=f"Minimum residual link capacity. Default: {DEFAULT_QUEUE_EPS}.",
    )
    parser.add_argument(
        "--max-outer-iter",
        type=int,
        default=DEFAULT_MAX_OUTER_ITER,
        help=f"Maximum number of outer heuristic iterations. Default: {DEFAULT_MAX_OUTER_ITER}.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra convergence messages while running.",
    )
    return parser


def main(argv=None):
    """CLI entrypoint: read an instance path, solve, and print the report."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = solve_file(
        args.instance_path,
        tau3=args.tau3,
        rate_eps=args.rate_eps,
        queue_eps=args.queue_eps,
        max_outer_iter=args.max_outer_iter,
        verbose=args.verbose,
    )
    print_solution_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
