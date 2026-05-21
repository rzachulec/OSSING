"""
CAN mean-delay optimization solved as a MILP with piecewise-linear
approximations of the nonlinear delay terms.

The nonlinear model contains singular terms 1 / y and 1 / (C - load), so this
implementation uses two small numerical guards:
    rate_eps  : minimum positive rate for an active flow
    queue_eps : minimum residual capacity on each link
"""

from __future__ import annotations

import argparse

import numpy as np
from docplex.mp.model import Model

try:
    from . import ModelReader
except ImportError:
    import ModelReader


DEFAULT_TAU3 = 0.0
DEFAULT_RATE_EPS = 1e-4
DEFAULT_QUEUE_EPS = 1e-4


def route_capacity(route_bits, capacities):
    """Return the tightest link capacity on a route.

    Used during setup to derive per-flow upper bounds for rate variables.
    """
    used_links = [l for l, bit in enumerate(route_bits) if bit]
    if not used_links:
        return 0.0
    return float(min(capacities[l] for l in used_links))


def pwl_inverse(mdl, lo, hi, n_bps=18, include_zero=False):
    """Build a piecewise-linear approximation of 1 / y.

    Used in the objective to approximate the push-delay term.
    """
    bps = np.logspace(np.log10(lo), np.log10(hi), n_bps)
    vals = 1.0 / bps
    if include_zero:
        bps = np.concatenate([[0.0], bps])
        vals = np.concatenate([[0.0], vals])
    postslope = (vals[-1] - vals[-2]) / (bps[-1] - bps[-2])
    return mdl.piecewise(0.0, list(zip(bps.tolist(), vals.tolist())), float(postslope))


def pwl_queue(mdl, capacity, eps, n_bps=20):
    """Build a piecewise-linear approximation of 1 / (capacity - load).

    Used in the objective to approximate the queueing-delay term on each link.
    """
    residual = np.geomspace(capacity, eps, n_bps)
    bps = np.sort(np.unique(capacity - residual))
    vals = 1.0 / (capacity - bps)
    postslope = (vals[-1] - vals[-2]) / (bps[-1] - bps[-2])
    return mdl.piecewise(0.0, list(zip(bps.tolist(), vals.tolist())), float(postslope))


def binary_times_continuous(mdl, x_bin, cont_var, ub, name=None):
    """Linearize q = x_bin * cont_var for a nonnegative continuous variable.

    Used in the objective for:
    - active-flow push-delay terms
    - active-link queue-delay terms
    """
    q = mdl.continuous_var(lb=0.0, ub=ub, name=name)
    mdl.add_constraint(q <= ub * x_bin)
    mdl.add_constraint(q <= cont_var)
    mdl.add_constraint(q >= cont_var - ub * (1 - x_bin))
    return q


def link_term(mdl, x_bin, delay_var, delay_ub, name=None):
    """Linearize q = x_bin * delay_var for one routed link.

    Used when building the queue-delay contribution of each active flow/link
    pair in the objective.
    """
    return binary_times_continuous(mdl, x_bin, delay_var, delay_ub, name=name)


def build_model(
    instance, tau3=DEFAULT_TAU3, rate_eps=DEFAULT_RATE_EPS, queue_eps=DEFAULT_QUEUE_EPS
):
    """Create the DOcplex model and the variable references needed later."""
    mdl = Model(name="CAN_MeanDelay_paper")

    objects = range(instance.N)
    clients = range(instance.M)
    servers = range(instance.S)
    links = range(instance.L)

    pub_route_cap = {
        (n, s): route_capacity(instance.A[0, n, s, :], instance.Cl)
        for n in objects
        for s in servers
    }
    cli_route_cap = {
        (m, n, s): route_capacity(instance.A[m + 1, n, s, :], instance.Cl)
        for m in clients
        for n in objects
        for s in servers
    }

    x_0ns = mdl.binary_var_dict(
        ((n, s) for n in objects for s in servers),
        name=lambda ns: f"x0_n{ns[0]}_s{ns[1]}",
    )
    x_mns = mdl.binary_var_dict(
        ((m, n, s) for m in clients for n in objects for s in servers),
        name=lambda mns: f"xc_m{mns[0]}_n{mns[1]}_s{mns[2]}",
    )
    y_0ns = mdl.continuous_var_dict(
        ((n, s) for n in objects for s in servers),
        lb=0.0,
        ub={(n, s): pub_route_cap[n, s] for n in objects for s in servers},
        name=lambda ns: f"y0_n{ns[0]}_s{ns[1]}",
    )
    y_mns = mdl.continuous_var_dict(
        ((m, n, s) for m in clients for n in objects for s in servers),
        lb=0.0,
        ub={
            (m, n, s): cli_route_cap[m, n, s]
            for m in clients
            for n in objects
            for s in servers
        },
        name=lambda mns: f"yc_m{mns[0]}_n{mns[1]}_s{mns[2]}",
    )
    r_0ns = mdl.continuous_var_dict(
        ((n, s) for n in objects for s in servers),
        lb={
            (n, s): rate_eps if pub_route_cap[n, s] > rate_eps else 0.0
            for n in objects
            for s in servers
        },
        ub={(n, s): pub_route_cap[n, s] for n in objects for s in servers},
        name=lambda ns: f"r0_n{ns[0]}_s{ns[1]}",
    )
    r_mns = mdl.continuous_var_dict(
        ((m, n, s) for m in clients for n in objects for s in servers),
        lb={
            (m, n, s): rate_eps if cli_route_cap[m, n, s] > rate_eps else 0.0
            for m in clients
            for n in objects
            for s in servers
        },
        ub={
            (m, n, s): cli_route_cap[m, n, s]
            for m in clients
            for n in objects
            for s in servers
        },
        name=lambda mns: f"rc_m{mns[0]}_n{mns[1]}_s{mns[2]}",
    )
    load = mdl.continuous_var_dict(
        links,
        lb=0.0,
        ub=[instance.Cl[l] - queue_eps for l in links],
        name=lambda l: f"load_l{l}",
    )
    delay = mdl.continuous_var_dict(
        links,
        lb=0.0,
        name=lambda l: f"delay_l{l}",
    )

    for m in clients:
        for n in objects:
            mdl.add_constraint(
                mdl.sum(x_mns[m, n, s] for s in servers) == 1,
                ctname=f"one_server_m{m}_n{n}",
            )

    for m in clients:
        for n in objects:
            for s in servers:
                mdl.add_constraint(
                    x_mns[m, n, s] <= x_0ns[n, s],
                    ctname=f"feasible_m{m}_n{n}_s{s}",
                )

    for n in objects:
        for s in servers:
            mdl.add_constraint(
                y_0ns[n, s] >= rate_eps * x_0ns[n, s],
                ctname=f"pub_lb_n{n}_s{s}",
            )
            mdl.add_constraint(
                y_0ns[n, s] <= pub_route_cap[n, s] * x_0ns[n, s],
                ctname=f"pub_ub_n{n}_s{s}",
            )
            mdl.add_constraint(
                y_0ns[n, s] <= r_0ns[n, s], ctname=f"pub_match1_n{n}_s{s}"
            )
            mdl.add_constraint(
                y_0ns[n, s] >= r_0ns[n, s] - pub_route_cap[n, s] * (1 - x_0ns[n, s]),
                ctname=f"pub_match2_n{n}_s{s}",
            )

    for m in clients:
        for n in objects:
            for s in servers:
                mdl.add_constraint(
                    y_mns[m, n, s] >= rate_eps * x_mns[m, n, s],
                    ctname=f"cli_lb_m{m}_n{n}_s{s}",
                )
                mdl.add_constraint(
                    y_mns[m, n, s] <= cli_route_cap[m, n, s] * x_mns[m, n, s],
                    ctname=f"cli_ub_m{m}_n{n}_s{s}",
                )
                mdl.add_constraint(
                    y_mns[m, n, s] <= r_mns[m, n, s],
                    ctname=f"cli_match1_m{m}_n{n}_s{s}",
                )
                mdl.add_constraint(
                    y_mns[m, n, s]
                    >= r_mns[m, n, s] - cli_route_cap[m, n, s] * (1 - x_mns[m, n, s]),
                    ctname=f"cli_match2_m{m}_n{n}_s{s}",
                )

    for l in links:
        mdl.add_constraint(
            load[l]
            == mdl.sum(
                instance.A[0, n, s, l] * y_0ns[n, s] for n in objects for s in servers
            )
            + mdl.sum(
                instance.A[m + 1, n, s, l] * y_mns[m, n, s]
                for m in clients
                for n in objects
                for s in servers
            ),
            ctname=f"load_def_l{l}",
        )

    for l in links:
        pwl_q = pwl_queue(mdl, instance.Cl[l], eps=queue_eps)
        mdl.add_constraint(delay[l] == pwl_q(load[l]), ctname=f"queue_pwl_l{l}")

    max_route_cap = max(
        [cap for cap in pub_route_cap.values() if cap > 0.0]
        + [cap for cap in cli_route_cap.values() if cap > 0.0]
    )
    pwl_inv = pwl_inverse(mdl, rate_eps, max_route_cap, n_bps=18, include_zero=False)
    push_inv_max = 1.0 / rate_eps
    delay_max = [1.0 / queue_eps for _ in links]

    push_terms = []
    for n in objects:
        for s in servers:
            inv_term = mdl.continuous_var(
                lb=0.0, ub=push_inv_max, name=f"inv0_n{n}_s{s}"
            )
            mdl.add_constraint(
                inv_term == pwl_inv(r_0ns[n, s]),
                ctname=f"inv_pub_n{n}_s{s}",
            )
            push_terms.append(
                instance.bn[n]
                * binary_times_continuous(
                    mdl,
                    x_0ns[n, s],
                    inv_term,
                    push_inv_max,
                    name=f"push_pub_n{n}_s{s}",
                )
            )

    for m in clients:
        for n in objects:
            for s in servers:
                inv_term = mdl.continuous_var(
                    lb=0.0,
                    ub=push_inv_max,
                    name=f"invc_m{m}_n{n}_s{s}",
                )
                mdl.add_constraint(
                    inv_term == pwl_inv(r_mns[m, n, s]),
                    ctname=f"inv_cli_m{m}_n{n}_s{s}",
                )
                push_terms.append(
                    instance.bn[n]
                    * binary_times_continuous(
                        mdl,
                        x_mns[m, n, s],
                        inv_term,
                        push_inv_max,
                        name=f"push_cli_m{m}_n{n}_s{s}",
                    )
                )

    queue_terms = []
    for n in objects:
        for s in servers:
            for l in links:
                if instance.A[0, n, s, l]:
                    queue_terms.append(
                        link_term(
                            mdl,
                            x_0ns[n, s],
                            delay[l],
                            delay_max[l],
                            name=f"queue_pub_n{n}_s{s}_l{l}",
                        )
                    )

    for m in clients:
        for n in objects:
            for s in servers:
                for l in links:
                    if instance.A[m + 1, n, s, l]:
                        queue_terms.append(
                            link_term(
                                mdl,
                                x_mns[m, n, s],
                                delay[l],
                                delay_max[l],
                                name=f"queue_cli_m{m}_n{n}_s{s}_l{l}",
                            )
                        )

    push_delay = mdl.sum(push_terms)
    queue_delay = mdl.sum(queue_terms)
    const_delay = mdl.sum(
        tau3 * instance.A[0, n, s, l] * x_0ns[n, s]
        for n in objects
        for s in servers
        for l in links
    ) + mdl.sum(
        tau3 * instance.A[m + 1, n, s, l] * x_mns[m, n, s]
        for m in clients
        for n in objects
        for s in servers
        for l in links
    )

    mdl.minimize(push_delay + queue_delay + const_delay)

    return {
        "mdl": mdl,
        "objects": objects,
        "clients": clients,
        "servers": servers,
        "links": links,
        "x_0ns": x_0ns,
        "x_mns": x_mns,
        "y_0ns": y_0ns,
        "y_mns": y_mns,
        "load": load,
        "delay": delay,
        "pub_route_cap": pub_route_cap,
        "cli_route_cap": cli_route_cap,
        "tau3": tau3,
        "rate_eps": rate_eps,
        "queue_eps": queue_eps,
    }


def extract_solution(instance, refs):
    """Convert the solved model into plain numeric Python data."""
    mdl = refs["mdl"]
    objects = refs["objects"]
    clients = refs["clients"]
    servers = refs["servers"]
    links = refs["links"]
    x_0ns = refs["x_0ns"]
    x_mns = refs["x_mns"]
    y_0ns = refs["y_0ns"]
    y_mns = refs["y_mns"]
    load = refs["load"]
    delay = refs["delay"]
    tau3 = refs["tau3"]

    placement = {}
    publisher_rates = {}
    assignments = {}
    client_rates = {}
    aggregated_rates = {"pub": {}, "clients": {}}
    link_stats = {}

    for n in objects:
        placed = [s for s in servers if x_0ns[n, s].solution_value > 0.5]
        placement[n] = placed
        publisher_rates[n] = {s: y_0ns[n, s].solution_value for s in placed}
        aggregated_rates["pub"][n] = sum(y_0ns[n, s].solution_value for s in placed)

    for m in clients:
        aggregated_rates["clients"][m] = {}
        for n in objects:
            chosen = next(s for s in servers if x_mns[m, n, s].solution_value > 0.5)
            assignments[m, n] = chosen
            client_rates[m, n] = y_mns[m, n, chosen].solution_value
            aggregated_rates["clients"][m][n] = client_rates[m, n]

    for l in links:
        ld = load[l].solution_value
        dl = delay[l].solution_value
        link_stats[l] = {
            "load": ld,
            "capacity": float(instance.Cl[l]),
            "utilization": ld / float(instance.Cl[l]),
            "queue_delay": dl,
            "tau3": tau3,
        }

    active_pub = [
        (n, s) for n in objects for s in servers if x_0ns[n, s].solution_value > 0.5
    ]
    active_cli = [
        (m, n, s)
        for m in clients
        for n in objects
        for s in servers
        if x_mns[m, n, s].solution_value > 0.5
    ]

    exact_push = sum(
        instance.bn[n] / y_0ns[n, s].solution_value for (n, s) in active_pub
    ) + sum(instance.bn[n] / y_mns[m, n, s].solution_value for (m, n, s) in active_cli)

    exact_queue = 0.0
    exact_const = 0.0
    for l in links:
        c_minus_load = instance.Cl[l] - load[l].solution_value
        q_l = 1.0 / c_minus_load if c_minus_load > 1e-12 else float("inf")
        flows_l = sum(instance.A[0, n, s, l] for (n, s) in active_pub)
        flows_l += sum(instance.A[m + 1, n, s, l] for (m, n, s) in active_cli)
        exact_queue += flows_l * q_l
        exact_const += flows_l * tau3

    q_pwl = mdl.objective_value
    q_exact = exact_push + exact_queue + exact_const

    return {
        "objective_pwl": q_pwl,
        "objective_exact": q_exact,
        "pwl_error": abs(q_pwl - q_exact),
        "exact_push": exact_push,
        "exact_queue": exact_queue,
        "exact_const": exact_const,
        "placement": placement,
        "publisher_rates": publisher_rates,
        "assignments": assignments,
        "client_rates": client_rates,
        "aggregated_rates": aggregated_rates,
        "link_stats": link_stats,
        "solve_status": str(mdl.solve_status),
        "model_name": mdl.name,
        "instance_shape": {
            "M": instance.M,
            "N": instance.N,
            "S": instance.S,
            "L": instance.L,
        },
        "bn": instance.bn.tolist(),
        "Cl": instance.Cl.tolist(),
        "tau3": tau3,
    }


def solve_instance(
    instance,
    tau3=DEFAULT_TAU3,
    rate_eps=DEFAULT_RATE_EPS,
    queue_eps=DEFAULT_QUEUE_EPS,
    log_output=False,
):
    """Solve an already loaded instance and return plain numeric results."""
    refs = build_model(instance, tau3=tau3, rate_eps=rate_eps, queue_eps=queue_eps)
    sol = refs["mdl"].solve(log_output=log_output)
    if not sol:
        raise RuntimeError(f"No solution found. Status: {refs['mdl'].solve_status}")
    return extract_solution(instance, refs)


def solve_file(
    instance_path,
    tau3=DEFAULT_TAU3,
    rate_eps=DEFAULT_RATE_EPS,
    queue_eps=DEFAULT_QUEUE_EPS,
    log_output=False,
):
    """Load an instance from JSON, solve it, and return plain numeric results."""
    instance = ModelReader.load_input_data(instance_path)
    refs = build_model(instance, tau3=tau3, rate_eps=rate_eps, queue_eps=queue_eps)
    sol = refs["mdl"].solve(log_output=log_output)
    if not sol:
        raise RuntimeError(f"No solution found. Status: {refs['mdl'].solve_status}")
    return extract_solution(instance, refs)


def print_solution_report(result):
    """Pretty-print the numeric result dictionary returned by solve_file()."""
    shape = result["instance_shape"]
    print("=== Instance ===")
    print(f"  M={shape['M']}  N={shape['N']}  S={shape['S']}  L={shape['L']}")
    print(f"  b_n     = {result['bn']}")
    print(f"  C_l     = {result['Cl']}")
    print(f"  tau3    = {result['tau3']}")

    print("\n" + "=" * 64)
    print(f"  OPTIMAL total mean delay (PWL approx) Q = {result['objective_pwl']:.5f}")
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

    print("\n-- Delay decomposition (exact, non-PWL) --")
    print(f"  push   sum b/y          = {result['exact_push']:.5f}")
    print(f"  queue  sum f/(C-load)   = {result['exact_queue']:.5f}")
    print(f"  const  sum f*tau3       = {result['exact_const']:.5f}")
    print(f"  total Q (exact)         = {result['objective_exact']:.5f}")
    print(f"  total Q (CPLEX PWL)     = {result['objective_pwl']:.5f}")
    print(f"  PWL error               = {result['pwl_error']:.5f}")


def build_arg_parser():
    """Create the CLI parser used by main()."""
    parser = argparse.ArgumentParser(
        description="Solve the CAN mean-delay model for a JSON instance file.",
        epilog="Example: uv run python projekt/can_cplex.py projekt/data/data1.json",
    )
    parser.add_argument(
        "instance_path",
        help="Path to the input JSON instance file.",
    )
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
        help=f"Minimum positive active-flow rate used by the PWL approximation. Default: {DEFAULT_RATE_EPS}.",
    )
    parser.add_argument(
        "--queue-eps",
        type=float,
        default=DEFAULT_QUEUE_EPS,
        help=f"Minimum residual link capacity used by the PWL approximation. Default: {DEFAULT_QUEUE_EPS}.",
    )
    parser.add_argument(
        "--log-output",
        action="store_true",
        help="Show the raw CPLEX solve log.",
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
        log_output=args.log_output,
    )
    print_solution_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
