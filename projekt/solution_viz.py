"""
solution_viz.py — render a solved CAN instance on the topology map.

Two entry points:

  render(...)         low-level: takes solved docplex variable dicts (or plain
                      {key: value} dicts), e.g. from a can_cplex.py run.
  render_result(...)  takes the JSON-style result dict returned by
                      can_heuristic.solve_instance / solve_file plus the
                      routing tensor `a` (instance.A).
  visualize_file(...) drop-module entry: load a JSON instance, solve it with
                      the heuristic, and render — mirrors routing_viz.py.

Each produces a figure with

  * topology laid out exactly like routing_viz's schematic, but with every
    link coloured by utilisation (RdYlGn_r) and thickened in proportion;
  * server nodes badged with the objects placed on them;
  * a side panel listing active uploads / downloads with rates and the
    resulting pushing delay  b_n / y;
  * the objective Q in the figure title.

The module only depends on the routing tensor `a` and capacity vector `Cl`
from network.py — no coupling to the CPLEX model itself.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import Normalize

try:
    from . import ModelReader
    from .routing_viz import _infer_link_role
    from .can_heuristic import solve_instance
except ImportError:
    import ModelReader
    from routing_viz import _infer_link_role
    from can_heuristic import solve_instance


# ---------------------------------------------------------------------------
# value extraction helpers — accept docplex vars or plain floats
# ---------------------------------------------------------------------------

def _val(v):
    return v.solution_value if hasattr(v, "solution_value") else float(v)


def _vals_dict(d):
    return {k: _val(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def render(
    x_0ns, x_mns,
    y_0ns, y_mns,
    load, delay,
    a: np.ndarray,
    Cl: np.ndarray,
    bn: np.ndarray | None = None,
    *,
    Q: float | None = None,
    save_path: str | None = None,
    show: bool = True,
):
    x0v = _vals_dict(x_0ns)
    xcv = _vals_dict(x_mns)
    y0v = _vals_dict(y_0ns)
    ycv = _vals_dict(y_mns)
    ldv = _vals_dict(load)
    dlv = _vals_dict(delay)

    M_plus_1, N, S, _ = a.shape
    M = M_plus_1 - 1

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.0])

    ax_topo = fig.add_subplot(gs[0])
    _draw_topology(ax_topo, a, Cl, ldv, dlv, x0v)

    ax_tbl = fig.add_subplot(gs[1])
    _draw_flow_table(ax_tbl, M, N, S, x0v, xcv, y0v, ycv, bn)

    title = "Solved CAN instance"
    if Q is not None:
        title += f"   Q = {Q:.4f}"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# topology with solution overlay
# ---------------------------------------------------------------------------

def _box(ax, xy, text, color, h=0.55):
    ax.add_patch(FancyBboxPatch((xy[0] - 0.42, xy[1] - h / 2), 0.84, h,
                                boxstyle="round,pad=0.02",
                                fc=color, ec="black", zorder=3))
    ax.text(*xy, text, ha="center", va="center", zorder=4, fontsize=8)


def _link_scale(Cl, ldv, L):
    """Shared colour/width scale for links by utilisation = load / capacity."""
    cmap = plt.get_cmap("RdYlGn_r")
    utils = np.array([ldv[l] / Cl[l] for l in range(L)])
    max_util = max(utils.max(), 1e-6)
    norm = Normalize(vmin=0.0, vmax=max(max_util, 0.01))
    return cmap, norm, utils, max_util


def _draw_topology(ax, a, Cl, ldv, dlv, x0v):
    roles = _infer_link_role(a)
    M_plus_1, N, S, L = a.shape

    # -- infer routers as unique (server-set) fingerprints from L_cli links --
    fingerprints: list[tuple[frozenset, int]] = []
    for l, role in enumerate(roles):
        if role["kind"] == "user_router_access" and role["user"] == 0:
            fingerprints.append((frozenset(role["servers"]), l))
    routers: list[tuple[frozenset, int]] = []
    for fp, l in fingerprints:
        if fp not in [r[0] for r in routers]:
            routers.append((fp, l))
    R = len(routers)

    # The router layout needs at least one inferred router, and a single router
    # can't host a backbone (it would collapse to a zero-length edge).  Anything
    # else (e.g. a shared-core topology like data1.json) falls back to a
    # bipartite user<->server view so no links go missing.
    backbone_present = any(role["kind"] == "backbone" for role in roles)
    if R == 0 or (R < 2 and backbone_present):
        _draw_bipartite_solution(ax, a, Cl, ldv, dlv, x0v)
        return

    span = max(M_plus_1, S, R)
    ax.set_xlim(-0.8, 5.8)
    ax.set_ylim(-span, 1)
    ax.set_axis_off()

    user_x, router_x, srv_x = 0.0, 2.5, 5.0
    user_y   = {u: -u for u in range(M_plus_1)}
    router_y = {r: -(span - 1) * r / max(R - 1, 1) for r in range(R)}
    srv_y    = {s: -s for s in range(S)}

    server_to_router: dict[int, int] = {}
    for r_idx, (fp, _) in enumerate(routers):
        for s in fp:
            server_to_router[s] = r_idx

    # -- nodes --
    for u in range(M_plus_1):
        tag = "publisher" if u == 0 else f"client {u - 1}"
        _box(ax, (user_x, user_y[u]), tag, "#cfe2ff")
    for r in range(R):
        _box(ax, (router_x, router_y[r]), f"R{r}", "#e0e0e0")
    for s in range(S):
        placed = [n for n in range(N) if x0v.get((n, s), 0) > 0.5]
        label = f"server {s}"
        if placed:
            label += f"\nn={','.join(map(str, placed))}"
        _box(ax, (srv_x, srv_y[s]), label, "#ffd8b5",
             h=0.75 if placed else 0.55)

    # -- link colour scale: utilisation u = load / C, capped at 1 --
    cmap, norm, utils, max_util = _link_scale(Cl, ldv, L)

    def _edge(xy1, xy2, l):
        util = utils[l]
        colour = cmap(norm(util))
        lw = 1.0 + 6.0 * util / max_util
        ax.plot([xy1[0], xy2[0]], [xy1[1], xy2[1]],
                color=colour, lw=lw, zorder=1)
        mx, my = (xy1[0] + xy2[0]) / 2, (xy1[1] + xy2[1]) / 2
        cap_str = "∞" if Cl[l] >= 1e5 else f"{Cl[l]:g}"
        label = f"L{l}  {ldv[l]:.2f}/{cap_str}"
        if dlv[l] > 1e-4:
            label += f"\nq={dlv[l]:.3f}"
        ax.text(mx, my, label, ha="center", va="center", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.18",
                          fc="white", ec="0.6", alpha=0.9))

    for l, role in enumerate(roles):
        kind = role["kind"]
        if kind == "user_router_access":
            u = role["user"]
            fp = frozenset(role["servers"])
            r_idx = next(r for r, (rfp, _) in enumerate(routers) if rfp == fp)
            _edge((user_x, user_y[u]), (router_x, router_y[r_idx]), l)
        elif kind == "server_access":
            s = role["server"]
            r_idx = server_to_router.get(s, 0)
            _edge((router_x, router_y[r_idx]), (srv_x, srv_y[s]), l)
        elif kind == "backbone":
            y_a, y_b = next(iter(router_y.values())), list(router_y.values())[-1]
            _edge((router_x, y_a), (router_x, y_b), l)
        elif kind == "user_access":
            u = role["user"]
            _edge((user_x, user_y[u]), (router_x, router_y[0]), l)

    ax.set_title("topology — link colour = utilisation, width = load")


def _draw_bipartite_solution(ax, a, Cl, ldv, dlv, x0v):
    """Fallback overlay for topologies the router layout can't represent.

    Draws users on the left and servers on the right with one edge per routed
    user->server pair, coloured/sized by the pair's bottleneck link and labelled
    with every link on that route (id, load/capacity, queue delay).
    """
    M_plus_1, N, S, L = a.shape
    span = max(M_plus_1, S)
    ax.set_xlim(-0.8, 4.8)
    ax.set_ylim(-span, 1)
    ax.set_axis_off()

    user_x, srv_x = 0.0, 4.0
    user_y = {u: -u for u in range(M_plus_1)}
    srv_y = {s: -s for s in range(S)}

    cmap, norm, utils, max_util = _link_scale(Cl, ldv, L)

    for u in range(M_plus_1):
        tag = "publisher" if u == 0 else f"client {u - 1}"
        _box(ax, (user_x, user_y[u]), tag, "#cfe2ff")
    for s in range(S):
        placed = [n for n in range(N) if x0v.get((n, s), 0) > 0.5]
        label = f"server {s}"
        if placed:
            label += f"\nn={','.join(map(str, placed))}"
        _box(ax, (srv_x, srv_y[s]), label, "#ffd8b5",
             h=0.75 if placed else 0.55)

    # routing is ~ n-invariant, so read the route from object 0
    for u in range(M_plus_1):
        for s in range(S):
            links = np.where(a[u, 0, s])[0]
            if len(links) == 0:
                continue
            bottleneck = max(utils[l] for l in links)
            colour = cmap(norm(bottleneck))
            lw = 1.0 + 6.0 * bottleneck / max_util
            ax.plot([user_x, srv_x], [user_y[u], srv_y[s]],
                    color=colour, lw=lw, zorder=1)
            parts = []
            for l in links:
                cap_str = "∞" if Cl[l] >= 1e5 else f"{Cl[l]:g}"
                seg = f"L{l} {ldv[l]:.2f}/{cap_str}"
                if dlv[l] > 1e-4:
                    seg += f" q={dlv[l]:.3f}"
                parts.append(seg)
            mx, my = (user_x + srv_x) / 2, (user_y[u] + srv_y[s]) / 2
            ax.text(mx, my, "\n".join(parts), ha="center", va="center", fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.18",
                              fc="white", ec="0.6", alpha=0.9))

    ax.set_title("topology (bipartite) — link colour = utilisation, width = load")


# ---------------------------------------------------------------------------
# side panel — placement + active uploads/downloads
# ---------------------------------------------------------------------------

def _draw_flow_table(ax, M, N, S, x0v, xcv, y0v, ycv, bn):
    ax.set_axis_off()
    lines: list[str] = ["— Placement —"]
    for n in range(N):
        hosts = [s for s in range(S) if x0v.get((n, s), 0) > 0.5]
        lines.append(f"  obj {n} → server(s) {hosts}")

    lines.append("")
    lines.append("— Publisher uploads (pub → s) —")
    lines.append(f"  {'n':>2} {'s':>2}  {'y':>6}  {'push':>6}")
    for n in range(N):
        for s in range(S):
            if x0v.get((n, s), 0) > 0.5:
                b = float(bn[n]) if bn is not None else 1.0
                push = b / y0v[n, s]
                lines.append(f"  {n:>2} {s:>2}  {y0v[n,s]:>6.3f}  {push:>6.3f}")

    lines.append("")
    lines.append("— Client downloads (c ← s) —")
    lines.append(f"  {'m':>2} {'n':>2} {'s':>2}  {'y':>6}  {'push':>6}")
    for m in range(M):
        for n in range(N):
            for s in range(S):
                if xcv.get((m, n, s), 0) > 0.5:
                    b = float(bn[n]) if bn is not None else 1.0
                    push = b / ycv[m, n, s]
                    lines.append(f"  {m:>2} {n:>2} {s:>2}  "
                                 f"{ycv[m,n,s]:>6.3f}  {push:>6.3f}")

    ax.text(0.02, 0.5, "\n".join(lines),
            ha="left", va="center", family="monospace",
            fontsize=9, transform=ax.transAxes)


# ---------------------------------------------------------------------------
# heuristic result dict adapter
# ---------------------------------------------------------------------------

# serif styling so the figure drops cleanly into a LaTeX/Word report
_REPORT_RC = {
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 9,
    "figure.titlesize": 11,
}


def render_result(
    result: dict,
    a: np.ndarray,
    *,
    name: str | None = None,
    save_path: str | None = None,
    show: bool = True,
):
    """Render the result dict produced by can_heuristic.solve_instance/solve_file.

    The heuristic result is a plain (JSON-serialisable) dict and does not carry
    the routing tensor, so the instance's `a` (= instance.A) must be supplied
    for the topology layout.  Keys are coerced to int so the function also works
    on a result that has been round-tripped through JSON (string keys).

    `name` is shown on the figure (defaults to the save_path basename) so the
    rendered image identifies the instance it came from.
    """
    with plt.rc_context(_REPORT_RC):
        return _render_result(result, a, name=name, save_path=save_path, show=show)


def _render_result(result, a, *, name, save_path, show):
    shape = result["instance_shape"]
    M, N, S = int(shape["M"]), int(shape["N"]), int(shape["S"])
    Cl = np.asarray(result["Cl"], dtype=float)
    bn = np.asarray(result["bn"], dtype=float)

    # rebuild the {key: value} dicts the drawing helpers expect
    x0v: dict[tuple[int, int], float] = {}
    for n, servers in result["placement"].items():
        for s in servers:
            x0v[(int(n), int(s))] = 1.0

    y0v: dict[tuple[int, int], float] = {}
    for n, by_s in result["publisher_rates"].items():
        for s, rate in by_s.items():
            y0v[(int(n), int(s))] = float(rate)

    xcv: dict[tuple[int, int, int], float] = {}
    ycv: dict[tuple[int, int, int], float] = {}
    for (m, n), s in result["assignments"].items():
        xcv[(int(m), int(n), int(s))] = 1.0
        ycv[(int(m), int(n), int(s))] = float(result["client_rates"][(m, n)])

    ldv = {int(l): float(st["load"]) for l, st in result["link_stats"].items()}
    dlv = {int(l): float(st["queue_delay"]) for l, st in result["link_stats"].items()}

    fig = plt.figure(figsize=(9.5, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.0])

    ax_topo = fig.add_subplot(gs[0])
    _draw_topology(ax_topo, a, Cl, ldv, dlv, x0v)

    ax_tbl = fig.add_subplot(gs[1])
    _draw_flow_table(ax_tbl, M, N, S, x0v, xcv, y0v, ycv, bn)

    label = name or (os.path.basename(save_path) if save_path else None)
    q = result.get("objective_delay", result.get("objective_pwl"))
    title = "Solved CAN instance"
    if label:
        title += f"  —  {label}"
    if q is not None:
        title += f"   (Q = {q:.4f})"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# drop-module entry point + CLI (mirrors routing_viz.py)
# ---------------------------------------------------------------------------

def visualize_file(
    instance_path: str,
    *,
    save_path: str | None = None,
    show: bool = True,
    **solver_kwargs,
):
    """Load a JSON instance, solve it with the heuristic, and render the result.

    Extra keyword arguments (tau3, rate_eps, queue_eps, max_outer_iter, verbose)
    are forwarded to can_heuristic.solve_instance.
    """
    instance = ModelReader.load_input_data(instance_path)
    result = solve_instance(instance, **solver_kwargs)
    return render_result(
        result,
        instance.A,
        name=os.path.basename(instance_path),
        save_path=save_path,
        show=show,
    )


def build_arg_parser():
    """Create the CLI parser used by main()."""
    parser = argparse.ArgumentParser(
        description="Solve a JSON instance with the heuristic and render the solution.",
        epilog="Example: uv run python projekt/solution_viz.py projekt/data/data1.json --save solution_data1.png",
    )
    parser.add_argument("instance_path", help="Path to the input JSON instance file.")
    parser.add_argument(
        "--save",
        dest="save_path",
        help="Optional path for the output image file.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the matplotlib window; useful when only saving.",
    )
    parser.add_argument(
        "--tau3",
        type=float,
        default=0.0,
        help="Constant per-link delay term passed to the heuristic. Default: 0.0.",
    )
    parser.add_argument(
        "--max-outer-iter",
        type=int,
        default=10,
        help="Maximum number of outer heuristic iterations. Default: 10.",
    )
    return parser


def main(argv=None):
    """CLI entrypoint: read an instance path, solve, and render the solution."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    visualize_file(
        args.instance_path,
        save_path=args.save_path,
        show=not args.no_show,
        tau3=args.tau3,
        max_outer_iter=args.max_outer_iter,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
