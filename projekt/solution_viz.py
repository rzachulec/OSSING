"""
solution_viz.py — render a solved CAN instance on the topology map.

Called at the end of a can_cplex.py run.  Takes the solved docplex variable
dicts (or plain {key: value} dicts) and produces a figure with

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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

from routing_viz import _infer_link_role


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
    R = max(len(routers), 1)

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
    cmap = get_cmap("RdYlGn_r")
    utils = np.array([ldv[l] / Cl[l] for l in range(L)])
    max_util = max(utils.max(), 1e-6)
    norm = Normalize(vmin=0.0, vmax=max(max_util, 0.01))

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

    ax.text(0.02, 0.98, "\n".join(lines),
            ha="left", va="top", family="monospace",
            fontsize=9, transform=ax.transAxes)
