"""
routing_viz.py — visualize the CAN routing tensor  a[m, n, s, l].

Three complementary views, any of which can be called alone or stitched by
`visualize(a, ...)` into a single figure:

    link_usage_heatmap(a)    rows = individual flows (m,n,s), cols = links.
                             cell = 1 iff the flow traverses that link.
    flow_count_per_link(a)   bar chart — how many active flows share each
                             link (classic way to spot the backbone).
    topology_schematic(a)    attempts to infer a two-router star-of-stars
                             topology from link-usage patterns and draws
                             it.  Falls back to a bipartite layout if the
                             shape doesn't match.

Example:
    from routing_viz import visualize
    visualize(a, Cl=Cl, save_path="routing.png")
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ---------------------------------------------------------------------------
# flow-ID helpers
# ---------------------------------------------------------------------------

def _flow_labels(a: np.ndarray) -> list[str]:
    """Build human-readable labels for every flow (m, n, s)."""
    M_plus_1, N, S, _ = a.shape
    labels = []
    for m in range(M_plus_1):
        tag = "pub" if m == 0 else f"c{m-1}"
        for n in range(N):
            for s in range(S):
                labels.append(f"{tag}|n{n}|s{s}")
    return labels


def _is_n_invariant(a: np.ndarray) -> bool:
    """True if routing is identical for every object n — common case."""
    return all(np.array_equal(a[:, 0], a[:, n]) for n in range(a.shape[1]))


# ---------------------------------------------------------------------------
# view 1: link-usage heatmap
# ---------------------------------------------------------------------------

def link_usage_heatmap(a: np.ndarray, ax=None, *, collapse_n: bool | None = None):
    """Binary heatmap — rows = flows, columns = links.

    If `collapse_n=True` (or None and routing is n-invariant), rows collapse
    to (M+1)·S user-server pairs.  Otherwise rows are all (M+1)·N·S flows.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(1 + 0.6 * a.shape[-1], 0.3 * np.prod(a.shape[:-1]) + 2))

    if collapse_n is None:
        collapse_n = _is_n_invariant(a)

    if collapse_n:
        grid = a[:, 0].reshape(-1, a.shape[-1])
        labels = [f"{'pub' if m == 0 else f'c{m-1}'}→s{s}"
                  for m in range(a.shape[0]) for s in range(a.shape[2])]
    else:
        grid = a.reshape(-1, a.shape[-1])
        labels = _flow_labels(a)

    ax.imshow(grid, cmap="Greys", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels([f"L{l}" for l in range(grid.shape[1])])
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("link")
    ax.set_ylabel("flow" + (" (n-invariant)" if collapse_n else ""))
    ax.set_title("link usage per flow")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j]:
                ax.text(j, i, "1", ha="center", va="center",
                        color="white", fontsize=7)
    ax.set_xticks(np.arange(-.5, grid.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, grid.shape[0]), minor=True)
    ax.grid(which="minor", color="0.85", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    return ax


# ---------------------------------------------------------------------------
# view 2: flow-count per link
# ---------------------------------------------------------------------------

def flow_count_per_link(a: np.ndarray, ax=None, *, Cl: np.ndarray | None = None):
    """Bar chart — number of flows sharing each link.  Overlays capacity
    (as text) if `Cl` is provided so the user can spot saturation risk."""
    if ax is None:
        _, ax = plt.subplots(figsize=(1 + 0.6 * a.shape[-1], 3))

    counts = a.reshape(-1, a.shape[-1]).sum(axis=0)
    bars = ax.bar(range(len(counts)), counts, color="#4a7dc7")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([f"L{l}" for l in range(len(counts))])
    ax.set_ylabel("# flows through link")
    ax.set_title("flow count per link")
    for i, (bar, c) in enumerate(zip(bars, counts)):
        lbl = f"{int(c)}"
        if Cl is not None:
            lbl += f"\nC={Cl[i]:g}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                lbl, ha="center", va="bottom", fontsize=8)
    ax.margins(y=0.2)
    return ax


# ---------------------------------------------------------------------------
# view 3: topology schematic (inferred)
# ---------------------------------------------------------------------------

def _infer_link_role(a: np.ndarray) -> list[dict]:
    """For each link, decide its role from the set of users/servers it
    touches.  Recognises five patterns:

      user_access        one user, all servers   (access to single router)
      server_access      all users, one server   (server's access link)
      backbone           all users, all servers  (inter-router trunk)
      user_router_access one user, subset of servers (access to a specific
                         router that hosts that subset of servers)
      other              anything else — schematic falls back to bipartite
    """
    a_us = a[:, 0]                          # routing is ~ n-invariant
    M_plus_1, S, L = a_us.shape
    roles = []
    for l in range(L):
        users_using = {u for u in range(M_plus_1) if a_us[u, :, l].any()}
        servers_hit = {s for s in range(S) if a_us[:, s, l].any()}
        if len(users_using) == 1 and len(servers_hit) == S:
            roles.append({"kind": "user_access",
                          "user": next(iter(users_using))})
        elif len(servers_hit) == 1 and len(users_using) == M_plus_1:
            roles.append({"kind": "server_access",
                          "server": next(iter(servers_hit))})
        elif len(users_using) == M_plus_1 and len(servers_hit) == S:
            roles.append({"kind": "backbone"})
        elif len(users_using) == 1 and 0 < len(servers_hit) < S:
            roles.append({"kind": "user_router_access",
                          "user": next(iter(users_using)),
                          "servers": servers_hit})
        else:
            roles.append({"kind": "other",
                          "users": users_using, "servers": servers_hit})
    return roles


def _rounded_box(ax, xy, text, color):
    ax.add_patch(FancyBboxPatch((xy[0] - 0.35, xy[1] - 0.25), 0.7, 0.5,
                                boxstyle="round,pad=0.02",
                                fc=color, ec="black", zorder=3))
    ax.text(*xy, text, ha="center", va="center", zorder=4, fontsize=9)


def _labeled_edge(ax, xy1, xy2, label):
    ax.plot([xy1[0], xy2[0]], [xy1[1], xy2[1]],
            color="0.4", lw=1.2, zorder=1)
    mx, my = (xy1[0] + xy2[0]) / 2, (xy1[1] + xy2[1]) / 2
    ax.text(mx, my, label, ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7"))


def _draw_multirouter(ax, a: np.ndarray, roles: list[dict]):
    """Every user→router link is a dedicated access link; every server
    hangs off exactly one router.  Group servers by router via the
    user_router_access fingerprints."""
    a_us = a[:, 0]
    M_plus_1, S, L = a_us.shape

    # A router is defined by the set of servers reachable through it.
    # All users share the same set because access is symmetric, so take
    # the fingerprint from user 0's user_router_access links.
    fingerprints = []
    for l, role in enumerate(roles):
        if role["kind"] == "user_router_access" and role["user"] == 0:
            fingerprints.append((frozenset(role["servers"]), l))
    routers = []          # list of (server_set, representative_link)
    for fp, l in fingerprints:
        if fp not in [r[0] for r in routers]:
            routers.append((fp, l))
    R = len(routers)

    ax.set_axis_off()
    ax.set_title(f"inferred topology — {R} routers, no backbone")
    ax.set_xlim(-0.6, 5.6)
    span = max(M_plus_1, S, R)
    ax.set_ylim(-span, 1)

    user_x, router_x, srv_x = 0.0, 2.5, 5.0
    user_y   = {u: -u for u in range(M_plus_1)}
    router_y = {r: -(span - 1) * r / max(R - 1, 1) for r in range(R)}
    srv_y    = {s: -s for s in range(S)}

    # draw nodes
    for u in range(M_plus_1):
        tag = "publisher" if u == 0 else f"client {u-1}"
        _rounded_box(ax, (user_x, user_y[u]), tag, "#cfe2ff")
    for r in range(R):
        _rounded_box(ax, (router_x, router_y[r]), f"R{r}", "#e0e0e0")
    for s in range(S):
        _rounded_box(ax, (srv_x, srv_y[s]), f"server {s}", "#ffd8b5")

    # map server → router
    server_to_router = {}
    for r_idx, (fp, _) in enumerate(routers):
        for s in fp:
            server_to_router[s] = r_idx

    # draw links
    for l, role in enumerate(roles):
        if role["kind"] == "user_router_access":
            u = role["user"]
            fp = frozenset(role["servers"])
            r_idx = next(r for r, (rfp, _) in enumerate(routers) if rfp == fp)
            _labeled_edge(ax, (user_x, user_y[u]),
                          (router_x, router_y[r_idx]), f"L{l}")
        elif role["kind"] == "server_access":
            s = role["server"]
            r_idx = server_to_router.get(s, 0)
            _labeled_edge(ax, (router_x, router_y[r_idx]),
                          (srv_x, srv_y[s]), f"L{l}")
    return ax


def topology_schematic(a: np.ndarray, ax=None):
    """Draw an inferred star-of-stars topology.  Nodes: publisher + clients
    on the left, two routers in the middle, servers on the right.  If the
    topology isn't recognisable, the function draws a neutral bipartite
    user↔server view annotated by link id."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    a_us = a[:, 0]
    M_plus_1, S, L = a_us.shape
    roles = _infer_link_role(a)

    kinds = [r["kind"] for r in roles]
    matches_2router = (kinds.count("backbone") == 1
                       and kinds.count("user_access") == M_plus_1
                       and kinds.count("server_access") == S)

    # Multi-router "no-backbone": every server has its own router; every
    # user has one access link per router.  Each server-access link
    # corresponds 1-1 to a router.  Servers group by the router they're on
    # — inferred from which L_cli (user_router_access) links reach them.
    user_router_links = [r for r in roles if r["kind"] == "user_router_access"]
    matches_multirouter = (
        kinds.count("server_access") == S
        and kinds.count("backbone") == 0
        and len(user_router_links) > 0
        and all(len(r.get("servers", set())) >= 1 for r in user_router_links)
    )

    if matches_multirouter and not matches_2router:
        return _draw_multirouter(ax, a, roles)

    if not matches_2router:
        # fallback: bipartite user↔server with link lists on each edge
        for u in range(M_plus_1):
            ax.scatter(0, -u, s=300, color="#4a7dc7", zorder=3)
            tag = "publisher" if u == 0 else f"client {u-1}"
            ax.text(-0.1, -u, tag, ha="right", va="center")
        for s in range(S):
            ax.scatter(4, -s, s=300, color="#c77a4a", zorder=3)
            ax.text(4.1, -s, f"server {s}", ha="left", va="center")
        for u in range(M_plus_1):
            for s in range(S):
                links = np.where(a_us[u, s])[0]
                if len(links):
                    ax.plot([0, 4], [-u, -s], color="0.7", lw=1, zorder=1)
                    ax.text(2, (-u - s) / 2,
                            ",".join(f"L{l}" for l in links),
                            ha="center", va="center", fontsize=7,
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc="white", ec="0.7"))
        ax.set_title("routing — bipartite view (topology not inferred)")
        ax.set_axis_off()
        return ax

    # --- inferred 2-router star-of-stars ------------------------------------
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(-max(M_plus_1, S), 1)
    ax.set_axis_off()
    ax.set_title("inferred topology — two-router star-of-stars")

    # node coordinates
    user_x, r1_x, r2_x, srv_x = 0.0, 1.7, 3.3, 5.0
    user_y = {u: -u for u in range(M_plus_1)}
    srv_y  = {s: -s for s in range(S)}
    r_y = -(max(M_plus_1, S) - 1) / 2
    r1_pos, r2_pos = (r1_x, r_y), (r2_x, r_y)

    def _box(xy, text, color):
        ax.add_patch(FancyBboxPatch((xy[0] - 0.35, xy[1] - 0.25), 0.7, 0.5,
                                    boxstyle="round,pad=0.02",
                                    fc=color, ec="black", zorder=3))
        ax.text(*xy, text, ha="center", va="center", zorder=4, fontsize=9)

    for u in range(M_plus_1):
        tag = "publisher" if u == 0 else f"client {u-1}"
        _box((user_x, user_y[u]), tag, "#cfe2ff")
    _box(r1_pos, "R1", "#e0e0e0")
    _box(r2_pos, "R2", "#e0e0e0")
    for s in range(S):
        _box((srv_x, srv_y[s]), f"server {s}", "#ffd8b5")

    # draw links, labelled by L index
    def _draw(xy1, xy2, label, offset=(0, 0)):
        ax.plot([xy1[0], xy2[0]], [xy1[1], xy2[1]], color="0.4", lw=1.2, zorder=1)
        mx, my = (xy1[0] + xy2[0]) / 2 + offset[0], (xy1[1] + xy2[1]) / 2 + offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7"))

    for l, role in enumerate(roles):
        if role["kind"] == "user_access":
            u = role["user"]
            _draw((user_x, user_y[u]), r1_pos, f"L{l}")
        elif role["kind"] == "server_access":
            s = role["server"]
            _draw(r2_pos, (srv_x, srv_y[s]), f"L{l}")
        elif role["kind"] == "backbone":
            _draw(r1_pos, r2_pos, f"L{l}")

    return ax


# ---------------------------------------------------------------------------
# composite
# ---------------------------------------------------------------------------

def visualize(a: np.ndarray, *, Cl: np.ndarray | None = None,
              save_path: str | None = None, show: bool = True):
    """One-shot composite figure: heatmap + bar chart + topology schematic."""
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.4],
                          height_ratios=[2.3, 1.0], hspace=0.35, wspace=0.25)
    link_usage_heatmap(a, ax=fig.add_subplot(gs[0, 0]))
    topology_schematic(a,   ax=fig.add_subplot(gs[0, 1]))
    flow_count_per_link(a, Cl=Cl, ax=fig.add_subplot(gs[1, :]))
    fig.suptitle(f"routing tensor  a[m, n, s, l]  "
                 f"(M+1={a.shape[0]}, N={a.shape[1]}, S={a.shape[2]}, L={a.shape[3]})")
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# demo — run this file directly to see the paper instance
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Uses the shared instance from network.py so the picture always matches
    # whatever the solver is currently running on.
    from network import a, Cl
    visualize(a, Cl=Cl, save_path="routing_demo.png")
