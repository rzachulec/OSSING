import json
from dataclasses import dataclass

import numpy as np


def _route_capacity(route_bits, capacities):
    """Return the tightest positive link capacity on a routed path."""
    used_links = [l for l, bit in enumerate(route_bits) if bit]
    if not used_links:
        return 0.0
    return float(min(capacities[l] for l in used_links))


@dataclass
class InputData:
    M: int  # number of clients (not including publisher)
    N: int  # number of objects
    S: int  # number of servers
    L: int  # number of links
    bn: np.ndarray  # object sizes
    Cl: np.ndarray  # link capacities
    A: np.ndarray  # routing tensor

    def __post_init__(self):
        self.bn = np.asarray(self.bn, dtype=float)
        self.Cl = np.asarray(self.Cl, dtype=float)
        self.A = np.asarray(self.A, dtype=int)

        if self.bn.shape != (self.N,):
            raise ValueError(
                f"bn has invalid shape: {self.bn.shape}, expected {(self.N,)}"
            )

        if self.Cl.shape != (self.L,):
            raise ValueError(
                f"Cl has invalid shape: {self.Cl.shape}, expected {(self.L,)}"
            )

        if self.A.shape != (self.M + 1, self.N, self.S, self.L):
            raise ValueError(
                f"A has invalid shape: {self.A.shape}, expected {(self.M + 1, self.N, self.S, self.L)}"
            )

        if not np.all((self.A == 0) | (self.A == 1)):
            raise ValueError("A must contain only 0 or 1 values")

        for n in range(self.N):
            if all(
                _route_capacity(self.A[0, n, s, :], self.Cl) <= 0.0
                for s in range(self.S)
            ):
                raise ValueError(f"Object {n} has no publisher-to-server route.")

        for m in range(self.M):
            for n in range(self.N):
                if all(
                    _route_capacity(self.A[m + 1, n, s, :], self.Cl) <= 0.0
                    for s in range(self.S)
                ):
                    raise ValueError(f"Client {m}, object {n} has no client route.")


def load_input_data(path: str) -> InputData:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    M = int(raw["M"])
    N = int(raw["N"])
    S = int(raw["S"])
    L = int(raw["L"])

    bn = np.array(raw["bn"], dtype=float)
    Cl = np.array(raw["Cl"], dtype=float)

    if "A" in raw:
        # dense format
        A = np.array(raw["A"], dtype=int)

    elif "A_ones" in raw:
        # sparse format -> expand to full tensor
        A = np.zeros((M + 1, N, S, L), dtype=int)

        for entry in raw["A_ones"]:
            if len(entry) != 4:
                raise ValueError(
                    f"Each A_ones entry must contain 4 indices, got: {entry}"
                )

            m, n, s, l = entry

            if not (0 <= m < M + 1):
                raise ValueError(f"Invalid m index: {m}")
            if not (0 <= n < N):
                raise ValueError(f"Invalid n index: {n}")
            if not (0 <= s < S):
                raise ValueError(f"Invalid s index: {s}")
            if not (0 <= l < L):
                raise ValueError(f"Invalid l index: {l}")

            A[m, n, s, l] = 1

    else:
        raise ValueError("Input file must contain either 'A' or 'A_ones'")

    return InputData(
        M=M,
        N=N,
        S=S,
        L=L,
        bn=bn,
        Cl=Cl,
        A=A,
    )
