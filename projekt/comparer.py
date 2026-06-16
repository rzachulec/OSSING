from __future__ import annotations

import argparse
import csv
from pathlib import Path

import ModelReader
import can_cplex
import can_heuristic


def get_objective(result):
    if "objective_exact" in result:
        return result["objective_exact"]

    if "objective_delay" in result:
        return result["objective_delay"]

    return None


def get_max_util(result):
    return max(
        stats["utilization"]
        for stats in result["link_stats"].values()
    )


def run_single_instance(instance_path):
    instance = ModelReader.load_input_data(instance_path)

    cplex_result = can_cplex.solve_instance(instance)

    heuristic_result = can_heuristic.solve_instance(instance)

    cplex_q = get_objective(cplex_result)
    heur_q = get_objective(heuristic_result)

    gap_pct = None
    if cplex_q and cplex_q != 0:
        gap_pct = 100.0 * (heur_q - cplex_q) / cplex_q

    return {
        "file": Path(instance_path).name,

        "M": instance.M,
        "N": instance.N,
        "S": instance.S,
        "L": instance.L,

        "cplex_q": cplex_q,
        "heur_q": heur_q,

        "gap_pct": gap_pct,

        "cplex_time_s": cplex_result.get("solve_time_s"),
        "heur_time_s": heuristic_result.get("solve_time_s"),

        "heur_iterations":
            len(heuristic_result.get("iteration_history", [])),

        "cplex_max_util":
            get_max_util(cplex_result),

        "heur_max_util":
            get_max_util(heuristic_result),
    }


def collect_files(path):
    path = Path(path)

    if path.is_file():
        return [path]

    return sorted(path.glob("*.json"))


def save_csv(rows, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "M",
                "N",
                "S",
                "L",
                "cplex_q",
                "heur_q",
                "gap_pct",
                "cplex_time_s",
                "heur_time_s",
                "heur_iterations",
                "cplex_max_util",
                "heur_max_util",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "path",
        help="JSON file or directory containing JSON files",
    )

    parser.add_argument(
        "--csv",
        default="comparison_results.csv",
        help="Output CSV file",
    )

    args = parser.parse_args()

    rows = []

    files = collect_files(args.path)

    for file_path in files:

        print(f"Running: {file_path}")

        try:
            row = run_single_instance(str(file_path))
            rows.append(row)

            print(
                f"  CPLEX={row['cplex_q']:.5f} "
                f"HEUR={row['heur_q']:.5f} "
                f"GAP={row['gap_pct']:.2f}%"
            )

        except Exception as e:

            print(f"  ERROR: {e}"
             f"HEUR={row['heur_q']:.5f} ")

    save_csv(rows, args.csv)

    print()
    print(f"Saved results to: {args.csv}")


if __name__ == "__main__":
    main()