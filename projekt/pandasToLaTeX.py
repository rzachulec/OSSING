import pandas as pd

df = pd.read_csv("comparison_results.csv")

cols = [
    "M",
    "N",
    "S",
    "L",
    "cplex_q",
    "heur_q",
    "gap_pct",
    "cplex_time_s",
    "heur_time_s",
]

print(
    df[cols].to_latex(
        index=False,
        float_format="%.4f"
    )
)