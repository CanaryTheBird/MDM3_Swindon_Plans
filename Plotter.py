import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load results
df = pd.read_csv("bayesian_v4_investment_results.csv")

sectors = ["MATL", "ENG", "C27", "TRAN", "INFO", "FIN", "M72"]

jobs = "Cumulative job-years 2027-2031"
capital = "Capital income change in 2031"

# Filter successful simulations
df = df[df["Status"] == "Success"].copy()
df = df[~df["Scenario"].isin([218, 298, 304])]
df = df.dropna(subset=sectors + [jobs, capital])

# Convert investment amounts into proportions
X = df[sectors].to_numpy(dtype=float)
X = X / X.sum(axis=1, keepdims=True)

# K-means clustering
n_clusters = 12

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=30
)

df["Cluster"] = kmeans.fit_predict(X)

print("Silhouette score:", silhouette_score(X, df["Cluster"]))

# Order clusters by mean employment impact
cluster_order = (
    df.groupby("Cluster")[jobs]
    .mean()
    .sort_values(ascending=False)
    .index
)

# Create plot
fig, ax = plt.subplots(figsize=(12, 7))

colours = plt.get_cmap("tab20")
markers = ["o"] * 10 + ["s", "^"]

profiles = []

for rank, i in enumerate(cluster_order):

    group = df[df["Cluster"] == i]

    # Average investment distribution
    centre = pd.Series(
        kmeans.cluster_centers_[i],
        index=sectors
    )

    # Two largest average investment recipients
    top = centre.sort_values(ascending=False).head(2)

    label = " / ".join(
        f"{sector} {share:.0%}"
        for sector, share in top.items()
    )

    ax.scatter(
        group[jobs],
        group[capital],
        color=colours(rank),
        marker=markers[rank],
        label=f"{rank+1}: {label} (n={len(group)})",
        s=35,
        alpha=0.75
    )

    # Record cluster statistics
    profiles.append({
        "Cluster": rank + 1,
        "Count": len(group),
        "Mean job-years": group[jobs].mean(),
        "Mean capital income": group[capital].mean(),
        **centre.to_dict()
    })

# Calculate Pareto frontier
ordered = df.sort_values(
    [jobs, capital],
    ascending=False
)

frontier = ordered[
    ordered[capital] >
    ordered[capital].cummax().shift(fill_value=-np.inf)
].sort_values(jobs)

# Plot Pareto frontier
# Formatting
ax.axhline(0, color="grey", linestyle=":")
ax.axvline(0, color="grey", linestyle=":")

ax.set_xlabel("Cumulative net job-years (2027–2031)")
ax.set_ylabel("Real capital income change in 2031")

ax.set_title(
    "Employment–capital income trade-off: 12 investment clusters"
)

ax.legend(
    title="Cluster investment profile",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=9
)

ax.grid(alpha=0.2)

plt.tight_layout()

# Save figures
plt.savefig("12_cluster_tradeoff.pdf", bbox_inches="tight")
plt.savefig("12_cluster_tradeoff.png", dpi=300, bbox_inches="tight")

# Save cluster profiles
profiles = pd.DataFrame(profiles)

profiles.to_csv("12_cluster_profiles.csv", index=False)

print(profiles.round(3).to_string(index=False))

plt.show()