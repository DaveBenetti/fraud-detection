
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    dataset_path = project_root / "data" / "raw" / "creditcard.csv"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    df = pd.read_csv(dataset_path)

    # 1. Class distribution
    class_counts = df["Class"].value_counts().sort_index()

    print("\n--- CLASS DISTRIBUTION ---")
    print(class_counts)
    print("\nPercentages:")
    print((df["Class"].value_counts(normalize=True).sort_index() * 100).round(4))

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x="Class", ax=ax)
    ax.set_title("Distribution of transaction classes")
    ax.set_xlabel("Class (0 = legitimate, 1 = fraud)")
    ax.set_ylabel("Number of transactions")
    fig.tight_layout()
    fig.savefig(reports_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Transaction amounts
    print("\n--- AMOUNT STATISTICS BY CLASS ---")
    print(
        df.groupby("Class")["Amount"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="Class", y="Amount", showfliers=False, ax=ax)
    ax.set_title("Transaction amounts by class")
    ax.set_xlabel("Class (0 = legitimate, 1 = fraud)")
    ax.set_ylabel("Amount")
    fig.tight_layout()
    fig.savefig(reports_dir / "amount_by_class.png", dpi=150)
    plt.close(fig)
    
    
    # 2b. Amount distributions on a logarithmic scale
    fig, ax = plt.subplots(figsize=(9, 5))

    for class_value, label in [(0, "Legitimate"), (1, "Fraud")]:
        amounts = df.loc[df["Class"] == class_value, "Amount"]
        sns.histplot(
            amounts,
            bins=50,
            stat="density",
            element="step",
            fill=False,
            ax=ax,
            label=label,
        )

    ax.set_xscale("symlog", linthresh=1)
    ax.set_title("Transaction amount distributions")
    ax.set_xlabel("Amount (symlog scale)")
    ax.set_ylabel("Density")
    ax.legend()

    fig.tight_layout()
    fig.savefig(reports_dir / "amount_distributions.png", dpi=150)
    plt.close(fig)

    # 3. Transaction timeline
    print("\n--- TIME RANGE ---")
    print(f"Minimum time: {df['Time'].min():.0f} seconds")
    print(f"Maximum time: {df['Time'].max():.0f} seconds")

    df["TimeHours"] = df["Time"] / 3600
    time_counts = (
        df.assign(TimeBin=(df["TimeHours"] // 1).astype(int))
        .groupby(["TimeBin", "Class"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    time_counts.plot(ax=ax)
    ax.set_title("Transactions over time")
    ax.set_xlabel("Hours since the beginning of the dataset")
    ax.set_ylabel("Number of transactions")
    ax.legend(["Legitimate", "Fraud"])
    fig.tight_layout()
    fig.savefig(reports_dir / "transactions_over_time.png", dpi=150)
    plt.close(fig)
    
    
    # 4. Compare PCA features between legitimate transactions and fraud
    feature_columns = [f"V{i}" for i in range(1, 29)]

    feature_means = df.groupby("Class")[feature_columns].mean().T
    feature_means.columns = ["Legitimate", "Fraud"]

    feature_means["Absolute difference"] = (
        feature_means["Fraud"] - feature_means["Legitimate"]
    ).abs()

    feature_means = feature_means.sort_values(
        "Absolute difference", ascending=False
    )

    print("\n--- FEATURES WITH LARGEST DIFFERENCE IN MEANS ---")
    print(feature_means.head(10).round(4))

    selected_features = feature_means.head(6).index.tolist()
    plot_data = df[["Class", *selected_features]].melt(
        id_vars="Class",
        var_name="Feature",
        value_name="Value",
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=plot_data,
        x="Feature",
        y="Value",
        hue="Class",
        showfliers=False,
        ax=ax,
    )
    ax.set_title("Selected feature distributions by class")
    ax.set_xlabel("PCA feature")
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(reports_dir / "feature_comparison.png", dpi=150)
    plt.close(fig)

    print("\n--- REPORTS GENERATED ---")
    for path in sorted(reports_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()