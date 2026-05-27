import os
import pandas as pd
import matplotlib.pyplot as plt


RESULT_DIR = "experiments/results"


def ensure_result_dir():
    os.makedirs(RESULT_DIR, exist_ok=True)


def annotate_bars(ax, bars, fmt="{:.3f}", offset=0.01, fontsize=10):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold"
        )


def plot_main_results():
    path = os.path.join(RESULT_DIR, "main_results.csv")
    df = pd.read_csv(path)

    plt.figure(figsize=(11, 7))
    colors = ["#5B8FF9", "#61DDAA", "#65789B", "#F6BD16", "#E8684A"]

    bars = plt.bar(
        df["method"],
        df["accuracy"],
        color=colors,
        edgecolor="black",
        linewidth=1.0
    )

    plt.title("KLUE-NLI Main Accuracy Comparison", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("Method", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(0, 0.65)
    plt.xticks(rotation=25, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    annotate_bars(plt.gca(), bars, fmt="{:.3f}", offset=0.01, fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(RESULT_DIR, "main_results_bar_chart.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_tokenizer_ratio():
    path = os.path.join(RESULT_DIR, "tokenizer_analysis.csv")
    df = pd.read_csv(path)

    plt.figure(figsize=(8, 6))
    colors = ["#5470C6", "#91CC75", "#FAC858"]

    bars = plt.bar(
        df["language"],
        df["subword_word_ratio"],
        color=colors,
        edgecolor="black",
        linewidth=1.0
    )

    plt.title("Tokenizer Fragmentation Analysis", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("Language", fontsize=14)
    plt.ylabel("Subword-to-Word Ratio", fontsize=14)
    plt.ylim(0, max(df["subword_word_ratio"]) + 0.4)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    annotate_bars(plt.gca(), bars, fmt="{:.3f}", offset=0.05, fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(RESULT_DIR, "tokenizer_ratio_bar_chart.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_attention_entropy():
    path = os.path.join(RESULT_DIR, "attention_entropy_analysis.csv")
    df = pd.read_csv(path)

    plt.figure(figsize=(10, 6))
    colors = ["#73C0DE", "#EE6666", "#91CC75"]

    bars = plt.bar(
        df["model"],
        df["attention_entropy"],
        color=colors,
        edgecolor="black",
        linewidth=1.0
    )

    plt.title("Raw Attention Entropy Comparison", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Average Attention Entropy", fontsize=14)
    plt.xticks(rotation=20, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    annotate_bars(plt.gca(), bars, fmt="{:.2f}", offset=2.0, fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(RESULT_DIR, "attention_entropy_bar_chart.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_entropy_distance():
    path = os.path.join(RESULT_DIR, "attention_entropy_analysis.csv")
    df = pd.read_csv(path)

    # Only compare student models with teacher distance larger than zero.
    student_df = df[df["entropy_distance_to_teacher"] > 0].copy()

    plt.figure(figsize=(9, 6))
    colors = ["#EA7CCC", "#5470C6"]

    bars = plt.bar(
        student_df["model"],
        student_df["entropy_distance_to_teacher"],
        color=colors,
        edgecolor="black",
        linewidth=1.0
    )

    plt.title("Attention Entropy Distance to Teacher", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Entropy Distance to Teacher", fontsize=14)
    plt.xticks(rotation=20, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    annotate_bars(plt.gca(), bars, fmt="{:.3f}", offset=0.8, fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(RESULT_DIR, "entropy_distance_bar_chart.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_ablation():
    path = os.path.join(RESULT_DIR, "ablation_results.csv")
    df = pd.read_csv(path)

    df = df.copy()
    df["beta_str"] = df["beta"].astype(str)

    beta_order = ["0.1", "0.01", "0.001"]
    df["beta_str"] = pd.Categorical(df["beta_str"], categories=beta_order, ordered=True)
    df = df.sort_values("beta_str")

    x_labels = df["beta_str"].astype(str).tolist()
    y = df["accuracy"].tolist()

    plt.figure(figsize=(8, 6))

    plt.plot(
        x_labels,
        y,
        marker="o",
        markersize=10,
        linewidth=3,
        color="#9A60B4"
    )

    plt.scatter(
        x_labels,
        y,
        s=160,
        color=["#EE6666", "#FAC858", "#91CC75"],
        edgecolor="black",
        linewidth=1.0,
        zorder=3
    )

    for x, val in zip(x_labels, y):
        plt.text(
            x,
            val + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    plt.title("SEAD Beta Ablation Study", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("Beta", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(0.30, 0.60)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(RESULT_DIR, "ablation_beta_chart.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ensure_result_dir()
    print("Generating colorful paper-style result figures...")

    plot_main_results()
    plot_tokenizer_ratio()
    plot_attention_entropy()
    plot_entropy_distance()
    plot_ablation()

    print("All figures have been generated successfully.")


if __name__ == "__main__":
    main()
