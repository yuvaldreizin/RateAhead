import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from pathlib import Path

def _default_order(labels: list[str]) -> list[str]:
    numeric = []
    non_numeric = []
    for lab in labels:
        if isinstance(lab, str) and lab.isdigit():
            numeric.append(int(lab))
        else:
            non_numeric.append(str(lab))
    numeric_sorted = [str(x) for x in sorted(set(numeric))]
    non_numeric_sorted = sorted(set(non_numeric))
    return numeric_sorted + non_numeric_sorted

def boxplot_by_group(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    order: Optional[list] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    color: str = "#c9d8ff",
    overlay_mean_std: bool = True,
    save_path: Optional[Path] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (12, 4),
):
    if df is None or df.empty:
        raise ValueError("Dataframe is empty; cannot plot.")

    work = df.copy()
    work[group_col] = work[group_col].astype(str)

    if order is None:
        order = _default_order(work[group_col].unique())

    fig, ax = plt.subplots(figsize=figsize)

    # Smaller, subtler outliers
    flierprops = dict(
        marker="o",
        markersize=2.2,
        markerfacecolor="gray",
        markeredgecolor="gray",
        markeredgewidth=0.0,
        alpha=0.35,
        linestyle="none",
    )

    sns.boxplot(
        data=work,
        x=group_col,
        y=value_col,
        order=order,
        ax=ax,
        color=color,
        showfliers=True,
        flierprops=flierprops,
    )

    summary = work.groupby(group_col)[value_col].agg(["mean", "std"]).reindex(order)

    if overlay_mean_std:
        positions = ax.get_xticks()  # robust with seaborn categorical axis
        ax.errorbar(
            positions,
            summary["mean"],
            yerr=summary["std"],
            fmt="o",
            color="red",
            ecolor="red",
            alpha=0.7,
            label="mean ± std",
        )

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Fix the last long tick label overlap
    ax.tick_params(axis="x", labelrotation=30)
    for t in ax.get_xticklabels():
        t.set_ha("right")

    ax.legend()
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, summary
