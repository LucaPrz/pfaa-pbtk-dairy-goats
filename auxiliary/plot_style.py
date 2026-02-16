"""
Common plotting style helpers for the pfaa-pbtk-dairy-goats project.

LaTeX-friendly style: serif font and Computer Modern math so figures
match documents for inclusion in a .tex paper. Does not require a
LaTeX binary (uses mathtext.fontset="cm").
"""

from matplotlib import pyplot as plt
import seaborn as sns


def set_paper_plot_style() -> None:
    """
    Apply a LaTeX-friendly publication style (serif font, CM math, high DPI).
    """
    plt.rcParams.update({"figure.max_open_warning": 0})
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern", "DejaVu Serif", "Times New Roman", "serif"]

    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.edgecolor"] = "none"

    sns.set_theme(style="white", font="serif", font_scale=1.0)
