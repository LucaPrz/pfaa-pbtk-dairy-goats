import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
@dataclass
class DMIParams:
    """Species-level empirical coefficients for the dry matter intake equation."""
    intercept: float
    coef_BW: float
    coef_lact: float
    coef_concentrate: float


@dataclass
class PhysiologyParams:
    """All physiological parameters for one (species, breed, parity) combination."""
    species: str
    breed: str
    parity: str
    # Body weight parameters
    BW_min: float
    BW0: float
    a: float
    b: float
    day0: float
    # Lactation parameters
    potlac: float
    lact_shape_coeffs: tuple[float, float, float, float]
    # (A1, k1, A2, k2) → shape(d) = A1*exp(-k1*d) − A2*exp(-k2*d)



PHYSIOLOGY_REGISTRY: dict[tuple[str, str, str], PhysiologyParams] = {
    ("goat", "Alpine", "primiparous"): PhysiologyParams(
        species="goat", breed="Alpine", parity="primiparous",
        BW_min=48.8, BW0=52.1, a=0.158, b=0.0095, day0=8,
        potlac=880,
        lact_shape_coeffs=(0.00669, 0.00342, 0.00345, 0.0555),
    ),
    ("goat", "Alpine", "multiparous"): PhysiologyParams(
        species="goat", breed="Alpine", parity="multiparous",
        BW_min=62.6, BW0=68.8, a=0.077, b=0.0079, day0=27,
        potlac=950,
        lact_shape_coeffs=(0.0054, 0.00342, 0.00222, 0.0555),
    ),
    ("goat", "Saanen", "primiparous"): PhysiologyParams(
        species="goat", breed="Saanen", parity="primiparous",
        BW_min=51.8, BW0=56.4, a=0.158, b=0.0095, day0=8,
        potlac=880,
        lact_shape_coeffs=(0.00669, 0.00342, 0.00345, 0.0555),
    ),
    ("goat", "Saanen", "multiparous"): PhysiologyParams(
        species="goat", breed="Saanen", parity="multiparous",
        BW_min=70.3, BW0=78.7, a=0.077, b=0.0079, day0=27,
        potlac=950,
        lact_shape_coeffs=(0.0054, 0.00342, 0.00222, 0.0555),
    ),
}

# Species-level DMI coefficients (empirical, from INRA red book for goats)
DMI_REGISTRY: dict[str, DMIParams] = {
    "goat": DMIParams(
        intercept=0.23,
        coef_BW=0.014,
        coef_lact=0.298,
        coef_concentrate=0.260,
    ),
}

def get_params(species: str, breed: str, parity: str) -> PhysiologyParams:
    key = (species, breed, parity)
    if key not in PHYSIOLOGY_REGISTRY:
        raise ValueError(
            f"Unknown combination: {species=}, {breed=}, {parity=}. "
            f"Available: {list(PHYSIOLOGY_REGISTRY.keys())}"
        )
    return PHYSIOLOGY_REGISTRY[key]


def get_dmi_params(species: str) -> DMIParams:
    if species not in DMI_REGISTRY:
        raise ValueError(
            f"No DMI parameters for species '{species}'. "
            f"Available: {list(DMI_REGISTRY.keys())}"
        )
    return DMI_REGISTRY[species]


def lactation_curve(days: np.ndarray, params: PhysiologyParams) -> np.ndarray:
    """Milk yield (kg/day) following a bi-exponential shape function."""
    A1, k1, A2, k2 = params.lact_shape_coeffs
    shape = A1 * np.exp(-k1 * days) - A2 * np.exp(-k2 * days)
    return np.maximum(params.potlac * shape, 0.0)


def body_weight_curve(days: np.ndarray, params: PhysiologyParams) -> np.ndarray:
    """Body weight (kg) trajectory over the lactation period."""
    return (
        params.BW_min
        + (params.BW0 - params.BW_min) * np.exp(-params.a * days)
        + np.exp(params.b * (days - params.day0))
    )


def dry_matter_intake_curve(
    days: np.ndarray,
    BW: np.ndarray,
    lactation: np.ndarray,
    species: str,
    DMI_co: float = 0.5,
) -> np.ndarray:
    """Dry matter intake (kg/day): empirical linear model."""
    p = get_dmi_params(species)
    return (
        p.intercept
        + p.coef_BW * BW
        + p.coef_lact * lactation
        + p.coef_concentrate * DMI_co
    )

def plot_curves(
    species: str,
    day_range: tuple[int, int] = (0, 300),
    save_path: str | None = None,
) -> None:
    """Plot lactation, body weight, and DMI curves for all registry entries
    matching the given species."""
    days = np.arange(day_range[0], day_range[1] + 1)

    entries = [p for p in PHYSIOLOGY_REGISTRY.values() if p.species == species]
    if not entries:
        raise ValueError(f"No registry entries found for species '{species}'.")

    cmap = plt.cm.tab10
    colors = {(p.breed, p.parity): cmap(i) for i, p in enumerate(entries)}

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

    for params in entries:
        lact = lactation_curve(days, params)
        bw = body_weight_curve(days, params)
        dmi = dry_matter_intake_curve(days, bw, lact, species=params.species)

        label = f"{params.breed} – {params.parity}"
        color = colors[(params.breed, params.parity)]
        axes[0].plot(days, lact, label=label, color=color)
        axes[1].plot(days, bw, label=label, color=color)
        axes[2].plot(days, dmi, label=label, color=color)

    d0, d1 = day_range
    axes[0].set_ylabel("Milk (kg/day)")
    axes[0].set_title(f"Lactation curves ({d0}–{d1} d) — {species}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Body weight (kg)")
    axes[1].set_title(f"Body weight trajectories ({d0}–{d1} d) — {species}")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("DMI (kg/day)")
    axes[2].set_xlabel("Days in milk")
    axes[2].set_title(f"Dry matter intake ({d0}–{d1} d) — {species}")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

def main() -> None:
    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.normpath(os.path.join(this_dir, "..", "Results"))
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "breed_parity_physiology_curves.png")
        plot_curves(species="goat", day_range=(0, 300), save_path=save_path)
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()