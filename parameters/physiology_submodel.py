import numpy as np
import matplotlib.pyplot as plt
import os


# Potential lactation (kg per lactation)
potlact_primiparous = 880.0
potlact_multiparous = 950.0

# Cow-specific potential lactations (scaled to realistic Holstein yields).
# Using the same shape functions, but with higher potlact so that
# multiparous Holstein cows peak at ~32 kg/day instead of ~4.25 kg/day.
potlact_multiparous_cow = 7153.0
potlact_primiparous_cow = potlact_multiparous_cow / 2.0

# Breed- and parity-specific body weight parameters
Alpine_primiparous = {
    "BWmin": 48.8,
    "BW0": 52.1,
    "a": 0.158,
    "b": 0.0095,
    "day0": 8,
}
Alpine_multiparous = {
    "BWmin": 62.6,
    "BW0": 68.8,
    "a": 0.077,
    "b": 0.0079,
    "day0": 27,
}

Saanen_primiparous = {
    "BWmin": 51.8,
    "BW0": 56.4,
    "a": 0.158,
    "b": 0.0095,
    "day0": 8,
}

Saanen_multiparous = {
    "BWmin": 70.3,
    "BW0": 78.7,
    "a": 0.077,
    "b": 0.0079,
    "day0": 27,
}

# Simple Holstein cow profile (multiparous) for body weight.
# Uses goat multiparous shape parameters but scaled to cow BW.
Holstein_cow_multiparous = {
    "BWmin": 580.0,
    "BW0": 600.0,
    "a": 0.077,
    "b": 0.0079,
    "day0": 27,
}

Holstein_cow_primiparous = {
    "BWmin": 580.0,
    "BW0": 600.0,
    "a": 0.158,
    "b": 0.0095,
    "day0": 8,
}


def lactation_curve(days: np.ndarray, parity: str, breed: str | None = None) -> np.ndarray:
    shape_multiparous = 0.0054 * np.exp(-0.00342 * days) - 0.00222 * np.exp(-0.0555 * days)
    shape_primiparous = 0.00669 * np.exp(-0.00342 * days) - 0.00345 * np.exp(-0.0555 * days)

    if breed is not None and breed.startswith("Holstein_cow"):
        pot = potlact_primiparous_cow if parity == "primiparous" else potlact_multiparous_cow
    else:
        pot = potlact_primiparous if parity == "primiparous" else potlact_multiparous

    lactation = pot * (shape_primiparous if parity == "primiparous" else shape_multiparous)
    return np.maximum(lactation, 0.0)


def body_weight_curve(days: np.ndarray, params: dict) -> np.ndarray:
    BW = params["BWmin"] + (params["BW0"] - params["BWmin"]) * np.exp(-params["a"] * days) + np.exp(params["b"] * (days - params["day0"]))
    return BW


def dry_matter_intake_curve(days: np.ndarray, BW: np.ndarray, lactation: np.ndarray, DMI_co: float = 0.5) -> np.ndarray:
    """
    Dry Matter Intake (kg/day). Empirical linear model in BW, milk, and concentrate.
    """
    return 0.23 + 0.014 * BW + 0.298 * lactation + 0.260 * DMI_co


def get_params(breed: str, parity: str) -> dict:
    key = f"{breed}_{parity}"
    mapping = {
        "Alpine_primiparous": Alpine_primiparous,
        "Alpine_multiparous": Alpine_multiparous,
        "Saanen_primiparous": Saanen_primiparous,
        "Saanen_multiparous": Saanen_multiparous,
        "Holstein_cow_primiparous": Holstein_cow_primiparous,
        "Holstein_cow_multiparous": Holstein_cow_multiparous,
    }
    if key not in mapping:
        raise ValueError(f"Unknown combination: {breed=} {parity=}")
    return mapping[key]


def plot_curves_0_300_days(save_path: str | None = None) -> None:
    days = np.arange(0, 301)
    combos = [
        ("Alpine", "primiparous"),
        ("Alpine", "multiparous"),
        ("Saanen", "primiparous"),
        ("Saanen", "multiparous"),
        ("Holstein_cow", "multiparous"),
    ]
    colors = {
        ("Alpine", "primiparous"): "tab:blue",
        ("Alpine", "multiparous"): "tab:orange",
        ("Saanen", "primiparous"): "tab:green",
        ("Saanen", "multiparous"): "tab:red",
        ("Holstein_cow", "multiparous"): "k",
    }

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

    for breed, parity in combos:
        params = get_params(breed, parity)
        lact = lactation_curve(days, parity, breed=breed)
        bw = body_weight_curve(days, params)
        dmi = dry_matter_intake_curve(days, bw, lact)

        label = f"{breed} - {parity}"
        axes[0].plot(days, lact, label=label, color=colors[(breed, parity)])
        axes[1].plot(days, bw, label=label, color=colors[(breed, parity)])
        axes[2].plot(days, dmi, label=label, color=colors[(breed, parity)])

    axes[0].set_ylabel("Milk (kg/day)")
    axes[0].set_title("Lactation curves (0–300 d)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Body weight (kg)")
    axes[1].set_title("Body weight trajectories (0–300 d)")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("DMI (kg/day)")
    axes[2].set_xlabel("Days in milk")
    axes[2].set_title("Dry matter intake (0–300 d)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()


def main():
    # Save into Ziegen_modell/Results by default, independent of working directory
    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.normpath(os.path.join(this_dir, "..", "Results"))
        os.makedirs(results_dir, exist_ok=True)
        default_path = os.path.join(results_dir, "breed_parity_physiology_curves.png")
        plot_curves_0_300_days(default_path)
        print(f"Saved plot to {default_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()