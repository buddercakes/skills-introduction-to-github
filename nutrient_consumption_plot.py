"""Plot transient nutrient concentration and consumption in a 1 cm hydrogel.

The model treats nutrient delivery as a one-dimensional diffusion-reaction
process in a porous hydrogel seeded with metabolically active cells. Nutrients
are supplied by a blood vessel at x = 0 cm (Dirichlet boundary) and diffuse
across the material while being consumed by the cells according to a
first-order uptake rate (k*C), a common approximation for aggregate cellular
uptake within engineered tissues.

Running this script will save a publication-ready figure illustrating how the
nutrient concentration and the local consumption rate vary with distance from
the blood vessel wall over several hours (1, 2, 5, 10, 20, and 30 h).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np


def backward_euler_step(
    concentration: np.ndarray,
    diffusivity: float,
    consumption_rate_constant: float,
    dx: float,
    dt: float,
    vessel_concentration: float,
) -> np.ndarray:
    """Advance the concentration field by one backward Euler time step."""

    n_points = concentration.size
    alpha = diffusivity * dt / dx**2
    beta = consumption_rate_constant * dt

    matrix = np.zeros((n_points, n_points))
    rhs = concentration.copy()

    # Dirichlet boundary at the vessel wall (x = 0 cm).
    matrix[0, 0] = 1.0
    rhs[0] = vessel_concentration

    # Interior points (central finite differences for the Laplacian).
    for i in range(1, n_points - 1):
        matrix[i, i - 1] = -alpha
        matrix[i, i] = 1 + 2 * alpha + beta
        matrix[i, i + 1] = -alpha

    # Neumann boundary at the scaffold edge (zero flux).
    matrix[-1, -2] = -2 * alpha
    matrix[-1, -1] = 1 + 2 * alpha + beta

    updated = np.linalg.solve(matrix, rhs)
    updated[0] = vessel_concentration
    return updated


def transient_profiles(
    x: np.ndarray,
    diffusivity: float,
    consumption_rate_constant: float,
    vessel_concentration: float,
    output_times: Iterable[float],
    base_time_step: float = 300.0,
) -> Dict[float, np.ndarray]:
    """Return concentration profiles at the requested time points (seconds)."""

    dx = x[1] - x[0]
    concentration = np.zeros_like(x)
    concentration[0] = vessel_concentration

    profiles: Dict[float, np.ndarray] = {}
    current_time = 0.0

    for target_time in sorted(output_times):
        while current_time < target_time:
            dt = min(base_time_step, target_time - current_time)
            concentration = backward_euler_step(
                concentration=concentration,
                diffusivity=diffusivity,
                consumption_rate_constant=consumption_rate_constant,
                dx=dx,
                dt=dt,
                vessel_concentration=vessel_concentration,
            )
            current_time += dt

        profiles[target_time] = concentration.copy()

    return profiles


def plot_profiles() -> None:
    """Compute and plot nutrient concentration and consumption profiles."""

    # Domain definition: 1 cm hydrogel near a blood vessel.
    thickness_cm = 1.0
    thickness_m = thickness_cm / 100.0
    x = np.linspace(0.0, thickness_m, 200)

    # Biophysical parameters for oxygen-like nutrient transport.
    diffusivity = 2.0e-9  # m^2/s, representative for porous hydrogels
    consumption_rate_constant = 4.0e-4  # 1/s, aggregated cellular uptake rate
    vessel_concentration = 0.2  # mol/m^3 (~0.2 mM nutrient at the vessel wall)

    output_hours = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
    output_seconds = [hour * 3600.0 for hour in output_hours]

    concentration_over_time = transient_profiles(
        x=x,
        diffusivity=diffusivity,
        consumption_rate_constant=consumption_rate_constant,
        vessel_concentration=vessel_concentration,
        output_times=output_seconds,
    )

    # Unit conversions for more intuitive plotting.
    distance_mm = x * 1e3

    fig, (ax_conc, ax_cons) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.95, len(output_hours)))

    for color, hour, seconds in zip(colors, output_hours, output_seconds):
        concentration = concentration_over_time[seconds]
        consumption = consumption_rate_constant * concentration

        concentration_uM = concentration * 1e6
        consumption_rate_uMol_per_L_s = consumption * 1e6

        label = f"{hour:g} h"

        ax_conc.plot(
            distance_mm,
            concentration_uM,
            color=color,
            linewidth=2.0,
            label=label,
        )

        ax_cons.plot(
            distance_mm,
            consumption_rate_uMol_per_L_s,
            color=color,
            linewidth=2.0,
            linestyle="--",
        )

    ax_conc.set_ylabel("Nutrient concentration (µM)")
    ax_conc.set_title("Nutrient transport dynamics in a 1 cm hydrogel")
    ax_conc.grid(True, which="major", linestyle=":", linewidth=0.8)
    ax_conc.legend(title="Elapsed time")

    ax_cons.set_xlabel("Distance from blood vessel (mm)")
    ax_cons.set_ylabel("Consumption rate (µmol·L⁻¹·s⁻¹)")
    ax_cons.grid(True, which="major", linestyle=":", linewidth=0.8)

    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "images" / "nutrient_consumption_profile.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved nutrient transport profile figure to {output_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    plot_profiles()
