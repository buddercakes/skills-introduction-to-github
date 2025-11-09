"""Core biomedical equations and modeling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, pi, sqrt
from typing import Sequence, Tuple

import numpy as np

R_GAS_CONSTANT = 8.314462618  # J/(mol*K)
FARADAY_CONSTANT = 96485.33212  # C/mol
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K


def goldman_equation(
    permeabilities: Sequence[float],
    outside_concentrations: Sequence[float],
    inside_concentrations: Sequence[float],
    temperature_celsius: float = 37.0,
    ion_valences: Sequence[int] | None = None,
) -> float:
    """Compute membrane potential using the Goldman-Hodgkin-Katz voltage equation."""

    if not (
        len(permeabilities)
        == len(outside_concentrations)
        == len(inside_concentrations)
    ):
        raise ValueError("Permeabilities and concentration sequences must have equal length")

    temperature_kelvin = temperature_celsius + 273.15
    if ion_valences is None:
        ion_valences = [1 for _ in permeabilities]

    numerator = 0.0
    denominator = 0.0
    for p, c_out, c_in, z in zip(
        permeabilities, outside_concentrations, inside_concentrations, ion_valences
    ):
        if z > 0:
            numerator += p * c_out
            denominator += p * c_in
        else:
            numerator += p * c_in
            denominator += p * c_out

    return (R_GAS_CONSTANT * temperature_kelvin / (FARADAY_CONSTANT)) * log(numerator / denominator)


@dataclass
class HodgkinHuxleyState:
    """State variables for the classic Hodgkin-Huxley neuron model."""

    voltage: float
    m: float
    h: float
    n: float


def _alpha_m(v: float) -> float:
    return (0.1 * (25.0 - v)) / (exp((25.0 - v) / 10.0) - 1.0)


def _beta_m(v: float) -> float:
    return 4.0 * exp(-v / 18.0)


def _alpha_h(v: float) -> float:
    return 0.07 * exp(-v / 20.0)


def _beta_h(v: float) -> float:
    return 1.0 / (exp((30.0 - v) / 10.0) + 1.0)


def _alpha_n(v: float) -> float:
    return (0.01 * (10.0 - v)) / (exp((10.0 - v) / 10.0) - 1.0)


def _beta_n(v: float) -> float:
    return 0.125 * exp(-v / 80.0)


def hodgkin_huxley_step(
    state: HodgkinHuxleyState,
    stimulus_current: float,
    dt: float,
    g_na: float = 120.0,
    g_k: float = 36.0,
    g_l: float = 0.3,
    e_na: float = 115.0,
    e_k: float = -12.0,
    e_l: float = 10.613,
    c_m: float = 1.0,
) -> HodgkinHuxleyState:
    """Advance the Hodgkin-Huxley model one explicit Euler time step."""

    v = state.voltage
    m = state.m
    h = state.h
    n = state.n

    dm = _alpha_m(v) * (1.0 - m) - _beta_m(v) * m
    dh = _alpha_h(v) * (1.0 - h) - _beta_h(v) * h
    dn = _alpha_n(v) * (1.0 - n) - _beta_n(v) * n

    m_next = m + dt * dm
    h_next = h + dt * dh
    n_next = n + dt * dn

    g_na_current = g_na * (m_next**3) * h_next * (v - e_na)
    g_k_current = g_k * (n_next**4) * (v - e_k)
    g_l_current = g_l * (v - e_l)

    dv = (stimulus_current - g_na_current - g_k_current - g_l_current) / c_m
    v_next = v + dt * dv

    return HodgkinHuxleyState(voltage=v_next, m=m_next, h=h_next, n=n_next)


def nernst_potential(concentration_out: float, concentration_in: float, valence: int, temperature_celsius: float = 37.0) -> float:
    """Calculate equilibrium potential for an ion via the Nernst equation."""

    temperature_kelvin = temperature_celsius + 273.15
    return (
        (R_GAS_CONSTANT * temperature_kelvin) / (valence * FARADAY_CONSTANT)
    ) * log(concentration_out / concentration_in)


def reynolds_number(density: float, velocity: float, characteristic_length: float, viscosity: float) -> float:
    """Compute Reynolds number for blood flow."""

    return density * velocity * characteristic_length / viscosity


def laplace_wall_stress(pressure: float, radius: float, wall_thickness: float) -> float:
    """Calculate circumferential wall stress using Laplace's law."""

    return pressure * radius / (2.0 * wall_thickness)


def poiseuille_flow_rate(pressure_drop: float, radius: float, length: float, viscosity: float) -> float:
    """Poiseuille's law for volumetric flow rate in a cylindrical vessel."""

    return (pi * radius**4 * pressure_drop) / (8.0 * viscosity * length)


def fick_first_law(diffusivity: float, concentration_gradient: float) -> float:
    """Compute diffusive flux using Fick's first law."""

    return -diffusivity * concentration_gradient


def beer_lambert(absorptivity: float, path_length: float, concentration: float) -> float:
    """Beer-Lambert law relating absorbance to concentration."""

    return absorptivity * path_length * concentration


def gibbs_free_energy(delta_h: float, temperature_celsius: float, delta_s: float) -> float:
    """Calculate Gibbs free energy change."""

    return delta_h - (temperature_celsius + 273.15) * delta_s


def boltzmann_probability(energy: float, temperature_celsius: float) -> float:
    """Boltzmann distribution probability for a state of given energy."""

    temperature_kelvin = temperature_celsius + 273.15
    return exp(-energy / (BOLTZMANN_CONSTANT * temperature_kelvin))


def fourier_transform_2d(image: np.ndarray) -> np.ndarray:
    """Compute a centered 2D Fourier transform for medical imaging tasks."""

    return np.fft.fftshift(np.fft.fft2(image))


def kalman_filter_step(
    x_prior: np.ndarray,
    p_prior: np.ndarray,
    observation: np.ndarray,
    observation_model: np.ndarray,
    observation_covariance: np.ndarray,
    process_model: np.ndarray,
    process_covariance: np.ndarray,
    control_input: np.ndarray | None = None,
    control_model: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a linear Kalman filter predict-update cycle."""

    if control_input is not None and control_model is not None:
        x_predict = process_model @ x_prior + control_model @ control_input
    else:
        x_predict = process_model @ x_prior

    p_predict = process_model @ p_prior @ process_model.T + process_covariance

    innovation = observation - observation_model @ x_predict
    s = observation_model @ p_predict @ observation_model.T + observation_covariance
    k_gain = p_predict @ observation_model.T @ np.linalg.inv(s)

    x_posterior = x_predict + k_gain @ innovation
    p_posterior = (np.eye(p_predict.shape[0]) - k_gain @ observation_model) @ p_predict

    return x_posterior, p_posterior


def fast_ica(signals: np.ndarray, n_components: int, max_iter: int = 200, tol: float = 1e-4) -> np.ndarray:
    """Perform FastICA for EEG source separation."""

    signals_centered = signals - signals.mean(axis=1, keepdims=True)
    cov = np.cov(signals_centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    whitening_matrix = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    white = whitening_matrix @ signals_centered

    components = np.zeros((n_components, white.shape[0]))

    for i in range(n_components):
        w = np.random.randn(white.shape[0])
        w /= np.linalg.norm(w)

        for _ in range(max_iter):
            wx = w @ white
            g = np.tanh(wx)
            g_der = 1.0 - g**2
            w_new = (white * g).mean(axis=1) - g_der.mean() * w

            if i > 0:
                w_new -= components[:i].T @ (components[:i] @ w_new)

            w_new /= np.linalg.norm(w_new)

            if np.linalg.norm(w_new - w) < tol:
                break
            w = w_new

        components[i] = w

    return components @ white


def bayesian_diagnostic_probability(
    sensitivity: float, specificity: float, prevalence: float, positive_test: bool
) -> float:
    """Compute posterior probability for a diagnostic test result."""

    if positive_test:
        numerator = sensitivity * prevalence
        denominator = numerator + (1.0 - specificity) * (1.0 - prevalence)
    else:
        numerator = (1.0 - sensitivity) * prevalence
        denominator = numerator + specificity * (1.0 - prevalence)

    return numerator / denominator


def hidden_markov_forward(
    initial_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    observations: Sequence[int],
) -> np.ndarray:
    """Forward algorithm for hidden Markov models."""

    n_states = transition_matrix.shape[0]
    alpha = np.zeros((len(observations), n_states))
    alpha[0] = initial_probabilities * emission_matrix[:, observations[0]]

    for t in range(1, len(observations)):
        alpha[t] = (
            alpha[t - 1] @ transition_matrix
        ) * emission_matrix[:, observations[t]]

    return alpha


def k_means(data: np.ndarray, n_clusters: int, max_iter: int = 100, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Run k-means clustering for genomic data grouping."""

    rng = np.random.default_rng()
    centroids = data[rng.choice(len(data), size=n_clusters, replace=False)]

    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)

        new_centroids = np.array([data[labels == k].mean(axis=0) for k in range(n_clusters)])

        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    return centroids, labels


def convolutional_layer_forward(
    inputs: np.ndarray,
    filters: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Simple convolutional neural network layer forward pass."""

    n_filters, _, kernel_h, kernel_w = filters.shape
    batch_size, _, in_h, in_w = inputs.shape

    out_h = (in_h + 2 * padding - kernel_h) // stride + 1
    out_w = (in_w + 2 * padding - kernel_w) // stride + 1

    padded = np.pad(inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    outputs = np.zeros((batch_size, n_filters, out_h, out_w))

    for b in range(batch_size):
        for f in range(n_filters):
            for i in range(out_h):
                for j in range(out_w):
                    region = padded[
                        b,
                        :,
                        i * stride : i * stride + kernel_h,
                        j * stride : j * stride + kernel_w,
                    ]
                    outputs[b, f, i, j] = np.sum(region * filters[f])
            if bias is not None:
                outputs[b, f] += bias[f]

    return outputs


def recurrent_layer_step(
    previous_hidden: np.ndarray,
    input_vector: np.ndarray,
    weight_input: np.ndarray,
    weight_hidden: np.ndarray,
    bias: np.ndarray,
    activation: callable = np.tanh,
) -> np.ndarray:
    """Single time-step update for a vanilla recurrent neural network."""

    return activation(weight_input @ input_vector + weight_hidden @ previous_hidden + bias)


def gradient_descent(
    initial_parameters: np.ndarray,
    gradient_function: callable,
    learning_rate: float,
    n_iterations: int,
) -> np.ndarray:
    """Generic gradient descent optimizer."""

    params = initial_parameters.copy()
    for _ in range(n_iterations):
        params -= learning_rate * gradient_function(params)
    return params


def lagrangian_mechanics(lagrangian: callable, coordinates: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Compute Euler-Lagrange residuals numerically."""

    eps = 1e-6
    residuals = []
    for i in range(len(coordinates)):
        coords_plus = coordinates.copy()
        coords_minus = coordinates.copy()
        coords_plus[i] += eps
        coords_minus[i] -= eps

        d_l_dq = (lagrangian(coords_plus, velocities) - lagrangian(coords_minus, velocities)) / (2 * eps)

        vel_plus = velocities.copy()
        vel_minus = velocities.copy()
        vel_plus[i] += eps
        vel_minus[i] -= eps

        d_l_dqdot = (
            lagrangian(coordinates, vel_plus) - lagrangian(coordinates, vel_minus)
        ) / (2 * eps)
        residuals.append(d_l_dq - d_l_dqdot)
    return np.array(residuals)


def euler_bernoulli_deflection(
    load: float,
    length: float,
    elastic_modulus: float,
    area_moment: float,
    position: float,
) -> float:
    """Compute beam deflection under uniform load."""

    return (load * position**2) * (3 * length - position) / (6 * elastic_modulus * area_moment)


def linear_1d_fem_stiffness(n_elements: int, length: float, conductivity: float) -> np.ndarray:
    """Assemble stiffness matrix for 1D linear finite elements."""

    h = length / n_elements
    k_local = conductivity / h * np.array([[1, -1], [-1, 1]])
    size = n_elements + 1
    k_global = np.zeros((size, size))

    for e in range(n_elements):
        nodes = [e, e + 1]
        for i in range(2):
            for j in range(2):
                k_global[nodes[i], nodes[j]] += k_local[i, j]

    return k_global


def two_compartment_toxicokinetics(
    dose: float,
    k12: float,
    k21: float,
    kel: float,
    t: float,
) -> Tuple[float, float]:
    """Compute amounts in central and peripheral compartments."""

    a = (-(k12 + kel + k21) + sqrt((k12 + kel + k21) ** 2 - 4 * k21 * kel)) / 2
    b = (-(k12 + kel + k21) - sqrt((k12 + kel + k21) ** 2 - 4 * k21 * kel)) / 2

    c1 = dose * (a + k21) / (a - b)
    c2 = dose - c1

    central = c1 * exp(a * t) + c2 * exp(b * t)
    peripheral = dose - central
    return central, peripheral


def state_space_step(
    state: np.ndarray,
    control: np.ndarray,
    system_matrix: np.ndarray,
    input_matrix: np.ndarray,
) -> np.ndarray:
    """Discrete-time state-space propagation for control systems."""

    return system_matrix @ state + input_matrix @ control


def maxwell_displacement_current(permittivity: float, electric_field_change_rate: float) -> float:
    """Compute displacement current density from Maxwell's equations."""

    return permittivity * electric_field_change_rate


def respiratory_compliance(pressure: Sequence[float], volume: Sequence[float]) -> float:
    """Estimate respiratory system compliance from pressure-volume data."""

    pressure = np.asarray(pressure)
    volume = np.asarray(volume)
    coeffs = np.polyfit(pressure, volume, 1)
    return coeffs[0]


def discrete_wavelet_transform(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute single-level Haar wavelet transform for ECG analysis."""

    low_pass = (signal[0::2] + signal[1::2]) / sqrt(2.0)
    high_pass = (signal[0::2] - signal[1::2]) / sqrt(2.0)
    return low_pass, high_pass


def pca_reduce(data: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Principal component analysis for biomedical datasets."""

    data_centered = data - data.mean(axis=0)
    u, s, vh = np.linalg.svd(data_centered, full_matrices=False)
    components = vh[:n_components]
    transformed = data_centered @ components.T
    explained_variance = (s[:n_components] ** 2) / (len(data) - 1)
    return transformed, components, explained_variance


def transcripts_per_million(counts: np.ndarray, gene_lengths_kb: np.ndarray) -> np.ndarray:
    """Convert raw counts to TPM values."""

    rpk = counts / gene_lengths_kb
    scaling = rpk.sum() / 1e6
    return rpk / scaling


def needleman_wunsch(
    seq_a: str,
    seq_b: str,
    match_score: float = 1.0,
    mismatch_penalty: float = -1.0,
    gap_penalty: float = -1.0,
) -> Tuple[float, str, str]:
    """Perform global sequence alignment."""

    m, n = len(seq_a), len(seq_b)
    scores = np.zeros((m + 1, n + 1))
    scores[:, 0] = np.arange(m + 1) * gap_penalty
    scores[0, :] = np.arange(n + 1) * gap_penalty

    traceback = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = scores[i - 1, j - 1] + (match_score if seq_a[i - 1] == seq_b[j - 1] else mismatch_penalty)
            delete = scores[i - 1, j] + gap_penalty
            insert = scores[i, j - 1] + gap_penalty
            scores[i, j] = max(match, delete, insert)
            traceback[i, j] = np.argmax([match, delete, insert])

    align_a = []
    align_b = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and traceback[i, j] == 0:
            align_a.append(seq_a[i - 1])
            align_b.append(seq_b[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or traceback[i, j] == 1):
            align_a.append(seq_a[i - 1])
            align_b.append("-")
            i -= 1
        else:
            align_a.append("-")
            align_b.append(seq_b[j - 1])
            j -= 1

    return scores[m, n], "".join(reversed(align_a)), "".join(reversed(align_b))


def control_coefficient(flux: float, enzyme_activity: float, delta_flux: float, delta_activity: float) -> float:
    """Calculate metabolic control analysis flux control coefficient."""

    return (delta_flux / flux) / (delta_activity / enzyme_activity)


def wright_fisher_step(population: int, allele_frequency: float, selection_coefficient: float = 0.0) -> float:
    """One generation update for Wright-Fisher model with selection."""

    fitness_a = 1.0 + selection_coefficient
    fitness_b = 1.0
    mean_fitness = allele_frequency * fitness_a + (1.0 - allele_frequency) * fitness_b
    p_prime = (allele_frequency * fitness_a) / mean_fitness
    counts = np.random.binomial(2 * population, p_prime)
    return counts / (2 * population)


def transporter_flux(vmax: float, substrate_concentration: float, km: float) -> float:
    """Membrane transporter kinetics using Michaelis-Menten form."""

    return vmax * substrate_concentration / (km + substrate_concentration)


def pcr_amplification(initial_copies: float, efficiency: float, cycles: int) -> float:
    """PCR amplification model with constant efficiency."""

    return initial_copies * (1.0 + efficiency) ** cycles


def diffusion_tensor_metrics(tensor: np.ndarray) -> Tuple[float, float, float]:
    """Return mean diffusivity and fractional anisotropy metrics."""

    eigvals, _ = np.linalg.eigh(tensor)
    mean_diffusivity = eigvals.mean()
    numerator = 1.5 * np.sum((eigvals - mean_diffusivity) ** 2)
    denominator = np.sum(eigvals**2)
    fractional_anisotropy = sqrt(numerator / denominator)
    axial_diffusivity = eigvals.max()
    return mean_diffusivity, fractional_anisotropy, axial_diffusivity


def degree_centrality(adj_matrix: np.ndarray) -> np.ndarray:
    """Degree centrality for biological networks."""

    return adj_matrix.sum(axis=1)


def betweenness_centrality(adj_matrix: np.ndarray) -> np.ndarray:
    """Approximate betweenness centrality using Floyd-Warshall distances."""

    n = adj_matrix.shape[0]
    dist = np.where(adj_matrix > 0, adj_matrix, np.inf)
    np.fill_diagonal(dist, 0)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    centrality = np.zeros(n)
    for s in range(n):
        for t in range(n):
            if s == t:
                continue
            shortest = dist[s, t]
            if np.isinf(shortest) or shortest == 0:
                continue
            for v in range(n):
                if v == s or v == t:
                    continue
                if dist[s, v] + dist[v, t] == shortest:
                    centrality[v] += 1
    return centrality


def brownian_motion(n_steps: int, diffusion_coefficient: float, dt: float) -> np.ndarray:
    """Simulate 1D Brownian motion trajectory."""

    std = sqrt(2.0 * diffusion_coefficient * dt)
    steps = np.random.normal(0.0, std, size=n_steps)
    return np.cumsum(steps)


def inverse_dynamics(moment_of_inertia: float, angular_acceleration: float) -> float:
    """Basic biomechanical inverse dynamics to compute joint torque."""

    return moment_of_inertia * angular_acceleration


def sir_step(susceptible: float, infected: float, recovered: float, beta: float, gamma: float, dt: float) -> Tuple[float, float, float]:
    """Single integration step for the SIR infectious disease model."""

    dS = -beta * susceptible * infected
    dI = beta * susceptible * infected - gamma * infected
    dR = gamma * infected

    return (
        susceptible + dS * dt,
        infected + dI * dt,
        recovered + dR * dt,
    )


def docking_score(binding_energy: float, desolvation_energy: float, entropy_penalty: float) -> float:
    """Simple molecular docking scoring function."""

    return binding_energy + desolvation_energy + entropy_penalty


def conditional_probability(parent_states: Sequence[int], conditional_table: np.ndarray, query_state: int) -> float:
    """Evaluate probability in a graphical model for gene regulation."""

    index = tuple(parent_states)
    return conditional_table[index + (query_state,)]


def emg_root_mean_square(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Compute RMS envelope for EMG signal processing."""

    squared = signal**2
    cumsum = np.cumsum(np.insert(squared, 0, 0))
    rms = np.sqrt((cumsum[window_size:] - cumsum[:-window_size]) / window_size)
    return rms


def metabolic_flux_analysis(stoichiometry: np.ndarray, fluxes: np.ndarray) -> np.ndarray:
    """Compute residuals for flux balance analysis."""

    return stoichiometry @ fluxes


def nanoparticle_diffusion_coefficient(temperature_celsius: float, viscosity: float, radius: float) -> float:
    """Stokes-Einstein diffusion coefficient for nanoparticles."""

    temperature_kelvin = temperature_celsius + 273.15
    return BOLTZMANN_CONSTANT * temperature_kelvin / (6 * pi * viscosity * radius)


def lattice_boltzmann_step(
    distribution: np.ndarray,
    relaxation_time: float,
    equilibrium: np.ndarray,
) -> np.ndarray:
    """Single BGK lattice Boltzmann relaxation step."""

    return distribution - (distribution - equilibrium) / relaxation_time


def cfd_cfl_number(velocity: float, dt: float, dx: float) -> float:
    """Compute the Courant-Friedrichs-Lewy number for CFD stability."""

    return velocity * dt / dx


def genetic_algorithm_step(
    population: np.ndarray,
    fitness: np.ndarray,
    mutation_rate: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Perform selection, crossover, and mutation for a simple GA."""

    if rng is None:
        rng = np.random.default_rng()

    probabilities = fitness / fitness.sum()
    parents_idx = rng.choice(len(population), size=len(population), p=probabilities)
    parents = population[parents_idx]

    offspring = parents.copy()
    for i in range(0, len(offspring), 2):
        if i + 1 < len(offspring):
            crossover_point = rng.integers(1, offspring.shape[1])
            offspring[i, crossover_point:], offspring[i + 1, crossover_point:] = (
                offspring[i + 1, crossover_point:].copy(),
                offspring[i, crossover_point:].copy(),
            )

    mutations = rng.random(offspring.shape) < mutation_rate
    offspring = np.where(mutations, 1 - offspring, offspring)
    return offspring


def crispr_editing_probability(guide_efficiency: float, repair_probability: float, delivery_efficiency: float) -> float:
    """Model CRISPR editing success probability."""

    return guide_efficiency * repair_probability * delivery_efficiency


def tensor_cp_decomposition(tensor: np.ndarray, rank: int, max_iter: int = 100, tol: float = 1e-5) -> Tuple[list[np.ndarray], float]:
    """Canonical polyadic decomposition via alternating least squares."""

    factors = [
        np.random.rand(tensor.shape[mode], rank)
        for mode in range(tensor.ndim)
    ]

    for _ in range(max_iter):
        for mode in range(tensor.ndim):
            unfolded = np.reshape(
                np.moveaxis(tensor, mode, 0),
                (tensor.shape[mode], -1),
            )

            khatri_rao = factors[(mode + 1) % tensor.ndim]
            for other in range(mode + 2, mode + tensor.ndim):
                khatri_rao = np.kron(khatri_rao, factors[other % tensor.ndim])

            pseudo_inv = np.linalg.pinv(khatri_rao)
            factors[mode] = unfolded @ pseudo_inv

        reconstructed = np.zeros_like(tensor)
        for r in range(rank):
            outer = factors[0][:, r]
            for mode in range(1, tensor.ndim):
                outer = np.multiply.outer(outer, factors[mode][:, r])
            reconstructed += outer

        error = np.linalg.norm(tensor - reconstructed)
        if error < tol:
            break

    return factors, error

