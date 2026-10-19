"""
pca_direction_density.py

Helper functions for:

1. PCA projection of an m x n matrix to 3 x n and 2 x n.
2. Direction-only density visualization on the unit circle from 2D PCA results.
3. Direction-only density visualization on the unit sphere from 3D PCA results.

Assumption:
    X has shape (m, n), where columns are samples/vectors and rows are features.

Dependencies:
    numpy
    scikit-learn
    matplotlib
    plotly

Optional but recommended:
    scipy

Install:
    pip install numpy scikit-learn matplotlib plotly scipy
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, Any


def pca_project_rows(
    X: np.ndarray,
    n_components: int = 3,
    *,
    center: bool = True,
    random_state: Optional[int] = 0,
    svd_solver: str = "auto",
) -> Dict[str, Any]:
    """
    Perform PCA on a matrix X of shape m x n, reducing the row dimension.

    This treats columns of X as samples. Internally, sklearn expects samples x features,
    so X.T has shape n x m.

    Parameters
    ----------
    X:
        Input matrix of shape (m, n). Columns are m-dimensional vectors.
    n_components:
        Number of PCA components to compute. Use 3 if you want both 3D and 2D outputs.
    center:
        If True, PCA will mean-center features. sklearn PCA centers by default.
        If False, use TruncatedSVD instead of PCA-like centering.
    random_state:
        Random state for randomized solvers.
    svd_solver:
        Solver passed to sklearn.decomposition.PCA.
        Options include "auto", "full", "randomized".

    Returns
    -------
    dict with:
        "pca_3":
            3 x n projected data if n_components >= 3, else None.
        "pca_2":
            2 x n projected data if n_components >= 2.
        "model":
            Fitted PCA or TruncatedSVD model.
        "explained_variance_ratio":
            Explained variance ratio, if available.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D with shape m x n. Got shape {X.shape}.")

    m, n = X.shape

    if n_components < 2:
        raise ValueError("n_components must be at least 2.")

    if n_components > min(m, n):
        raise ValueError(
            f"n_components={n_components} cannot exceed min(m, n)={min(m, n)}."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")

    # sklearn expects samples x features, so transpose to n x m.
    XT = X.T

    if center:
        from sklearn.decomposition import PCA

        model = PCA(
            n_components=n_components,
            svd_solver=svd_solver,
            random_state=random_state,
        )
    else:
        # Useful for sparse-like or already-centered data where you do not want centering.
        from sklearn.decomposition import TruncatedSVD

        model = TruncatedSVD(
            n_components=n_components,
            random_state=random_state,
        )

    projected = model.fit_transform(XT)  # shape: n x n_components
    projected = projected.T              # shape: n_components x n

    pca_2 = projected[:2, :]
    pca_3 = projected[:3, :] if n_components >= 3 else None

    explained = getattr(model, "explained_variance_ratio_", None)

    return {
        "pca_2": pca_2,
        "pca_3": pca_3,
        "model": model,
        "explained_variance_ratio": explained,
    }


def scale_alpha_vector(
    alpha_vector: np.ndarray,
    *,
    expected_length: int,
    valid_mask: Optional[np.ndarray] = None,
    constant_value: float = 1.0,
) -> np.ndarray:
    """
    Scale a user-provided vector to [0, 1] for use as per-point alpha values.

    Parameters
    ----------
    alpha_vector:
        Array-like vector of length n, where n is the number of columns/samples.
    expected_length:
        Expected length of alpha_vector.
    valid_mask:
        Optional boolean mask of length n. If provided, only valid entries are returned.
        This should match the mask used to remove zero vectors after normalization.
    constant_value:
        If all values in alpha_vector are the same, all scaled alpha values are set
        to this value. Default is 1.0 so points remain visible.

    Returns
    -------
    scaled_alpha:
        Array of alpha values in [0, 1].
    """
    alpha_vector = np.asarray(alpha_vector, dtype=float).reshape(-1)

    if alpha_vector.shape[0] != expected_length:
        raise ValueError(
            f"alpha_vector must have length {expected_length}. "
            f"Got length {alpha_vector.shape[0]}."
        )

    if not np.all(np.isfinite(alpha_vector)):
        raise ValueError("alpha_vector contains NaN or infinite values.")

    min_val = np.min(alpha_vector)
    max_val = np.max(alpha_vector)

    if np.isclose(max_val, min_val):
        scaled = np.full_like(alpha_vector, fill_value=constant_value, dtype=float)
    else:
        scaled = (alpha_vector - min_val) / (max_val - min_val)

    scaled = np.clip(scaled, 0.0, 1.0)

    if valid_mask is not None:
        scaled = scaled[valid_mask]

    return scaled


def pca_project_to_2d_and_3d(
    X: np.ndarray,
    *,
    center: bool = True,
    random_state: Optional[int] = 0,
    svd_solver: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Convenience wrapper that returns both 2D and 3D PCA projections.

    Parameters
    ----------
    X:
        Matrix of shape m x n.

    Returns
    -------
    pca_2:
        Array of shape 2 x n.
    pca_3:
        Array of shape 3 x n.
    model:
        Fitted PCA or TruncatedSVD model.

    Notes
    -----
    This computes 3 components once and uses the first two components for the 2D result.
    For ordinary PCA, this is equivalent to doing a separate 2-component PCA, up to
    numerical/sign conventions.
    """
    result = pca_project_rows(
        X,
        n_components=3,
        center=center,
        random_state=random_state,
        svd_solver=svd_solver,
    )
    return result["pca_2"], result["pca_3"], result["model"]


def _normalize_columns(
    Y: np.ndarray,
    *,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize columns of Y.

    Parameters
    ----------
    Y:
        Array of shape d x n.

    Returns
    -------
    Y_unit:
        Column-normalized array of shape d x n.
    valid_mask:
        Boolean mask of length n indicating nonzero vectors.
    """
    norms = np.linalg.norm(Y, axis=0)
    valid = norms > eps

    Y_unit = np.zeros_like(Y, dtype=float)
    Y_unit[:, valid] = Y[:, valid] / norms[valid]

    return Y_unit, valid


def _smooth_circular_1d(
    counts: np.ndarray,
    sigma_bins: float,
) -> np.ndarray:
    """
    Smooth a 1D circular histogram.

    Uses scipy if available. Falls back to a simple wrapped Gaussian convolution.
    """
    counts = np.asarray(counts, dtype=float)

    if sigma_bins is None or sigma_bins <= 0:
        return counts

    try:
        from scipy.ndimage import gaussian_filter1d

        return gaussian_filter1d(counts, sigma=sigma_bins, mode="wrap")
    except Exception:
        # Fallback: manual circular Gaussian convolution.
        radius = max(1, int(np.ceil(4 * sigma_bins)))
        offsets = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
        kernel /= kernel.sum()

        smoothed = np.zeros_like(counts, dtype=float)
        n = len(counts)

        for offset, weight in zip(offsets, kernel):
            smoothed += weight * np.roll(counts, offset)

        return smoothed
    

def _rgb_string_to_rgba_string(rgb_string: str, alpha: float) -> str:
    """
    Convert a Plotly rgb color string to an rgba color string.

    Example
    -------
    'rgb(68, 1, 84)' -> 'rgba(68, 1, 84, 0.5)'
    """
    rgb_string = rgb_string.strip()

    if rgb_string.startswith("rgb("):
        values = rgb_string[4:-1]
        return f"rgba({values}, {alpha})"

    if rgb_string.startswith("#"):
        hex_color = rgb_string.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"

    # Fallback. This should rarely happen with Plotly colorscales.
    return rgb_string


def _plotly_rgba_colors_from_values(
    values: np.ndarray,
    alpha_values: np.ndarray,
    *,
    colorscale: str = "Viridis",
) -> list[str]:
    """
    Map numeric values to Plotly colors and attach per-point alpha values.

    Parameters
    ----------
    values:
        Numeric values used for color.
    alpha_values:
        Per-point alpha values in [0, 1].
    colorscale:
        Plotly colorscale name.

    Returns
    -------
    rgba_colors:
        List of rgba color strings.
    """
    from plotly.colors import sample_colorscale

    values = np.asarray(values, dtype=float)
    alpha_values = np.asarray(alpha_values, dtype=float)

    vmin = np.min(values)
    vmax = np.max(values)

    if np.isclose(vmin, vmax):
        scaled_values = np.full_like(values, 0.5, dtype=float)
    else:
        scaled_values = (values - vmin) / (vmax - vmin)

    sampled = sample_colorscale(colorscale, scaled_values)

    rgba_colors = [
        _rgb_string_to_rgba_string(color, float(alpha))
        for color, alpha in zip(sampled, alpha_values)
    ]

    return rgba_colors


def compute_2d_direction_density(
    Y2: np.ndarray,
    *,
    bins: int = 180,
    smooth_sigma_bins: float = 2.0,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """
    Compute direction-only angular density for 2D vectors.

    Magnitudes are ignored. Each vector is projected to the unit circle, and density
    is estimated as a smoothed histogram over vector angle.

    Parameters
    ----------
    Y2:
        PCA result of shape 2 x n.
    bins:
        Number of angular bins around the circle.
    smooth_sigma_bins:
        Gaussian smoothing width, measured in histogram bins.
    eps:
        Minimum norm for a vector to be considered valid.

    Returns
    -------
    dict with:
        "unit_vectors":
            Array of shape 2 x n_valid.
        "angles":
            Angles in radians, length n_valid, in [0, 2pi).
        "density":
            Density value per valid vector.
        "bin_centers":
            Angular bin centers.
        "bin_density":
            Smoothed density per angular bin.
        "valid_mask":
            Boolean mask indicating which original columns were used.
    """
    Y2 = np.asarray(Y2, dtype=float)

    if Y2.ndim != 2 or Y2.shape[0] != 2:
        raise ValueError(f"Y2 must have shape 2 x n. Got {Y2.shape}.")

    Y2_unit, valid = _normalize_columns(Y2, eps=eps)
    U = Y2_unit[:, valid]

    if U.shape[1] == 0:
        raise ValueError("No nonzero 2D vectors found after PCA projection.")

    angles = np.mod(np.arctan2(U[1], U[0]), 2 * np.pi)

    edges = np.linspace(0, 2 * np.pi, bins + 1)
    counts, _ = np.histogram(angles, bins=edges)

    bin_density = _smooth_circular_1d(counts, smooth_sigma_bins)

    # Convert each angle to its bin's density.
    bin_idx = np.searchsorted(edges, angles, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, bins - 1)
    density = bin_density[bin_idx]

    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    return {
        "unit_vectors": U,
        "angles": angles,
        "density": density,
        "bin_centers": bin_centers,
        "bin_density": bin_density,
        "valid_mask": valid,
    }


def plot_2d_direction_density_circle(
    Y2: np.ndarray,
    *,
    bins: int = 180,
    smooth_sigma_bins: float = 2.0,
    point_size: float = 18,
    alpha: float = 0.85,
    alpha_vector: Optional[np.ndarray] = None,
    cmap: str = "viridis",
    ax=None,
    show_colorbar: bool = True,
    title: str = "2D PCA Direction Density",
):
    """
    Create a 2D unit-circle plot colored by direction-only density.

    Magnitude is ignored. All valid vectors are normalized to the unit circle.
    Vectors with similar angle receive similar positions and contribute to the same
    angular density.

    Parameters
    ----------
    Y2:
        PCA result of shape 2 x n.
    bins:
        Number of angle bins.
    smooth_sigma_bins:
        Gaussian smoothing width in bins.
    point_size:
        Scatter point size.
    alpha:
        Scalar scatter alpha used when alpha_vector is not provided.
    alpha_vector:
        Optional vector of length n, where n is the number of columns in Y2.
        This vector is min-max scaled to [0, 1], then used as the per-point alpha
        value for each corresponding vector.
    cmap:
        Matplotlib colormap.
    ax:
        Optional matplotlib axis.
    show_colorbar:
        If True, add a colorbar.
    title:
        Plot title.

    Returns
    -------
    fig, ax, density_info
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    density_info = compute_2d_direction_density(
        Y2,
        bins=bins,
        smooth_sigma_bins=smooth_sigma_bins,
    )

    U = density_info["unit_vectors"]
    density = density_info["density"]
    valid_mask = density_info["valid_mask"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    if alpha_vector is None:
        sc = ax.scatter(
            U[0],
            U[1],
            c=density,
            s=point_size,
            alpha=alpha,
            cmap=cmap,
            edgecolors="none",
        )
    else:
        per_point_alpha = scale_alpha_vector(
            alpha_vector,
            expected_length=Y2.shape[1],
            valid_mask=valid_mask,
        )

        norm = mpl.colors.Normalize(
            vmin=np.min(density),
            vmax=np.max(density),
        )
        cmap_obj = plt.get_cmap(cmap)

        colors = cmap_obj(norm(density))
        colors[:, 3] = per_point_alpha

        sc = ax.scatter(
            U[0],
            U[1],
            c=colors,
            s=point_size,
            edgecolors="none",
        )

        # Create a separate ScalarMappable so the colorbar still represents density.
        sc_for_colorbar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sc_for_colorbar.set_array(density)
        sc = sc_for_colorbar

    # Draw unit circle boundary.
    theta = np.linspace(0, 2 * np.pi, 512)
    ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1.0, alpha=0.6)

    # Draw x/y axes lightly.
    ax.axhline(0, color="gray", linewidth=0.7, alpha=0.4)
    ax.axvline(0, color="gray", linewidth=0.7, alpha=0.4)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel("Normalized PC1")
    ax.set_ylabel("Normalized PC2")
    ax.set_title(title)

    if show_colorbar:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
        cbar.set_label("Direction density")

    if alpha_vector is not None:
        density_info["alpha"] = per_point_alpha

    return fig, ax, density_info


def _smooth_spherical_histogram(
    counts: np.ndarray,
    sigma_z_bins: float,
    sigma_theta_bins: float,
) -> np.ndarray:
    """
    Smooth a 2D spherical histogram.

    counts shape should be z_bins x theta_bins.
    The theta dimension is periodic. The z dimension is not.
    """
    counts = np.asarray(counts, dtype=float)

    if (
        sigma_z_bins is None
        or sigma_theta_bins is None
        or sigma_z_bins <= 0
        or sigma_theta_bins <= 0
    ):
        return counts

    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(
            counts,
            sigma=(sigma_z_bins, sigma_theta_bins),
            mode=("nearest", "wrap"),
        )
    except Exception:
        # Fallback: no smoothing if scipy is unavailable.
        return counts


def compute_3d_direction_density(
    Y3: np.ndarray,
    *,
    theta_bins: int = 72,
    z_bins: int = 36,
    smooth_sigma_theta_bins: float = 1.5,
    smooth_sigma_z_bins: float = 1.0,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """
    Compute direction-only density for 3D vectors on the unit sphere.

    Magnitudes are ignored. Each vector is normalized to the unit sphere.
    Density is estimated using an equal-area-ish spherical binning:
        theta = atan2(y, x) in [0, 2pi)
        z = normalized z-coordinate in [-1, 1]

    Uniform bins in theta and z correspond to equal-area patches on a sphere.

    Parameters
    ----------
    Y3:
        PCA result of shape 3 x n.
    theta_bins:
        Number of azimuth bins.
    z_bins:
        Number of z-coordinate bins.
    smooth_sigma_theta_bins:
        Gaussian smoothing width along azimuth bins.
    smooth_sigma_z_bins:
        Gaussian smoothing width along z bins.
    eps:
        Minimum norm for a vector to be considered valid.

    Returns
    -------
    dict with:
        "unit_vectors":
            Array of shape 3 x n_valid.
        "theta":
            Azimuth angle per valid vector.
        "z":
            z-coordinate per valid vector.
        "density":
            Density value per valid vector.
        "hist_density":
            Smoothed 2D histogram with shape z_bins x theta_bins.
        "theta_edges":
            Bin edges for theta.
        "z_edges":
            Bin edges for z.
        "valid_mask":
            Boolean mask indicating which original columns were used.
    """
    Y3 = np.asarray(Y3, dtype=float)

    if Y3.ndim != 2 or Y3.shape[0] != 3:
        raise ValueError(f"Y3 must have shape 3 x n. Got {Y3.shape}.")

    Y3_unit, valid = _normalize_columns(Y3, eps=eps)
    U = Y3_unit[:, valid]

    if U.shape[1] == 0:
        raise ValueError("No nonzero 3D vectors found after PCA projection.")

    x, y, z = U[0], U[1], U[2]
    theta = np.mod(np.arctan2(y, x), 2 * np.pi)

    theta_edges = np.linspace(0, 2 * np.pi, theta_bins + 1)
    z_edges = np.linspace(-1.0, 1.0, z_bins + 1)

    counts, _, _ = np.histogram2d(
        z,
        theta,
        bins=[z_edges, theta_edges],
    )

    hist_density = _smooth_spherical_histogram(
        counts,
        sigma_z_bins=smooth_sigma_z_bins,
        sigma_theta_bins=smooth_sigma_theta_bins,
    )

    z_idx = np.searchsorted(z_edges, z, side="right") - 1
    theta_idx = np.searchsorted(theta_edges, theta, side="right") - 1

    z_idx = np.clip(z_idx, 0, z_bins - 1)
    theta_idx = np.clip(theta_idx, 0, theta_bins - 1)

    density = hist_density[z_idx, theta_idx]

    return {
        "unit_vectors": U,
        "theta": theta,
        "z": z,
        "density": density,
        "hist_density": hist_density,
        "theta_edges": theta_edges,
        "z_edges": z_edges,
        "valid_mask": valid,
    }


def plot_3d_direction_density_interactive(
    Y3: np.ndarray,
    *,
    theta_bins: int = 72,
    z_bins: int = 36,
    smooth_sigma_theta_bins: float = 1.5,
    smooth_sigma_z_bins: float = 1.0,
    marker_size: float = 4,
    opacity: float = 0.85,
    alpha_vector: Optional[np.ndarray] = None,
    colorscale: str = "Viridis",
    title: str = "3D PCA Direction Density",
    show_unit_sphere: bool = True,
):
    """
    Create an interactive 3D unit-sphere plot colored by direction-only density.

    Magnitude is ignored. All valid vectors are normalized to the unit sphere.
    Direction density is estimated using binned spherical coordinates.

    Parameters
    ----------
    Y3:
        PCA result of shape 3 x n.
    theta_bins:
        Number of azimuth bins.
    z_bins:
        Number of z-coordinate bins.
    smooth_sigma_theta_bins:
        Gaussian smoothing width along azimuth bins.
    smooth_sigma_z_bins:
        Gaussian smoothing width along z bins.
    marker_size:
        Plotly marker size.
    opacity:
        Scalar marker opacity used when alpha_vector is not provided.
    alpha_vector:
        Optional vector of length n, where n is the number of columns in Y3.
        This vector is min-max scaled to [0, 1], then used as the per-point alpha
        value for each corresponding vector.
    colorscale:
        Plotly colorscale.
    title:
        Plot title.
    show_unit_sphere:
        If True, adds a translucent unit sphere surface.

    Returns
    -------
    fig, density_info
        Plotly Figure and density data dictionary.
    """
    import plotly.graph_objects as go

    density_info = compute_3d_direction_density(
        Y3,
        theta_bins=theta_bins,
        z_bins=z_bins,
        smooth_sigma_theta_bins=smooth_sigma_theta_bins,
        smooth_sigma_z_bins=smooth_sigma_z_bins,
    )

    U = density_info["unit_vectors"]
    density = density_info["density"]
    valid_mask = density_info["valid_mask"]

    fig = go.Figure()

    if show_unit_sphere:
        # Low-resolution translucent sphere for context.
        u = np.linspace(0, 2 * np.pi, 48)
        v = np.linspace(0, np.pi, 24)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))

        fig.add_trace(
            go.Surface(
                x=xs,
                y=ys,
                z=zs,
                opacity=0.12,
                showscale=False,
                colorscale=[[0, "lightgray"], [1, "lightgray"]],
                hoverinfo="skip",
                name="unit sphere",
            )
        )

    if alpha_vector is None:
        marker = dict(
            size=marker_size,
            color=density,
            colorscale=colorscale,
            opacity=opacity,
            colorbar=dict(title="Direction density"),
        )
    else:
        per_point_alpha = scale_alpha_vector(
            alpha_vector,
            expected_length=Y3.shape[1],
            valid_mask=valid_mask,
        )

        rgba_colors = _plotly_rgba_colors_from_values(
            density,
            per_point_alpha,
            colorscale=colorscale,
        )

        marker = dict(
            size=marker_size,
            color=rgba_colors,
        )

        density_info["alpha"] = per_point_alpha

    fig.add_trace(
        go.Scatter3d(
            x=U[0],
            y=U[1],
            z=U[2],
            mode="markers",
            marker=marker,
            text=[
                f"density={d:.3g}<br>"
                f"alpha={a:.3g}<br>"
                f"x={x:.3f}<br>"
                f"y={y:.3f}<br>"
                f"z={z:.3f}"
                for x, y, z, d, a in zip(
                    U[0],
                    U[1],
                    U[2],
                    density,
                    density_info.get("alpha", np.full_like(density, opacity)),
                )
            ],
            hoverinfo="text",
            name="vectors",
        )
    )

    if alpha_vector is not None:
        # Add an invisible numeric-color trace solely to preserve a density colorbar.
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(
                    size=0,
                    color=[np.min(density), np.max(density)],
                    colorscale=colorscale,
                    cmin=np.min(density),
                    cmax=np.max(density),
                    showscale=True,
                    colorbar=dict(title="Direction density"),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Normalized PC1",
            yaxis_title="Normalized PC2",
            zaxis_title="Normalized PC3",
            aspectmode="cube",
            xaxis=dict(range=[-1.05, 1.05]),
            yaxis=dict(range=[-1.05, 1.05]),
            zaxis=dict(range=[-1.05, 1.05]),
        ),
        margin=dict(l=0, r=0, b=0, t=45),
    )

    return fig, density_info


if __name__ == "__main__":
    """
    Example usage with synthetic data.

    Run:
        python pca_direction_density.py
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)

    # Example m x n matrix: m features, n samples.
    m = 50
    n = 5000

    # Create random data with a few latent directional clusters.
    latent = rng.normal(size=(5, n))
    mixing = rng.normal(size=(m, 5))
    X = mixing @ latent + 0.15 * rng.normal(size=(m, n))

    # PCA projection.
    Y2, Y3, pca_model = pca_project_to_2d_and_3d(
        X,
        center=True,
        svd_solver="randomized",
        random_state=42,
    )

    print("Y2 shape:", Y2.shape)
    print("Y3 shape:", Y3.shape)
    print("Explained variance ratio:", pca_model.explained_variance_ratio_)

    # 2D plot.
    fig2, ax2, info2 = plot_2d_direction_density_circle(
        Y2,
        bins=180,
        smooth_sigma_bins=2.0,
        title="2D Direction Density on Unit Circle",
    )
    plt.show()

    # 3D interactive plot.
    fig3, info3 = plot_3d_direction_density_interactive(
        Y3,
        theta_bins=72,
        z_bins=36,
        title="3D Direction Density on Unit Sphere",
    )

    # In a notebook, use:
    # fig3.show()

    # Or save to HTML:
    fig3.write_html("pca_3d_direction_density.html")
    print("Saved interactive 3D plot to pca_3d_direction_density.html")

if __name__ == "__main__":
    rng = np.random.default_rng()
    X = rng.normal(loc=0.0, scale=1.0, size=(1000, 25))
    X[:, :3] *= 20
    X[:, 3:6] *= 0.2
    pca_project_rows(X)