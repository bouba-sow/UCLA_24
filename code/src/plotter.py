import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.ndimage import gaussian_filter1d
from nilearn import plotting
from mni_to_atlas import AtlasBrowser
import os
from joblib import Parallel, delayed
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from nilearn import datasets, surface
from plotly.subplots import make_subplots


class Plotter:
    """
    Unified plotting interface for intracranial EEG analyses.

    Provides publication-ready visualizations for electrode localization, trial-level heatmaps,
    median signal summaries, encoding feature importance, decoding performance, and whole-brain
    searchlight outputs. The class supports both single-electrode figure composition and
    parallel batch rendering across electrodes.

    The plotter provides:
    - **Single electrode analysis**: Multi-panel layouts combining brain localization,
      neural heatmaps, feature importance, and performance metrics
        - **Flexible heatmap sorting**: Trial sorting by a primary key plus optional secondary
            tie-break key, with y-axis labels reflecting grouped sorting categories
    - **Encoding visualization**: Feature importance over time with encoding performance
    - **Decoding visualization**: Classification accuracy/AUC timecourses with chance lines
    - **Brain localization**: Glass brain views showing electrode positions and highlighted contacts
    - **Searchlight results**: 3D interactive plots and brain surface projections
    - **Flexible layouts**: Configurable multi-panel figures via plot_configs

    All plotting methods integrate seamlessly with results from Encoder, Decoder, and
    Searchlight classes.

    Parameters
    ----------
    epochs : xarray.DataArray
        Epoched neural data containing electrode coordinates (x, y, z in MNI space) and
        experimental metadata. Used for extracting electrode locations and trial information.
    save_dir : str, optional
        Default output directory for saving figures. Can be overridden in individual methods.
        If None, figures are displayed interactively instead of saved.

    Attributes
    ----------
    epochs : xarray.DataArray
        Reference to input epochs data.
    save_dir : str or None
        Default save directory.

    Examples
    --------
    **Single electrode with encoding results:**

    >>> from plotter import Plotter
    >>> from encoder import Encoder
    >>> from feature_builder import FeatureBuilder
    >>>
    >>> # Run encoding analysis
    >>> feature_builder = FeatureBuilder(analysis='encoder', experiment='houston')
    >>> encoder = Encoder(
    ...     alphas=np.logspace(-3, 3, 7),
    ...     times=epochs.time.values,
    ...     feature_names=['congruency', 'violation', 'number'],
    ...     window_size=0.2,
    ...     step_size=0.05,
    ...     metrics=('spearman',)
    ... )
    >>> all_results = encoder.fit_predict_electrodes(
    ...     epochs=epochs,
    ...     feature_builder=feature_builder,
    ...     electrodes=['LAH1', 'LAH2'],
    ...     compute_fi=True
    ... )
    >>>
    >>> # Create plotter
    >>> plotter = Plotter(epochs, save_dir='encoding_plots')
    >>>
    >>> # Plot single electrode with custom layout
    >>> plot_configs = [
    ...     {'type': 'glass_brain', 'height': 1},
    ...     {'type': 'encoding_fi', 'height': 3, 'kwargs': {
    ...         'results': all_results,
    ...         'feature_names': ['congruency', 'violation', 'number'],
    ...         'metric': 'spearman'
    ...     }}
    ... ]
    >>> plotter.single_electrode(
    ...     electrode='LAH1',
    ...     trials=[],
    ...     plot_configs=plot_configs,
    ...     results_dict=all_results
    ... )

    **Multiple electrodes with batch processing:**

    >>> # Process all electrodes in parallel
    >>> plotter.all_electrodes(
    ...     trials=[],
    ...     electrodes=['LAH1', 'LAH2', 'LAH3'],  # or 'all' for all electrodes
    ...     save_dir='encoding_plots',
    ...     plot_configs=plot_configs,
    ...     results_dict=all_results,
    ...     n_jobs=-1  # parallel execution
    ... )

    **Decoding visualization:**

    >>> from decoder import Decoder
    >>>
    >>> # Run decoding
    >>> decoder = Decoder(
    ...     alphas=np.logspace(-3, 3, 7),
    ...     times=epochs.time.values,
    ...     window_size=0.2,
    ...     step_size=0.05,
    ...     metrics=('accuracy', 'auc')
    ... )
    >>> results = decoder.fit(X, y)
    >>> results = decoder.predict(results)
    >>>
    >>> # Plot decoding performance
    >>> plotter = Plotter(epochs)
    >>> fig, ax = plt.subplots(figsize=(12, 5))
    >>> plotter.decoding(
    ...     results=results,
    ...     feature_name='violation',
    ...     ax=ax
    ... )
    >>> plt.show()

    **Custom multi-panel layout with heatmaps:**

    >>> plotter = Plotter(epochs, save_dir='custom_plots')
    >>>
    >>> # Define layout: glass brain + 2 heatmaps + median signals
    >>> plot_configs = [
    ...     {'type': 'glass_brain', 'height': 1},
    ...     {'type': 'heatmap', 'height': 2, 'kwargs': {
    ...         'first_key': 'Embedding',
    ...         'second_key': 'violation',
    ...         'third_key': 'number',
    ...         'option': 'PP',
    ...         'title': 'PP Embedding (sorted by violation + number)'
    ...     }},
    ...     {'type': 'heatmap', 'height': 2, 'kwargs': {
    ...         'first_key': 'Embedding',
    ...         'second_key': 'violation',
    ...         'third_key': 'number',
    ...         'option': 'objRC',
    ...         'title': 'Object Relative Clause (sorted by violation + number)'
    ...     }},
    ...     {'type': 'median_signals', 'height': 1.5}
    ... ]
    >>>
    >>> plotter.single_electrode(
    ...     electrode='LAH1',
    ...     trials=[],  # no trials to exclude
    ...     freq='broadband',
    ...     plot_configs=plot_configs
    ... )

    **Searchlight 3D visualization:**

    >>> from searchlight import Searchlight
    >>>
    >>> # Run searchlight analysis
    >>> searchlight = Searchlight(
    ...     analyzer=decoder,
    ...     analysis_type='decoding',
    ...     epochs=epochs,
    ...     radius=20,
    ...     density=30
    ... )
    >>> sl_results = searchlight.fit(X, y)
    >>> sl_predictions = searchlight.predict(sl_results)
    >>>
    >>> # Interactive 3D plot with time animation
    >>> plotter = Plotter(epochs)
    >>> fig = plotter.plot_grids(
    ...     results=sl_predictions,
    ...     feature_name='violation',
    ...     metric='accuracy',
    ...     grid_opacity=0.8
    ... )
    >>> fig.write_html('searchlight_3d.html')

    **Glass brain with highlighted electrode:**

    >>> import matplotlib.pyplot as plt
    >>>
    >>> plotter = Plotter(epochs)
    >>> fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    >>> plotter.glass_brain(highlight='LAH1', axes=axes)
    >>> plt.savefig('electrode_location.png', dpi=150, bbox_inches='tight')

    **Standalone heatmap visualization:**

    >>> fig, ax = plt.subplots(figsize=(10, 6))
    >>> overall_median, median_by_key = plotter.heatmap(
    ...     first_key='Embedding',
    ...     second_key='violation',
    ...     third_key='number',
    ...     option_first_key='PP',
    ...     trials=[],
    ...     electrode='LAH1',
    ...     ax=ax,
    ...     title='Neural Response sorted by violation + number'
    ... )
    >>> # Y axis groups will appear as: "<violation> | <number>"
    >>> plt.tight_layout()
    >>> plt.show()

    Notes
    -----
    - **Layout system**: The `plot_configs` parameter enables flexible multi-panel figures.
      Each config dict specifies plot type, relative height, and type-specific kwargs.
    - **Automatic extraction**: When passing `results_dict` keyed by electrode name,
      the plotter automatically extracts electrode-specific results for single_electrode plots.
    - **Parallel processing**: `all_electrodes()` uses joblib for parallel execution.
      Set n_jobs=-1 to use all cores, or adjust based on memory constraints.
    - **Coordinate system**: Assumes MNI space (x, y, z) electrode coordinates in epochs.
      Glass brain and 3D visualizations use these coordinates for spatial mapping.
        - **Heatmap aggregation**: Automatically computes an overall median signal and
            per-group medians from the heatmap row grouping.
    - **Interactive vs static**: Plotly methods (plot_grids, scatter_electrodes_3d)
      create interactive HTML figures. Matplotlib methods create static publication figures.
    - **Atlas integration**: Glass brain method uses mni_to_atlas to automatically label
      highlighted electrode anatomical regions (requires AAL3 atlas).
    - **Memory considerations**: Batch processing many electrodes with complex layouts can
      be memory-intensive. Process in smaller batches if needed.
    - **DPI settings**: Default figure DPI is 150-300 for publication quality. Adjust in
      savefig calls if needed for different output sizes.

    See Also
    --------
    Encoder : Time-resolved encoding analysis
    Decoder : Time-resolved decoding analysis
    Searchlight : Whole-brain localization
    FeatureBuilder : Data preparation utilities

    Methods
    -------
    heatmap : Plot neural heatmap sorted by 2-3 experimental keys
    median_signals : Plot median neural signals across conditions
    glass_brain : Brain views with electrode locations
    encoding_FI : Feature importance and encoding performance
    decoding : Decoding performance timecourse
    layout : Build custom multi-panel electrode figures
    single_electrode : Complete single-electrode visualization
    all_electrodes : Batch process multiple electrodes
    add_brain_wireframe : Add 3D brain mesh to Plotly figures
    scatter_electrodes_3d : 3D electrode scatter with brain
    plot_grids : Animated 3D searchlight results
    """

    def __init__(
        self,
        epochs,
        save_dir=None,
    ):
        self.epochs = epochs
        self.save_dir = save_dir

    ########
    # SINGLE ELECTRODE PLOTTING
    ########

    def heatmap(
        self,
        first_key,
        second_key,
        third_key=None,
        option_first_key=None,
        trials="all",
        electrode=None,
        ax=None,
        title=None,
        draw_group_separators=True,
    ):
        dataset = np.squeeze(self.epochs.sel(channels=(self.epochs.ch_name == electrode)))
        if trials != "all":
            dataset = dataset.isel(trials=dataset.Base_condition.isin(trials))

        data = dataset.sel(trials=(dataset.coords[first_key] == option_first_key))
        sort_keys = [second_key]
        if third_key is not None:
            sort_keys.append(third_key)
        data = data.sortby(sort_keys)
        v_min = np.nanpercentile(data.values, 5)
        v_max = np.nanpercentile(data.values, 95)

        ext = [data.time.min(), data.time.max(), len(data.trials) - 0.5, -0.5]

        ax.imshow(data.values, aspect="auto", cmap="coolwarm", extent=ext, vmin=v_min, vmax=v_max)

        # Build y-axis groups from sorting keys so labels match the actual row ordering.
        second_vals = np.asarray(data[second_key].values)
        if third_key is not None:
            third_vals = np.asarray(data[third_key].values)
            row_labels = np.array([f"{s} | {t}" for s, t in zip(second_vals, third_vals)], dtype=object)
            ylabel = f"{second_key} | {third_key}"
        else:
            row_labels = second_vals.astype(object)
            ylabel = second_key

        change_points = np.where(row_labels[1:] != row_labels[:-1])[0] + 1
        starts = np.r_[0, change_points]
        ends = np.r_[change_points, len(row_labels)]
        ytick_positions = (starts + ends - 1) / 2
        ytick_labels = [str(row_labels[s]) for s in starts]

        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels)
        if draw_group_separators and len(change_points) > 0:
            ax.hlines(change_points - 0.5, data.time.min(), data.time.max(), colors="black", lw=1)

        xticks = np.arange(float(data.time.min()), float(data.time.max()), 0.5)
        for x in xticks:
            ax.axvline(x, color="black", linestyle="-", lw=1, alpha=1)

        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time")

        vals = np.asarray(data.values)
        median_signal = np.nanmedian(vals, axis=0)
        median_by_second_key = {}
        unique_labels = np.unique(row_labels)
        for lab in unique_labels:
            m = row_labels == lab
            median_by_second_key[str(lab)] = np.nanmedian(vals[m, :], axis=0)

        return median_signal, median_by_second_key

    def _label_matches_category(self, label, category):
        l = str(label).strip().lower()
        c = str(category).strip().lower()
        return (l == c) or l.endswith(f"| {c}") or l.endswith(f"|{c}")

    def median_signals(
        self,
        median_signals,
        time=None,
        labels=None,
        ax=None,
        title="Median signals",
        xlim=None,
        xticks=None,
        smooth_sigma=0.0,
        smooth_mode="nearest",
        contrast_overlays=None,
        electrode=None,
        base_lw=2.0,
        bold_lw=3.8,
        shade_alpha=0.12,
        plot_differences=False,
        differences=None,
        plot_base_signals=True,
    ):
        if isinstance(median_signals, dict):
            labels = list(median_signals.keys())
            signals = list(median_signals.values())
        else:
            signals = median_signals if isinstance(median_signals, (list, tuple)) else [median_signals]
            if labels is None:
                labels = [f"signal_{i + 1}" for i in range(len(signals))]

        if time is None:
            time = np.asarray(self.epochs.time.values)

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 3))

        base_labels = list(labels)
        base_signals = [np.asarray(sig, dtype=float) for sig in signals]

        # Optional pairwise differences added as extra traces.
        if plot_differences:
            if differences is None:
                differences = []
            label_to_signal = {str(lab): np.asarray(sig, dtype=float) for lab, sig in zip(base_labels, base_signals)}
            for diff in differences:
                if isinstance(diff, (tuple, list)) and len(diff) == 2:
                    minuend, subtrahend = str(diff[0]), str(diff[1])
                elif isinstance(diff, str) and "-" in diff:
                    parts = diff.split("-", 1)
                    minuend, subtrahend = parts[0].strip(), parts[1].strip()
                else:
                    raise ValueError(
                        "Each item in differences must be a 2-tuple/list like "
                        "('A','B') or a string like 'A - B'."
                    )

                if minuend not in label_to_signal or subtrahend not in label_to_signal:
                    raise ValueError(
                        f"Cannot compute difference '{minuend} - {subtrahend}'. "
                        f"Available labels: {list(label_to_signal.keys())}"
                    )

                labels.append(f"{minuend} - {subtrahend}")
                signals.append(label_to_signal[minuend] - label_to_signal[subtrahend])

        if not plot_base_signals:
            if not (plot_differences and differences):
                raise ValueError(
                    "plot_base_signals=False requires plot_differences=True and a non-empty differences list."
                )
            n_base = len(base_labels)
            labels = labels[n_base:]
            signals = signals[n_base:]

        line_handles = {}
        for sig, lab in zip(signals, labels):
            y = np.asarray(sig, dtype=float)
            if smooth_sigma and smooth_sigma > 0:
                y = gaussian_filter1d(y, sigma=smooth_sigma, mode=smooth_mode)
            line = ax.plot(time, y, lw=base_lw, label=lab)[0]
            line_handles[str(lab)] = (line, y)

        if contrast_overlays:
            for ov in contrast_overlays:
                color = ov.get("color", "gray")
                categories = ov.get("categories", [])

                # Prefer electrode-specific windows when available
                ewin = ov.get("electrode_windows", {}) if isinstance(ov.get("electrode_windows", {}), dict) else {}
                if electrode is not None and electrode in ewin:
                    windows = ewin[electrode]
                else:
                    windows = ov.get("global_windows", ov.get("windows", []))

                for (t0, t1) in windows:
                    ax.axvspan(t0, t1, color=color, alpha=shade_alpha, lw=0, zorder=0)

                # Re-draw involved category traces in bold
                for cat in categories:
                    for lab in labels:
                        if self._label_matches_category(lab, cat):
                            base_line, y = line_handles[str(lab)]

                            for (t0, t1) in windows:
                                m = (time >= t0) & (time <= t1)
                                if np.any(m):
                                    ax.plot(
                                        time[m],
                                        y[m],
                                        lw=bold_lw,
                                        color=base_line.get_color(),
                                        alpha=0.98,
                                        zorder=5,
                                    )

        ax.axvline(0, color="k", ls="--", lw=1, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Median amplitude")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if xticks is not None:
            ax.set_xticks(xticks)
        if len(signals) > 0:
            ax.legend(frameon=False)
        return ax

    def glass_brain(self, highlight=None, axes=None):
        # ---- Extract electrode coordinates ----
        coords = np.vstack(
            [
                self.epochs.coords["x"].values,
                self.epochs.coords["y"].values,
                self.epochs.coords["z"].values,
            ]
        ).T

        # ---- Brain views ----
        views = ["l", "r", "z", "y"]
        titles = ["Left Sagittal", "Right Sagittal", "Axial", "Coronal"]

        # ---- Create axes if not provided ----
        if axes is None:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            show_fig = True
        else:
            fig = axes[0].figure
            show_fig = False

        # ---- Find region for highlighted electrode ----
        if highlight is not None:
            idx = np.where(self.epochs.ch_name.values == highlight)[0]
            if len(idx) > 0:
                atlas = AtlasBrowser("AAL3")

                coord = np.array([coords[idx[0]]])
                region_name = atlas.find_regions(coord)[0]
            else:
                region_name = "Unknown"
        else:
            region_name = None

        # ---- Plot brains ----
        for ax, view, title in zip(axes, views, titles):
            display = plotting.plot_glass_brain(
                None, display_mode=view, axes=ax, colorbar=False, plot_abs=False
            )
            # all electrodes in gray
            display.add_markers(coords, marker_color="gray", marker_size=15, alpha=0.3)

            # highlighted electrode in red
            if highlight is not None and len(idx) > 0:
                display.add_markers([coords[idx[0]]], marker_color="red", marker_size=100, alpha=1)

            ax.set_title(title, fontsize=11, pad=3)

        if region_name is not None:
            y_top = axes[0].get_position().y1 + 0.02
            fig.text(0.5, y_top, f"{region_name}", ha="center", fontsize=14, fontweight="bold")
        if show_fig:
            plt.tight_layout()
            plt.show()

    ########
    # ENCODER PLOTTING
    ########
    def encoding_FI(
        self,
        results=None,  # Single electrode results dict from encoder
        feature_names=None,
        electrode=None,
        ax=None,
        alpha_signif=0.05,
        metric="spearman",
        aggregate_pca_components=True,
        # Legacy parameters for backward compatibility
        all_results=None,
        fi_median=None,
        fi_signif=None,
        times=None,
    ):
        """
        Plot encoding feature importance and performance for a single electrode.
        Can accept either:
        - New format: results dict from encoder.fit_predict()
        - Legacy format: separate fi_median, fi_signif, all_results, times
        """
        # Handle new format
        if results is not None:
            all_results = {electrode: results}
            fi_median = results["fi"]["mean"]
            fi_signif = np.ones(fi_median.shape, dtype=bool)
            times = results["times"]
            if feature_names is None:
                feature_names = results.get("model_feature_names")

        if ax is None:
            fig, ax1 = plt.subplots(figsize=(12, 6))
        else:
            ax1 = ax
            fig = ax.get_figure()

        n_fi_features = fi_median.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i + 1}" for i in range(n_fi_features)]
        if len(feature_names) != n_fi_features:
            feature_names = [f"pca_component_{i + 1:02d}" for i in range(n_fi_features)]

        pca_indices = [i for i, fname in enumerate(feature_names) if str(fname).startswith("pca_component_")]
        plotted_indices = set()
        if aggregate_pca_components and pca_indices:
            pca_curve = fi_median[:, pca_indices].mean(axis=1)
            pca_signif = np.any(fi_signif[:, pca_indices].astype(bool), axis=1)
            if np.any(pca_signif) and np.any(pca_curve[pca_signif] != 0):
                ax1.plot(
                    times[pca_signif],
                    pca_curve[pca_signif],
                    label="all_pca_components_mean",
                )
                ax1.fill_between(
                    times[pca_signif],
                    pca_curve[pca_signif],
                    pca_curve[pca_signif],
                    alpha=0.2,
                )
            plotted_indices.update(pca_indices)

        for i, fname in enumerate(feature_names):
            if i in plotted_indices:
                continue
            signif_mask = fi_signif[:, i].astype(bool)
            if np.any(signif_mask) and np.any(fi_median[:, i][signif_mask] != 0):
                ax1.plot(times[signif_mask], fi_median[:, i][signif_mask], label=fname)
                ax1.fill_between(
                    times[signif_mask], fi_median[:, i][signif_mask], fi_median[:, i][signif_mask], alpha=0.2
                )

        # Overlay encoding performance for the selected electrode
        enc = all_results[electrode]
        encoding_median = enc["scores"][metric]["median"]
        encoding_std = enc["scores"][metric]["std"]

        ax2 = ax1.twinx()
        ax2.plot(
            times,
            encoding_median,
            color="black",
            linewidth=2.5,
            linestyle="--",
            label=f"{metric.capitalize()} ({electrode})",
        )
        ax2.fill_between(times, encoding_median - encoding_std, encoding_median + encoding_std, color="black", alpha=0.1)
        ax2.set_ylabel(f"{metric.capitalize()}")

        # --- Align zero of both y-axes ---
        y1_min, y1_max = ax1.get_ylim()
        y2_min, y2_max = ax2.get_ylim()
        y1_absmax = max(abs(y1_min), abs(y1_max))
        y2_absmax = max(abs(y2_min), abs(y2_max))
        ax1.set_ylim(-y1_absmax, y1_absmax)
        ax2.set_ylim(-y2_absmax, y2_absmax)

        # Legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=len(lines1 + lines2),
            frameon=False,
        )

        # Formatting
        ax1.set_xlabel("Time (s)")
        ax1.set_xlim(times[0], times[-1])
        ax1.set_ylabel("Permutation Importance")
        ax1.set_title(f"Feature Importance (p<{alpha_signif}) & Encoding – {electrode}")

        return fig, ax1, ax2

    ########
    # DECODER PLOTTING
    ########
    def decoding(self, results, feature_name, save_dir=None, alpha_signif=0.05, ax=None):
        """Plot decoding performance over time."""
        times = results["times"]
        output_dir = save_dir or self.save_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        created_ax = ax is None

        if created_ax:
            fig, ax = plt.subplots(figsize=(12, 5))
        else:
            fig = ax.get_figure()

        metric_avgs = []
        for metric, d in results["scores"].items():
            ax.plot(times, d["median"], label=f"{metric}")
            ax.fill_between(times, d["median"] - d["std"], d["median"] + d["std"], alpha=0.2)
            metric_avgs.append(f"{metric}: {d['median'].mean():.3f}")

        ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Chance")
        ax.set_xlabel("Time (s)")
        ax.set_xlim(times[0], times[-1])
        ax.set_ylabel("Score")
        ax.set_title(f"Decoding: {feature_name} | {', '.join(metric_avgs)}")
        ax.legend(frameon=False)

        if created_ax:
            plt.savefig(f"{output_dir}/decoder_{feature_name}.png", bbox_inches="tight", dpi=150)

        return fig

    ########
    # GENERAL SINGLE ELECTRODE PLOTTING
    ########

    def layout(self, electrode, trials, plot_configs, freq="broadband", results_dict=None):
        """
        Build a multi-panel layout for one electrode.

        Args:
            electrode: Channel name
            trials: Trials to exclude
            plot_configs: List of dicts with keys:
                - 'type': 'glass_brain', 'heatmap', 'median_signals', 'encoding_fi', 'decoding'
                - 'height': relative height in GridSpec
                - 'kwargs': dict of function-specific arguments
            freq: Frequency band label
            results_dict: Optional dict of results keyed by electrode. Automatically extracts electrode-specific data.

        Returns:
            fig: Matplotlib figure
        """
        n_rows = len(plot_configs)
        heights = [cfg.get("height", 1) for cfg in plot_configs]

        fig = plt.figure(figsize=(16, 4 * sum(heights)))
        fig.suptitle(f"Electrode: {electrode} - {freq}", fontsize=18, fontweight="bold", y=0.98)

        gs = GridSpec(n_rows, 1, height_ratios=heights, hspace=0.4)

        t_min, t_max = self.epochs.time.min().item(), self.epochs.time.max().item()
        median_signals = {}
        reference_ax = None

        for i, cfg in enumerate(plot_configs):
            plot_type = cfg["type"]
            kwargs = cfg.get("kwargs", {})

            # If results_dict is provided, extract electrode-specific results
            if results_dict is not None and "results" in kwargs:
                if isinstance(kwargs["results"], dict) and electrode in kwargs["results"]:
                    kwargs = {**kwargs, "results": kwargs["results"][electrode]}

            if plot_type == "glass_brain":
                gs_sub = gs[i].subgridspec(1, 4)
                axes = [fig.add_subplot(gs_sub[0, j]) for j in range(4)]
                self.glass_brain(highlight=electrode, axes=axes)

            elif plot_type == "heatmap":
                ax = fig.add_subplot(gs[i])
                overall_median, median_by_second_key = self.heatmap(
                    first_key=kwargs.get("first_key", "Embedding"),
                    second_key=kwargs.get("second_key", "violation"),
                    third_key=kwargs.get("third_key", None),
                    option_first_key=kwargs["option"],
                    trials=trials,
                    electrode=electrode,
                    ax=ax,
                    title=kwargs.get("title", kwargs["option"]),
                    draw_group_separators=kwargs.get("draw_group_separators", True),
                )
                if kwargs.get("hide_yticklabels", False):
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                if "ylabel" in kwargs:
                    ax.set_ylabel(kwargs["ylabel"])
                median_mode = kwargs.get("median_mode", "overall")  # overall | by_second_key | both
                prefix = kwargs.get("median_label_prefix", kwargs["option"])

                if median_mode in ("overall", "both"):
                    median_signals[prefix] = overall_median

                if median_mode in ("by_second_key", "both"):
                    for key_name, sig in median_by_second_key.items():
                        median_signals[f"{prefix} | {key_name}"] = sig
                ax.set_xlim(t_min, t_max)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                if reference_ax is None:
                    reference_ax = ax

            elif plot_type == "median_signals":
                ax = fig.add_subplot(gs[i])
                m_kwargs = cfg.get("kwargs", {})
                plot_data = median_signals

                prefix = m_kwargs.get("filter_prefix")
                if prefix:
                    plot_data = {k: v for k, v in median_signals.items() if k.startswith(prefix)}

                self.median_signals(
                    median_signals=plot_data,
                    ax=ax,
                    title=m_kwargs.get("title", "Median Signals"),
                    xticks=reference_ax.get_xticks() if reference_ax else None,
                    smooth_sigma=m_kwargs.get("smooth_sigma", 0.0),
                    smooth_mode=m_kwargs.get("smooth_mode", "nearest"),
                    contrast_overlays=m_kwargs.get("contrast_overlays", None),
                    electrode=electrode,
                    base_lw=m_kwargs.get("base_lw", 2.0),
                    bold_lw=m_kwargs.get("bold_lw", 3.8),
                    shade_alpha=m_kwargs.get("shade_alpha", 0.12),
                    plot_differences=m_kwargs.get("plot_differences", False),
                    differences=m_kwargs.get("differences", None),
                    plot_base_signals=m_kwargs.get("plot_base_signals", True),
                )

            elif plot_type == "encoding_fi":
                ax = fig.add_subplot(gs[i])
                self.encoding_FI(
                    results=kwargs["results"],
                    feature_names=kwargs["feature_names"],
                    electrode=electrode,
                    ax=ax,
                    alpha_signif=kwargs.get("alpha_signif", 0.05),
                    metric=kwargs.get("metric", "spearman"),
                )

            elif plot_type == "decoding":
                ax = fig.add_subplot(gs[i])
                self.decoding(
                    results=kwargs["results"],
                    feature_name=kwargs["feature_name"],
                    save_dir=kwargs.get("save_dir", None),
                    alpha_signif=kwargs.get("alpha_signif", 0.05),
                    ax=ax,
                )

        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        return fig

    def single_electrode(self, electrode, trials, save_dir=None, freq="broadband", plot_configs=None, results_dict=None):
        """Plot single electrode with configurable layout.

        Args:
            electrode: Electrode name
            trials: Trials to exclude
            save_dir: Output directory
            freq: Frequency band label
            plot_configs: List of plot configuration dicts
            results_dict: Optional dict of results keyed by electrode (automatically extracts electrode-specific data)
        """
        if plot_configs is None:
            # Default layout
            plot_configs = [
                {"type": "glass_brain", "height": 1},
                {"type": "heatmap", "height": 2, "kwargs": {"option": "PP"}},
                {"type": "heatmap", "height": 2, "kwargs": {"option": "objRC"}},
                {"type": "median_signals", "height": 1.5},
            ]

        fig = self.layout(electrode, trials, plot_configs, freq, results_dict=results_dict)

        output_dir = save_dir or self.save_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, f"{electrode}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig)
        else:
            plt.show()

        return None

    def all_electrodes(
        self, trials, electrodes="all", save_dir=None, freq="broadband", plot_configs=None, results_dict=None, n_jobs=-1
    ):
        """
        Plot multiple electrodes in parallel with custom layout.

        Args:
            trials: Trials to exclude
            electrodes: 'all' for all electrodes, or list of electrode names
            save_dir: Output directory
            freq: Frequency band label
            plot_configs: List of plot configuration dicts
            results_dict: Optional dict of results keyed by electrode (e.g., from encoder.fit_electrodes())
            n_jobs: Number of parallel jobs
        """
        output_dir = save_dir or self.save_dir
        if not output_dir:
            raise ValueError("save_dir required")

        os.makedirs(output_dir, exist_ok=True)

        # Handle electrodes parameter
        if electrodes == "all":
            electrode_list = self.epochs.ch_name.values
        else:
            electrode_list = list(electrodes)

        Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self.single_electrode)(
                electrode=elec,
                trials=trials,
                save_dir=output_dir,
                freq=freq,
                plot_configs=plot_configs,
                results_dict=results_dict,
            )
            for elec in electrode_list
        )

        print(f"\nCompleted: saved {len(electrode_list)} electrodes to {output_dir}")

    ########
    # BRAIN PLOTTING
    ########

    def add_brain_wireframe(self, fig, opacity=0.1, edge_step=3, edge_width=3, scene_id=None):
        """Add a wireframe fsaverage brain mesh to a Plotly figure.

        Uses line segments instead of solid mesh which prevents hover interactions.
        (c.f. plotly bug: https://github.com/plotly/plotly.js/issues/6669)

        Args:
            fig: Plotly Figure to add the wireframe to.
            opacity: Line transparency (0=invisible, 1=opaque). Default 0.1.
            edge_step: Subsample factor for edges (higher=sparser). Default 3.
            edge_width: Width of wireframe lines. Default 3.

        Returns:
            The modified Plotly Figure with brain wireframe added.
        """
        fsaverage = datasets.fetch_surf_fsaverage()
        for hemi in [fsaverage.pial_left, fsaverage.pial_right]:
            coords, faces = surface.load_surf_data(hemi)
            # Build unique edges from faces (avoid duplicates)
            edges = set()
            for face in faces:
                for j in range(3):
                    edge = tuple(sorted([face[j], face[(j + 1) % 3]]))
                    edges.add(edge)
            edges = list(edges)[::edge_step]  # Sample edges

            # Build line segments with None separators
            x, y, z = [], [], []
            for v1, v2 in edges:
                x.extend([coords[v1, 0], coords[v2, 0], None])
                y.extend([coords[v1, 1], coords[v2, 1], None])
                z.extend([coords[v1, 2], coords[v2, 2], None])

            if scene_id is None:
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="lines",
                        line=dict(color="black", width=edge_width),
                        opacity=opacity,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="lines",
                        line=dict(color="black", width=edge_width),
                        opacity=opacity,
                        hoverinfo="skip",
                        showlegend=False,
                        scene=scene_id,
                    )
                )

        fig.update_layout(
            scene=dict(
                xaxis_visible=False,
                yaxis_visible=False,
                zaxis_visible=False,
                aspectmode="data",
            )
        )

        return fig

    def scatter_electrodes_3d(
        self,
        df: pd.DataFrame,
        marker_size: int = 4,
        wireframe_kwargs: dict = {},
        **scatter_kwargs,
    ) -> go.Figure:
        """Create a 3D scatter plot of electrodes overlaid on a brain wireframe.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing electrode coordinates with columns 'x', 'y', 'z'.
        marker_size: int, default 4
            Size of electrode markers.
        wireframe_kwargs: dict, default {}
            Keyword arguments passed to `add_brain_wireframe`.
        **scatter_kwargs: dict
            Additional keyword arguments passed to `plotly.express.scatter_3d`.

        Returns
        -------
        go.Figure
            Plotly figure with electrodes and brain wireframe.
        """
        fig = px.scatter_3d(df, x="x", y="y", z="z", **scatter_kwargs)
        fig = self.add_brain_wireframe(fig, **wireframe_kwargs)
        fig.update_traces(marker=dict(size=marker_size), selector=dict(type="scatter3d"))

        return fig

    def plot_grids(self, results, feature_name, metric, grid_opacity=1):
        """Interactive 3-D animated Plotly scatter of searchlight grid points.
        Electrode locations are shown as static black dots."""

        scores = results["scores"][metric]
        grid = results["grid"]
        neighbors = results["neighbors"]
        locations = results["locations"]

        idx = results.get("point_ids", sorted(list(neighbors.keys())))
        times = results.get("times", np.arange(scores.shape[1]) * 0.05 - 0.5)
        t_labels = [round(t, 2) for t in times]
        df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "x": grid[idx, 0],
                        "y": grid[idx, 1],
                        "z": grid[idx, 2],
                        "score": scores[:, t],
                        "time_window": t_labels[t],
                    }
                )
                for t in range(scores.shape[1])
            ],
            ignore_index=True,
        )
        fig = self.scatter_electrodes_3d(
            df,
            marker_size=8,
            color="score",
            hover_data={"score": ":.3f", "x": ":.1f", "y": ":.1f", "z": ":.1f"},
            animation_frame="time_window",
            color_continuous_scale="RdBu_r",
            range_color=[0.0, 1.0],
            title=f"{feature_name} - {metric}",
            wireframe_kwargs={"opacity": 0.05, "edge_step": 5},
        )
        fig.data[0].marker.opacity = grid_opacity
        for frame in fig.frames:
            for trace in frame.data:
                if hasattr(trace, "marker") and trace.marker is not None:
                    trace.marker.opacity = grid_opacity
        # Electrode locations
        fig.add_trace(
            go.Scatter3d(
                x=locations[:, 0],
                y=locations[:, 1],
                z=locations[:, 2],
                mode="markers",
                marker=dict(size=3, color="black", opacity=0.6),
                name="electrodes",
                hoverinfo="skip",
            )
        )
        annotation_text = f"Max {metric}={np.nanmax(scores):.3f} | " f"Median {metric}={np.nanmedian(scores):.3f}"
        fig.add_annotation(
            text=annotation_text,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            font=dict(size=12),
        )
        return fig

    def brain_enc_summary_plot(self, df_long):
        metrics = df_long["metric_type"].unique()  # Assuming 7 metrics
        times = sorted(df_long["time"].unique())
        n_rows, n_cols = 3, 3

        # --- 1. Fix the Subplot Titles ---
        # We create a list of 9 empty strings and place our titles exactly where they belong
        all_titles = [""] * 9
        all_titles[1] = "<b>ENCODING PERFORMANCE</b>"  # Row 1, Col 2
        for i, m in enumerate(metrics[1:]):  # Remaining 6 metrics
            title_text = f"<b>{m.replace('fi_', '').replace('_', ' ').upper()}</b>"
            all_titles[i + 3] = title_text  # Start from index 3 (Row 2, Col 1)

        # --- 2. Calculate Percentiles ---
        enc_data = df_long[df_long["metric_type"] == "encoding_score"]["score"]
        enc_lims = (enc_data.quantile(0.10), enc_data.quantile(0.90))
        feat_data = df_long[df_long["metric_type"] != "encoding_score"]["score"]
        feat_lims = (feat_data.quantile(0.10), feat_data.quantile(0.90))

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            specs=[[{"type": "scene"}] * n_cols] * n_rows,
            subplot_titles=all_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.03,
        )

        electrode_indices = []
        occupied_scenes = []

        # --- 3. Precise Placement Loop ---
        for i, metric in enumerate(metrics):
            if i == 0:
                row, col = 1, 2
            elif 1 <= i <= 3:
                row, col = 2, i
            else:
                row, col = 3, i - 3

            scene_num = (row - 1) * n_cols + col
            scene_id = f"scene{scene_num}" if scene_num > 1 else "scene"
            occupied_scenes.append(scene_id)

            self.add_brain_wireframe(fig, opacity=0.04, edge_step=8, scene_id=scene_id)

            df_t0 = df_long[(df_long["metric_type"] == metric) & (df_long["time"] == times[0])]
            c_axis = "coloraxis" if i == 0 else "coloraxis2"
            label_name = "Encoding" if i == 0 else "Importance"

            fig.add_trace(
                go.Scatter3d(
                    x=df_t0["x"],
                    y=df_t0["y"],
                    z=df_t0["z"],
                    mode="markers",
                    marker=dict(size=6, color=df_t0["score"], coloraxis=c_axis),
                    text=df_t0["electrode"],
                    hovertemplate=(
                        "<b>Electrode: %{text}</b><br>" + f"{label_name}: %{{marker.color:.3f}}<extra></extra>"
                    ),
                    scene=scene_id,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            electrode_indices.append(len(fig.data) - 1)

        # --- 4. Animation Frames ---
        frames = []
        for t in times:
            frame_data = []
            for metric in metrics:
                df_t = df_long[(df_long["metric_type"] == metric) & (df_long["time"] == t)]
                frame_data.append(go.Scatter3d(marker=dict(color=df_t["score"])))
            frames.append(go.Frame(data=frame_data, name=str(t), traces=electrode_indices))
        fig.frames = frames

        # --- 5. Clean Up Layout & Empty Cells ---
        cam_lat = dict(eye=dict(x=1.6, y=0, z=0.5))
        cam_sup = dict(eye=dict(x=0, y=0, z=1.8))

        for s_num in range(1, 10):
            s_id = f"scene{s_num}" if s_num > 1 else "scene"
            if s_id in occupied_scenes:
                fig.layout[s_id].update(
                    camera=cam_lat, xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode="data"
                )
            else:
                # Fully hide unused grid cells
                fig.layout[s_id].update(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)

        fig.update_layout(
            height=1200,
            width=1400,
            margin=dict(l=50, r=150, t=100, b=50),
            coloraxis=dict(
                colorscale="Greens",
                cmin=enc_lims[0],
                cmax=enc_lims[1],
                colorbar=dict(title="Encoding", x=1.02, len=0.3, y=0.8, thickness=20),
            ),
            coloraxis2=dict(
                colorscale="Blues",
                cmin=feat_lims[0],
                cmax=feat_lims[1],
                colorbar=dict(title="FI Importance", x=1.02, len=0.3, y=0.3, thickness=20),
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0,
                    y=1.05,
                    buttons=[
                        dict(
                            label="Lateral View",
                            method="relayout",
                            args=[{f"{s}.camera": cam_lat for s in occupied_scenes}],
                        ),
                        dict(
                            label="Superior View",
                            method="relayout",
                            args=[{f"{s}.camera": cam_sup for s in occupied_scenes}],
                        ),
                    ],
                ),
                dict(
                    type="buttons",
                    direction="right",
                    x=0,
                    y=-0.05,
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 50, "redraw": True}}],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0}}],
                        ),
                    ],
                ),
            ],
            sliders=[
                dict(
                    active=0,
                    x=0.15,
                    y=-0.05,
                    len=0.85,
                    steps=[
                        dict(args=[[f.name], {"frame": {"duration": 0, "redraw": True}}], label=f.name, method="animate")
                        for f in frames
                    ],
                )
            ],
        )

        return fig
