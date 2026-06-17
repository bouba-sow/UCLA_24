## UCLA_24 BIDS Additions Summary

This repository's BIDS dataset (`data/bids`) includes a frame-wise enriched `events.tsv` table for `task-movie24presleep` with multimodal linguistic, acoustic, visual, and concept regressors.

### Core event timing and labels

- `onset`, `duration`: Event timing aligned to iEEG recording.
- `Event`: Frame-level labels with levels:
  - `first_word_onset`
  - `word_onset`
  - `j_bauer` (character onset transition)
  - `no_event`
- `frame`, `Time`: Frame index and movie-relative time.

### Linguistic / phonological columns

- `vowel_onset`, `word_onset`
- `vowel_duration`, `word_duration`
- `word_frequency` (Zipf)
- `word_ner` (named entity label)
- `pause_duration_ms`
- `word_char_len`

### Acoustic columns

- `env`
- `env_peak_rate`
- `pitch_hz`
- `pitch_norm`
- `pitch_up`
- `pitch_down`
- `mel_00` to `mel_15` (16 normalized log-mel bins)

### Visual character presence columns (frame-wise)

- `char_a_amar`
- `char_a_fayed`
- `char_b_buchanan`
- `char_c_manning`
- `char_c_obrian`
- `char_j_bauer`
- `char_j_wallace`
- `char_k_hayes`
- `char_m_obrian`
- `char_m_pressman`
- `char_n_yassir`
- `char_r_wallace`
- `char_s_wallace`
- `char_t_lennox`
- `char_w_palmer`
- `char_face`
- `char_person`
- `char_no_characters`

### High-level concept columns

- `concept_whitehouse`
- `concept_ctu`
- `concept_hostage`
- `concept_handcuff`
- `concept_j_bauer`
- `concept_b_buchanan`
- `concept_a_fayed`
- `concept_a_amar`

### Visual features (frame-wise): DINOv3

- `dinov3_npy_path` (path reference to frame-wise DINOv3 visual feature embeddings).

### Other BIDS metadata generated in this dataset

- iEEG metadata sidecars for recording/channel counts and sampling info.
- Electrode and coordinate metadata with MNI coordinate system (`MNI152NLin6ASym`).

---

## Spike sorting data — subject 572

### Raw wave_clus output
`data/ucla_data/572/Experiment-8-9-10-11/CSC_micro_spikes_removePLI-0_CAR-1_rejectNoiseSpikes-1/`

The folder name encodes preprocessing: PLI removal = OFF, Common Average Re-reference = ON, noise spike rejection = ON. Three file types per channel:

#### `{CH}_spikes.mat` — spike detection
Output of the first wave_clus step: voltage threshold detection on the bandpass-filtered signal.

| Field | Content |
|-------|---------|
| `spikes` | (74 × n_detected) — waveform snippets at 32 kHz. 74 samples = **2.34 ms**: `w_pre=23` samples (0.72 ms) before the threshold crossing, `w_post=51` samples after. |
| `spikeTimestamps` | (1 × n_detected) — timestamp of each detected waveform in series time (s from recording start) |
| `timestampsStart` | Unix timestamp of the recording start |
| `duration` | Total recording duration (s) |
| `param` | All wave_clus settings: `sr=32000 Hz`, bandpass `300–3000 Hz` (order 4), detection `"both"` (positive + negative crossings), refractory period `0.5 ms` |

#### `times_{CH}.mat` — spike sorting
Output of the second wave_clus step: waveform clustering via superparamagnetic clustering (SPC).

| Field | Content |
|-------|---------|
| `cluster_class` | (2 × n_sorted) — row 0 = cluster ID (0=noise, 1,2,…=isolated neurons), row 1 = spike time (ms) |
| `spikeIdxRejected` | (1 × n_detected) — boolean: which detected spikes were rejected during sorting |
| `inspk` | (10 × n_sorted) — wavelet features used for clustering (first 10 coefficients) |
| `forced` | Whether each spike assignment was set manually in the GUI |
| `Temp` | Superparamagnetic temperature used for the final clustering |

#### `{CH}_spikeCodes.mat` — pre-computed binned raster

| Field | Content |
|-------|---------|
| `spikeHist` | (1 × 2,661,322) — binary spike presence at **3 ms** bins (all clusters combined) |
| `spikeHistPrecise` | (1 × 127,743,430) — same at **0.0625 ms** bins (16 kHz resolution) |
| `spikeCodes` | MATLAB table (Fried Lab GUI internal, not directly readable in Python) |

#### `spc_log/{CH}_spikes_spc_log.txt`
Text log of the SPC temperature sweep and final cluster assignments. Useful for diagnosing channels with 0 sorted units.

---

### BIDS spike derivatives
`data/bids/derivatives/spike-sorted/sub-572/ses-01/ieeg/`

Four files per channel (64 microwire channels):

#### `*_spikewaveforms.npy`
(n_spikes × 74) — waveform snippets (µV) for all spikes matched to a sorted entry in `times_*.mat` (cluster 0 included). Pure numpy array.

#### `*_events.tsv` + `*_events.json`
Per-spike BIDS events table. One row per spike.

| Column | Content |
|--------|---------|
| `onset` | Spike time in iEEG recording clock (s), after drift correction |
| `duration` | Always 0 (instantaneous event) |
| `cluster_id` | wave_clus cluster ID (0=noise, 1+=isolated neuron) |
| `unit_class` | Fried Lab code: 1=single unit, 2=multi-unit, 3=noise |
| `series_onset` | Spike time in raw series clock (before drift correction) |
| `detection_index` | Index into the original `spikes.mat` waveform array |

#### `*_spikedata.npz` — self-contained analysis bundle
Everything needed for stimulus-aligned analyses, no other files required.

| Key | Shape | Content |
|-----|-------|---------|
| `spike_times_movie` | (n_spikes,) | Spike times in **movie frame time** (s) — use for stimulus-aligned analyses |
| `spike_times_recording` | (n_spikes,) | Spike times in iEEG clock (drift-corrected) |
| `spike_times_series` | (n_spikes,) | Raw series clock times (before drift correction) |
| `cluster_id` | (n_spikes,) | wave_clus cluster (0=noise, 1+=neuron) |
| `waveforms` | (n_spikes, 74) | Raw voltage snippets (µV), 74 samples = 2.34 ms |
| `firing_rate_counts` | (74378,) | Spike counts in 30 fps bins (~33 ms) over the full movie — all clusters including cluster 0 |
| `firing_rate_bin_edges` | (74379,) | Bin edge times (s) for firing rate histogram |
| `firing_rate_hz` | scalar | Bin rate = 30.0 Hz |
| `micro_movie_volts` | (2,479,236,) | Downsampled continuous microwire signal at **1000 Hz**, aligned to movie time |
| `micro_movie_times` | (2,479,236,) | Time axis for `micro_movie_volts` (s, movie frame time) |
| `micro_movie_downsample_hz` | scalar | 1000.0 Hz |
| `movie_duration_sec` | scalar | ~2479 s (~41 min) |
| `drift_correction_multiplier` | scalar | Audio-based drift factor aligning neural clock to video clock |
| `movie_start_rel` / `movie_start_series` | scalars | Movie onset in recording/series time |

> **Note on cluster 0:** `firing_rate_counts` in the NPZ includes all clusters (0+). For single-unit analyses, recompute the histogram from `spike_times_movie[cluster_id >= 1]`.
