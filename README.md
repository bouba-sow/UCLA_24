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
