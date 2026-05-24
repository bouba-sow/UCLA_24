"""
Feature extraction for 24_S06E01 (paper-aligned where possible).
- Biphone surprisal currently disabled (column set to 0)
- Envelope: Hilbert magnitude + 3rd-order Butterworth 1-10 Hz band-pass
- Pitch: torchcrepe with 100 ms smoothing
- Word frequency: wordfreq Zipf scores
"""
from __future__ import annotations

import sys
from bisect import bisect_right
from pathlib import Path

# Allow importing from code/src without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
try:
    import torchcrepe
except ImportError as exc:
    raise ImportError(
        "torchcrepe is required for pitch extraction. "
        "Install it with: uv pip install torchcrepe"
    ) from exc

import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, hilbert
from wordfreq import zipf_frequency

from utils import align, minmax01, normalize_token, onset_and_duration, parse_srt

# --- Paths ---
DATA_DIR = Path('/store/scratch/bsow/Documents/UCLA_24/data')
phoneme_csv = DATA_DIR / '24_S06E01_phonemes.csv'
word_csv    = DATA_DIR / '24_S06E01_words.csv'
srt_path    = DATA_DIR / '24_S06E01.srt'
audio_path  = DATA_DIR / '24_S06E01.wav'
out_csv     = DATA_DIR / '24_S06E01_events_vowel_word_features.csv'

# --- Audio constants ---
TARGET_SR          = 16_000
HOP_MS             = 10
WINDOW_MS          = 100
HOP_SAMPLES        = int(HOP_MS / 1000 * TARGET_SR)
N_FFT              = int(WINDOW_MS / 1000 * TARGET_SR)
N_MELS             = 16
ENV_BANDPASS_LOW_HZ  = 1.0
ENV_BANDPASS_HIGH_HZ = 10.0
PITCH_SMOOTH_MS    = 100
CREPE_CONF_THRESHOLD = 0.5


# =============================================================================
# Section 1 — Phonological / lexical tabular features
# =============================================================================

phon  = pd.read_csv(phoneme_csv)
words = pd.read_csv(word_csv)
subs  = parse_srt(srt_path)

if len(phon) != len(words):
    raise ValueError(f'Row mismatch: phonemes={len(phon)}, words={len(words)}')
if not np.allclose(phon['time'].to_numpy(), words['time'].to_numpy()):
    raise ValueError('Time columns not aligned between phoneme and word CSVs.')

frame_time = phon['time'].to_numpy(dtype=float)
frame_dt   = float(np.median(np.diff(frame_time)))

phoneme_cols  = [c for c in phon.columns if c not in ('frame', 'time')]
frame_phoneme = phon[phoneme_cols].idxmax(axis=1)

vowel_chars      = set('aeiouAEIOUæɑɒɔəɛɜɪʊʉɐɚ')
diphthongs       = {'aj', 'aw', 'ej', 'ow', 'əw', 'ɔj'}
non_speech_tokens = {'sil', 'spn'}
vowel_labels = {
    ph for ph in phoneme_cols
    if (ph in diphthongs or any(ch in vowel_chars for ch in ph))
    and ph not in non_speech_tokens
}

vowel_onset, vowel_duration = onset_and_duration(
    frame_phoneme, lambda x: isinstance(x, str) and x in vowel_labels, frame_time, frame_dt
)
phoneme_onset, phoneme_duration = onset_and_duration(
    frame_phoneme,
    lambda x: isinstance(x, str) and x not in non_speech_tokens and x != '',
    frame_time, frame_dt,
)

word_series = words['word']
word_onset, word_duration = onset_and_duration(
    word_series,
    lambda x: not pd.isna(x) and str(x).strip().lower() not in {'none', ''},
    frame_time, frame_dt,
)

word_onset_mask = word_onset != ''
onset_rows  = word_onset[word_onset_mask].index.to_numpy()
onset_words = word_onset[word_onset_mask].astype(str)

word_frequency = pd.Series(0.0, index=word_series.index, dtype=float)
for idx, token in onset_words.items():
    word_frequency.iat[idx] = float(zipf_frequency(token, 'en'))

word_char_len = pd.Series(0.0, index=word_series.index, dtype=float)
for idx in onset_rows:
    word_char_len.iat[idx] = len(str(word_onset.iat[idx]).strip())

pause_duration_ms = pd.Series(0.0, index=word_series.index, dtype=float)
for prev_idx, curr_idx in zip(onset_rows[:-1], onset_rows[1:]):
    prev_end = frame_time[prev_idx] + float(word_duration.iat[prev_idx])
    pause_duration_ms.iat[curr_idx] = max(0.0, frame_time[curr_idx] - prev_end) * 1000.0

# NER via spacy (gracefully skipped if unavailable)
word_ner = pd.Series('', index=word_series.index, dtype=object)
try:
    import spacy
    ner_nlp = spacy.load('en_core_web_sm')
    subtitle_entities = []
    for text in subs['text'].tolist():
        doc = ner_nlp(text)
        token_to_label = {}
        for ent in doc.ents:
            for tok in ent:
                key = normalize_token(tok.text)
                if key and key not in token_to_label:
                    token_to_label[key] = ent.label_
        subtitle_entities.append(token_to_label)

    sub_starts = subs['start'].to_numpy(dtype=float)
    sub_ends   = subs['end'].to_numpy(dtype=float)
    for idx in onset_rows:
        t     = float(frame_time[idx])
        token = normalize_token(onset_words.loc[idx])
        if not token:
            continue
        sub_i = bisect_right(sub_starts, t) - 1
        if sub_i < 0 or sub_i >= len(subs):
            continue
        if not (sub_starts[sub_i] <= t <= sub_ends[sub_i]):
            continue
        word_ner.iat[idx] = subtitle_entities[sub_i].get(token, '')
except Exception as exc:
    print(f'NER skipped: {exc}')
print('Finished NER')
sys.stdout.flush()

# Biphone surprisal disabled; column kept for schema stability.
biphone_surprisal = pd.Series(0.0, index=phon.index, dtype=float)
print('Biphone surprisal disabled: column set to 0.0 for all rows.')


# =============================================================================
# Section 2 — Audio features
# =============================================================================

print(f'Loading audio: {audio_path}')
y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
print(f'  Duration: {len(y)/sr:.1f}s  SR: {sr}Hz')

# Mel spectrogram
mel_power = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_SAMPLES,
    n_mels=N_MELS, fmin=0, fmax=8000, power=2.0,
)
mel_db = librosa.power_to_db(mel_power, ref=np.max)
mel_aligned = {}
for i in range(N_MELS):
    band_norm  = minmax01(mel_db[i])
    band_times = librosa.times_like(band_norm, sr=TARGET_SR, hop_length=HOP_SAMPLES)
    mel_aligned[f'mel_{i:02d}'] = align(band_norm, band_times, frame_time)

# Envelope: Hilbert magnitude → decimate → Butterworth 1-10 Hz band-pass
env_cont = np.abs(hilbert(y))
env_100  = env_cont[::HOP_SAMPLES]
env_fs   = TARGET_SR / HOP_SAMPLES  # 100 Hz
nyq  = 0.5 * env_fs
low  = max(0.01, min(ENV_BANDPASS_LOW_HZ,  nyq * 0.95))
high = max(low + 0.01, min(ENV_BANDPASS_HIGH_HZ, nyq * 0.99))
b, a = butter(3, [low / nyq, high / nyq], btype='band')
env_norm  = minmax01(filtfilt(b, a, env_100))
env_times = np.arange(len(env_norm), dtype=float) / env_fs
env_aligned = align(env_norm, env_times, frame_time)

# Envelope peak rate (from gradient peaks, no threshold)
env_d = np.gradient(env_norm)
peaks, _ = find_peaks(env_d)
rate_sparse = np.zeros_like(env_norm)
rate_sparse[peaks] = env_d[peaks]
env_peak_rate_aligned = align(rate_sparse, env_times, frame_time)

# Pitch via torchcrepe
device  = 'cuda' if torch.cuda.is_available() else 'cpu'
audio_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
print('Finished audio tensor')
sys.stdout.flush()

pitch_t, periodicity_t = torchcrepe.predict(
    audio_t, sr, HOP_SAMPLES,
    fmin=50, fmax=500, model='full', batch_size=1024, device=device,
    decoder=torchcrepe.decode.viterbi, return_periodicity=True,
)
pitch_hz   = pitch_t.squeeze(0).detach().cpu().numpy().astype(float)
confidence = periodicity_t.squeeze(0).detach().cpu().numpy().astype(float)
pitch_hz[confidence < CREPE_CONF_THRESHOLD] = 0.0

pitch_win    = max(1, int(round(PITCH_SMOOTH_MS / HOP_MS)))
pitch_smooth = np.convolve(pitch_hz, np.ones(pitch_win) / pitch_win, mode='same')

voiced_vals = pitch_smooth[pitch_smooth > 0]
if len(voiced_vals) > 0:
    lo5, hi95     = np.percentile(voiced_vals, 5), np.percentile(voiced_vals, 95)
    pitch_clipped = np.clip(pitch_smooth, lo5, hi95)
    pitch_clipped[pitch_smooth == 0] = 0.0
    pitch_norm = minmax01(pitch_clipped)
else:
    pitch_norm = np.zeros_like(pitch_smooth)

pitch_deriv = np.gradient(pitch_norm)
pitch_up    = np.maximum(pitch_deriv, 0.0)
pitch_down  = np.maximum(-pitch_deriv, 0.0)

pitch_times         = np.arange(len(pitch_smooth), dtype=float) * (HOP_SAMPLES / sr)
pitch_hz_aligned    = align(pitch_smooth, pitch_times, frame_time)
pitch_norm_aligned  = align(pitch_norm,   pitch_times, frame_time)
pitch_up_aligned    = align(pitch_up,     pitch_times, frame_time)
pitch_down_aligned  = align(pitch_down,   pitch_times, frame_time)

print('\nFeature lengths before alignment:')
print(f'  env_norm:   {len(env_norm)}')
print(f'  pitch_hz:   {len(pitch_smooth)}')
print(f'  mel band 0: {mel_db.shape[1]}')
print(f'  frame_time: {len(frame_time)}')


# =============================================================================
# Assemble and save output
# =============================================================================

out = pd.DataFrame({
    'frame':            phon['frame'],
    'time':             phon['time'],
    'vowel_onset':      vowel_onset,
    'vowel_duration':   vowel_duration,
    'phoneme_onset':    phoneme_onset,
    'phoneme_duration': phoneme_duration,
    'word_onset':       word_onset,
    'word_duration':    word_duration,
    'word_frequency':   word_frequency,
    'word_char_len':    word_char_len,
    'word_ner':         word_ner,
    'pause_duration_ms': pause_duration_ms,
    'biphone_surprisal': biphone_surprisal,
    'env':              env_aligned,
    'env_peak_rate':    env_peak_rate_aligned,
    'pitch_hz':         pitch_hz_aligned,
    'pitch_norm':       pitch_norm_aligned,
    'pitch_up':         pitch_up_aligned,
    'pitch_down':       pitch_down_aligned,
})
for k, v in mel_aligned.items():
    out[k] = v

out.to_csv(out_csv, index=False)

print(f'\nSaved: {out_csv}')
print(f'Rows:              {len(out)}')
print(f'Vowel onsets:      {(out["vowel_onset"] != "").sum()}')
print(f'Phoneme onsets:    {(out["phoneme_onset"] != "").sum()}')
print(f'Word onsets:       {(out["word_onset"] != "").sum()}')
print(f'Word freq onsets:  {(out["word_frequency"] > 0).sum()}')
print(f'Word NER onsets:   {(out["word_ner"] != "").sum()}')
print(f'Biphone rows > 0:  {(out["biphone_surprisal"] > 0).sum()}')
print(f'Mel bands:         {len(mel_aligned)}')
print(f'Total columns:     {len(out.columns)}')
