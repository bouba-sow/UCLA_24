"""
Preprocessing utilities for the MFA (Montreal Forced Aligner) pipeline.

Main entry point: build_phoneme_and_word_level_csv() — reads a video + TextGrid
and outputs per-frame phoneme one-hot and word label CSVs.
"""
from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pysrt
from praatio import textgrid as tgio
from pydub import AudioSegment


# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------

def add_separator(input_txt: str, output_txt: str, separator: str = '|') -> None:
    """Insert a separator token between paragraph breaks in a plain-text file."""
    with open(input_txt) as f, open(output_txt, 'w') as out_f:
        previous_empty = False
        for line in f:
            if line.strip() == '':
                if not previous_empty:
                    out_f.write(f' {separator} ')
                previous_empty = True
            else:
                out_f.write(line.replace('\n', ' '))
                previous_empty = False


def _seconds_to_srt_time(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total = int(td.total_seconds())
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    ms = int(td.microseconds / 1000)
    return f'{h:02}:{m:02}:{s:02},{ms:03}'


def create_srt_file(input_txt: str, output_srt: str) -> None:
    """Convert a CTM-style text file to SRT format."""
    srt_lines = []
    with open(input_txt) as f:
        for idx, line in enumerate(f, 1):
            parts = line.strip().split(' ', 4)
            if len(parts) < 5:
                continue
            start = float(parts[2])
            end   = start + float(parts[3])
            text  = parts[4].replace('<space>', ' ').replace('|', ' ').strip()
            srt_lines.append(
                f'{idx}\n{_seconds_to_srt_time(start)} --> {_seconds_to_srt_time(end)}\n{text}\n\n'
            )
    with open(output_srt, 'w') as f:
        f.writelines(srt_lines)


def srt_to_txt(srt_path: str, txt_path: str) -> None:
    """Extract subtitle text from an SRT file, one line per block."""
    import re
    with open(srt_path, encoding='utf-8') as f:
        content = f.read()
    blocks = re.split(r'\n\s*\n', content.strip())
    lines = []
    for block in blocks:
        parts = block.splitlines()
        if len(parts) >= 3:
            text = ' '.join(parts[2:]).strip()
            if text:
                lines.append(text)
    with open(txt_path, 'w', encoding='utf-8') as out_f:
        for line in lines:
            out_f.write(line + '\n')
    print(f'Converted {len(lines)} subtitles from {srt_path} → {txt_path}')


# ---------------------------------------------------------------------------
# MFA segmentation
# ---------------------------------------------------------------------------

def segment_for_mfa(srt_path: str, audio_path: str, output_dir: str) -> None:
    """Split audio + text into per-subtitle segments ready for MFA alignment."""
    os.makedirs(output_dir, exist_ok=True)
    print('Loading audio (this may take a moment for long files)...')
    full_audio = AudioSegment.from_wav(audio_path)
    subs = pysrt.open(srt_path)
    print(f'Processing {len(subs)} segments...')
    for sub in subs:
        start_ms = (sub.start.hours * 3_600_000 + sub.start.minutes * 60_000
                    + sub.start.seconds * 1_000 + sub.start.milliseconds)
        end_ms   = (sub.end.hours * 3_600_000 + sub.end.minutes * 60_000
                    + sub.end.seconds * 1_000 + sub.end.milliseconds)
        base = f'sub_{str(sub.index).zfill(4)}'
        full_audio[start_ms:end_ms].export(os.path.join(output_dir, f'{base}.wav'), format='wav')
        with open(os.path.join(output_dir, f'{base}.lab'), 'w', encoding='utf-8') as f:
            f.write(sub.text.replace('\n', ' ').strip())
    print(f'Done! Created {len(subs)} pairs in {output_dir}')


# ---------------------------------------------------------------------------
# TextGrid merging
# ---------------------------------------------------------------------------

def merge_textgrids(srt_path: str, audio_path: str, tg_folder: str, output_path: str) -> None:
    """Merge per-segment MFA TextGrids into a single master TextGrid."""
    audio = AudioSegment.from_wav(audio_path)
    total_duration_sec = len(audio) / 1000.0
    print(f'Total duration: {total_duration_sec:.2f} seconds')

    word_entries: list[tuple] = []
    phone_entries: list[tuple] = []
    subs = pysrt.open(srt_path)
    found_count = 0

    for sub in subs:
        offset = (sub.start.hours * 3600 + sub.start.minutes * 60
                  + sub.start.seconds + sub.start.milliseconds / 1000.0)
        base = f'sub_{str(sub.index).zfill(4)}'
        tg_path = os.path.join(tg_folder, f'{base}.TextGrid')
        if not os.path.exists(tg_path):
            tg_path = os.path.join(tg_folder, f'{base}.Textgrid')
            if not os.path.exists(tg_path):
                continue
        try:
            seg_tg = tgio.openTextgrid(tg_path, includeEmptyIntervals=False)
            found_count += 1
            for tier_name, target_list in (('words', word_entries), ('phones', phone_entries)):
                try:
                    tier = seg_tg.getTier(tier_name)
                except Exception:
                    continue
                for entry in tier.entries:
                    new_start = round(entry[0] + offset, 5)
                    new_end   = min(round(entry[1] + offset, 5), total_duration_sec)
                    if new_end > new_start and entry[2].strip():
                        target_list.append((new_start, new_end, entry[2]))
        except Exception as e:
            print(f'Error processing {base}: {e}')

    print(f'Successfully processed {found_count} files.')
    master_tg = tgio.Textgrid()
    master_tg.addTier(tgio.IntervalTier('words',  word_entries,  0, total_duration_sec))
    master_tg.addTier(tgio.IntervalTier('phones', phone_entries, 0, total_duration_sec))
    master_tg.save(output_path, format='long_textgrid', includeBlankSpaces=False)
    print(f'Master Textgrid saved to: {output_path}')


# ---------------------------------------------------------------------------
# Phoneme / word CSV builder
# ---------------------------------------------------------------------------

def build_phoneme_and_word_level_csv(
    video_path: str,
    text_grid_path: str,
    phoneme_csv_out: str,
    word_csv_out: str,
) -> None:
    """Build per-frame phoneme one-hot and word label CSVs from a TextGrid + video."""
    SIL_PHONEME = 'sil'
    SIL_WORD    = 'None'

    cap = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    cap.release()
    print(f'Video FPS: {fps}, Total Frames: {total_frames}, Duration: {duration_sec:.2f}s')

    tg = tgio.openTextgrid(text_grid_path, includeEmptyIntervals=False)
    word_entries  = tg.getTier('words').entries
    phone_entries = tg.getTier('phones').entries
    print(f'Loaded {len(word_entries)} word entries and {len(phone_entries)} phoneme entries.')

    phoneme_set = {e[2].strip() for e in phone_entries if e[2].strip()} | {SIL_PHONEME}
    phoneme_list = sorted(phoneme_set)
    phoneme_to_idx = {ph: i for i, ph in enumerate(phoneme_list)}

    def time_to_frame(t: float) -> int:
        return int(np.floor(t * fps))

    phoneme_matrix = np.zeros((total_frames, len(phoneme_list)), dtype=np.int8)
    phoneme_matrix[:, phoneme_to_idx[SIL_PHONEME]] = 1
    words_arr = np.array([SIL_WORD] * total_frames, dtype=object)

    for start, end, label in phone_entries:
        label = label.strip()
        if not label:
            continue
        sf, ef = max(0, time_to_frame(start)), min(total_frames, time_to_frame(end))
        if sf >= ef:
            continue
        p_idx = phoneme_to_idx[label]
        phoneme_matrix[sf:ef, phoneme_to_idx[SIL_PHONEME]] = 0
        phoneme_matrix[sf:ef, p_idx] = 1

    for start, end, label in word_entries:
        label = label.strip()
        if not label:
            continue
        sf, ef = max(0, time_to_frame(start)), min(total_frames, time_to_frame(end))
        if sf >= ef:
            continue
        words_arr[sf:ef] = label

    frame_ids = np.arange(total_frames)
    times     = frame_ids / fps

    phoneme_df = pd.DataFrame(phoneme_matrix, columns=phoneme_list)
    phoneme_df.insert(0, 'time',  times)
    phoneme_df.insert(0, 'frame', frame_ids)

    word_df = pd.DataFrame({'frame': frame_ids, 'time': times, 'word': words_arr})

    phoneme_df.to_csv(phoneme_csv_out, index=False)
    word_df.to_csv(word_csv_out, index=False)
    print(f'Saved phoneme CSV → {phoneme_csv_out}')
    print(f'Saved word CSV    → {word_csv_out}')


# ---------------------------------------------------------------------------
# Video annotation helper
# ---------------------------------------------------------------------------

def mark_video_with_concepts_and_characters(
    video_path: str,
    characters: str,
    concepts: str,
    output_path: str,
) -> None:
    """Overlay character and concept labels onto a video."""
    characters_df = pd.read_csv(characters)
    concept_df    = pd.read_csv(concepts)

    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out    = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        time_sec = frame_idx / fps

        if frame_idx < len(characters_df):
            row    = characters_df.iloc[frame_idx]
            active = [c for c in characters_df.columns if row[c] == 1]
            if active:
                cv2.putText(frame, f'Character: {", ".join(active)}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        sec_idx = int(time_sec)
        if sec_idx < len(concept_df):
            row    = concept_df.iloc[sec_idx]
            active = [c for c in concept_df.columns if row[c] == 1]
            if active:
                cv2.putText(frame, f'Concept: {", ".join(active)}',
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    build_phoneme_and_word_level_csv(
        video_path       = '/store/scratch/bsow/Documents/UCLA_24/data/40m_act_24_S06E01_30fps_subtitled_marked.mp4',
        text_grid_path   = '/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_MASTER.Textgrid',
        phoneme_csv_out  = '/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_phonemes.csv',
        word_csv_out     = '/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_words.csv',
    )
