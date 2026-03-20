def main():
    print("Hello from ucla-24!")

def add_separator(input_txt, output_txt, separator="|"):
    with open(input_txt, "r") as f, open(output_txt, "w") as out_f:
        previous_empty = False

        for line in f:
            if line.strip() == "":
                if not previous_empty:
                    out_f.write(" " +separator + " ")
                previous_empty = True
            else:
                # remove \n from the end of the line before writing
                line = line.replace("\n", " ")
                out_f.write(line)
                previous_empty = False

def create_srt_file(input_txt, output_srt):
    from datetime import timedelta

    def seconds_to_srt_time(seconds):
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int(td.microseconds / 1000 + (td.seconds - total_seconds) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    srt_lines = []
    with open(input_txt, "r") as f:  # replace with your file
        for idx, line in enumerate(f, 1):
            parts = line.strip().split(" ", 4)
            if len(parts) < 5:
                continue
            start = float(parts[2])
            duration = float(parts[3])
            text = parts[4].replace("<space>", " ").replace("|", " ").strip()
            end = start + duration
            srt_lines.append(f"{idx}\n{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}\n{text}\n\n")

    with open(output_srt, "w") as f:
        f.writelines(srt_lines)
                   
import re

def srt_to_txt(srt_path, txt_path):
    """
    Converts an SRT file to a plain text file where each subtitle
    becomes one line, ready for MFA.
    """
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into blocks based on SRT numbering
    blocks = re.split(r"\n\s*\n", content.strip())

    lines = []
    for block in blocks:
        # Each block typically has 3 lines: number, time, text
        parts = block.splitlines()
        if len(parts) >= 3:
            # Join all text lines in the block
            text = " ".join(parts[2:]).strip()
            if text:
                lines.append(text)

    # Save to TXT
    with open(txt_path, "w", encoding="utf-8") as out_f:
        for line in lines:
            out_f.write(line + "\n")

    print(f"Converted {len(lines)} subtitles from {srt_path} → {txt_path}")

import os
import pysrt
from pydub import AudioSegment

def segment_for_mfa(srt_path, audio_path, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Load the full audio and the subtitles
    print("Loading audio (this may take a moment for 40 mins)...")
    full_audio = AudioSegment.from_wav(audio_path)
    subs = pysrt.open(srt_path)
    
    print(f"Processing {len(subs)} segments...")

    for sub in subs:
        # Convert SRT time to milliseconds for pydub
        start_ms = (sub.start.hours * 3600000 + 
                    sub.start.minutes * 60000 + 
                    sub.start.seconds * 1000 + 
                    sub.start.milliseconds)
        
        end_ms = (sub.end.hours * 3600000 + 
                  sub.end.minutes * 60000 + 
                  sub.end.seconds * 1000 + 
                  sub.end.milliseconds)

        # 1. Extract the audio slice
        audio_slice = full_audio[start_ms:end_ms]

        # 2. Prepare the filenames (using the subtitle index)
        # We use zfill(4) so '1' becomes '0001', keeping files sorted in folders
        base_name = f"sub_{str(sub.index).zfill(4)}"
        
        # 3. Export .wav file
        wav_filename = os.path.join(output_dir, f"{base_name}.wav")
        audio_slice.export(wav_filename, format="wav")

        # 4. Export .lab file (MFA requires plain text, no timestamps)
        lab_filename = os.path.join(output_dir, f"{base_name}.lab")
        with open(lab_filename, "w", encoding="utf-8") as f:
            # Clean text: MFA prefers no newlines or special chars
            clean_text = sub.text.replace('\n', ' ').strip()
            f.write(clean_text)

    print(f"Done! Successfully created {len(subs)} pairs in {output_dir}")

import os
import pysrt
from praatio import textgrid as tgio
from pydub import AudioSegment

def merge_Textgrids_praatio(srt_path, audio_path, tg_folder, output_path):
    # 1. Get exact duration
    audio = AudioSegment.from_wav(audio_path)
    total_duration_sec = len(audio) / 1000.0
    print(f"Total duration: {total_duration_sec:.2f} seconds")

    # 2. Collect entries in lists before creating tiers
    word_entries = []
    phone_entries = []

    subs = pysrt.open(srt_path)
    
    found_count = 0
    print("Merging segments...")
    
    for sub in subs:
        offset = (sub.start.hours * 3600 + 
                  sub.start.minutes * 60 + 
                  sub.start.seconds + 
                  sub.start.milliseconds / 1000.0)
        
        # Check for .TextGrid (MFA standard)
        tg_filename = "sub_{}.TextGrid".format(str(sub.index).zfill(4))
        tg_path = os.path.join(tg_folder, tg_filename)

        if not os.path.exists(tg_path):
            # Fallback to .Textgrid
            tg_path = os.path.join(tg_folder, tg_filename.replace(".TextGrid", ".Textgrid"))
            if not os.path.exists(tg_path):
                continue

        try:
            # openTextgrid with mandatory includeEmptyIntervals=True
            seg_tg = tgio.openTextgrid(tg_path, includeEmptyIntervals=False)
            found_count += 1
            
            # Access tiers using getTier method instead of tierDict
            for tier_name in ["words", "phones"]:
                try:
                    tier = seg_tg.getTier(tier_name)
                except:
                    continue
                
                target_list = word_entries if tier_name == "words" else phone_entries
                
                # Access entries directly from the tier object
                for entry in tier.entries:
                    new_start = round(entry[0] + offset, 5) # Rounding prevents float jitter
                    new_end = round(entry[1] + offset, 5)
                    label = entry[2]
                    
                    if new_end > total_duration_sec:
                        new_end = total_duration_sec
                    
                    if new_end > new_start and label.strip() != "":
                        target_list.append((new_start, new_end, label))
                        
        except Exception as e:
            print(f"Error processing {tg_filename}: {e}")

    print(f"Successfully processed {found_count} files.")

    # 3. Create tiers with collected entries
    master_tg = tgio.Textgrid()
    word_tier = tgio.IntervalTier("words", word_entries, 0, total_duration_sec)
    phone_tier = tgio.IntervalTier("phones", phone_entries, 0, total_duration_sec)
    
    master_tg.addTier(word_tier)
    master_tg.addTier(phone_tier)
    
    # format="long_textgrid" is the most compatible with Praat
    master_tg.save(output_path, format="long_textgrid", includeBlankSpaces=False)
    print(f"Master Textgrid saved to: {output_path}")

def mark_video_with_concepts_and_characters(video_path, characters, concepts, output_path):
    import cv2
    import pandas as pd
    
    # Load the charaters and concepts
    characters_df = pd.read_csv(characters)
    concept_df = pd.read_csv(concepts)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    def get_character_label(frame_idx):
        if frame_idx >= len(characters_df):
            return None
        row = characters_df.iloc[frame_idx]
        active = [char for char in characters_df.columns if row[char] == 1]
        return ", ".join(active) if active else None
    
    def get_concept_label(time_sec):
        sec_idx = int(time_sec)
        if sec_idx >= len(concept_df):
            return None
        row = concept_df.iloc[sec_idx]
        active = [concept for concept in concept_df.columns if row[concept] == 1]
        return ", ".join(active) if active else None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        time_sec = frame_idx / fps
        char_label = get_character_label(frame_idx)
        concept_label = get_concept_label(time_sec)
        
        # Overlay character labels
        if char_label:
            cv2.putText(frame, f"Character: {char_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
        
        # Overlay concept labels
        if concept_label:
            cv2.putText(frame, f"Concept: {concept_label}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)
        
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()

def build_phoneme_and_word_level_csv(video_path, text_grid_path, phoneme_csv_out, word_csv_out):
    import numpy as np
    import pandas as pd
    import cv2
    from praatio import textgrid as tgio
    
    SIL_PHONEME = "sil"
    SIL_word = "None"
    
    # Load video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    
    print(f"Video FPS: {fps}, Total Frames: {total_frames}, Duration: {duration_sec:.2f} seconds")
    
    # Load TextGrid
    tg = tgio.openTextgrid(text_grid_path, includeEmptyIntervals=False)
    word_tier = tg.getTier("words")
    phone_tier = tg.getTier("phones")
    
    word_entries = word_tier.entries
    phone_entries = phone_tier.entries
    
    print(f"Loaded {len(word_entries)} word entries and {len(phone_entries)} phoneme entries from TextGrid.")
    
    # Build phoneme vocabulary
    phoneme_set = set()
    for start, end, label in phone_entries:
        label = label.strip()
        if label != "":
            phoneme_set.add(label)
    phoneme_set.add(SIL_PHONEME)
    phoneme_list = sorted(list(phoneme_set))
    phoneme_to_idx = {ph: idx for idx, ph in enumerate(phoneme_list)}
    print(f"Phoneme list: {phoneme_list}")
    print(f"Phoneme vocab size: {len(phoneme_list)}")
    
    # Init arrays
    # Phoneme one-hot
    phoneme_matrix = np.zeros((total_frames, len(phoneme_list)), dtype=np.int8)
    sil_idx = phoneme_to_idx[SIL_PHONEME]
    phoneme_matrix[:, sil_idx] = 1
    
    # Word list
    words = np.array([SIL_word] * total_frames, dtype=object)
    
    # =========================
    # HELPER
    # =========================
    def time_to_frame(t):
        return int(np.floor(t * fps))

    # =========================
    # FILL PHONEMES
    # =========================
    for start, end, label in phone_entries:
        label = label.strip()
        if label == "":
            continue

        start_f = max(0, time_to_frame(start))
        end_f = min(total_frames, time_to_frame(end))

        if start_f >= end_f:
            continue

        p_idx = phoneme_to_idx[label]

        for f in range(start_f, end_f):
            # remove silence
            phoneme_matrix[f, sil_idx] = 0
            # set phoneme
            phoneme_matrix[f, p_idx] = 1

    # =========================
    # FILL WORDS
    # =========================
    for start, end, label in word_entries:
        label = label.strip()
        if label == "":
            continue

        start_f = max(0, time_to_frame(start))
        end_f = min(total_frames, time_to_frame(end))

        if start_f >= end_f:
            continue

        for f in range(start_f, end_f):
            words[f] = label

    # =========================
    # BUILD DATAFRAMES
    # =========================
    frame_ids = np.arange(total_frames)
    times = frame_ids / fps

    # Phoneme DF
    phoneme_df = pd.DataFrame(phoneme_matrix, columns=phoneme_list)
    phoneme_df.insert(0, "time", times)
    phoneme_df.insert(0, "frame", frame_ids)

    # Word DF
    word_df = pd.DataFrame({
        "frame": frame_ids,
        "time": times,
        "word": words
    })

    # =========================
    # SAVE
    # =========================
    phoneme_df.to_csv(phoneme_csv_out, index=False)
    word_df.to_csv(word_csv_out, index=False)

    print("✅ Done!")
    print(f"Saved phoneme CSV → {phoneme_csv_out}")
    print(f"Saved word CSV → {word_csv_out}")
        
        
if __name__ == "__main__":
    #add_separator("/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_subtitles.txt", "/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_subtitles_with_separators.txt")
    #create_srt_file("/store/scratch/bsow/Documents/UCLA_24/output/ctm/segments/40m_act_24_S06E01_30fps_mono.ctm", "/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_subtitles.srt")
    #srt_to_txt("/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_subtitles.srt", "/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_subtitles.txt")
    # Execution
    #srt_file = "/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01.srt"
    #audio_file = "/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01.wav"
    #output_path = "/store/scratch/bsow/Documents/UCLA_24/data/mfa_data"
#
    #segment_for_mfa(srt_file, audio_file, output_path)
    
    # --- Execution ---
    #srt_file = "/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01.srt"
    #audio_file = "/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01.wav"
    #Textgrid_dir = "/store/scratch/bsow/Documents/UCLA_24/data/mfa_output"
    #output_file = "/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_MASTER.Textgrid"

    #merge_Textgrids_praatio(srt_file, audio_file, Textgrid_dir, output_file)
    
    #video_path = "/store/scratch/bsow/Documents/UCLA_24/data/40m_act_24_S06E01_30fps_subtitled.mp4"
    #characters_csv = "/store/scratch/bsow/Documents/UCLA_24/data/40m_act_24_S06E01_30fps_characters.csv"
    #concepts_csv = "/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_8concepts_merged.csv"
    #output_video = "/store/scratch/bsow/Documents/UCLA_24/data/40m_act_24_S06E01_30fps_subtitled_marked.mp4"
    #
    #mark_video_with_concepts_and_characters(video_path, characters_csv, concepts_csv, output_video)
    
    video_path = "/store/scratch/bsow/Documents/UCLA_24/data/40m_act_24_S06E01_30fps_subtitled_marked.mp4"
    text_grid_path = "/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_MASTER.Textgrid"
    phoneme_csv_out = "/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_phonemes.csv"
    word_csv_out = "/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_words.csv"

    build_phoneme_and_word_level_csv(video_path, text_grid_path, phoneme_csv_out, word_csv_out)