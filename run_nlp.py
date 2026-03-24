import numpy as np
import torch

try:
    torch.serialization.add_safe_globals([
        np.core.multiarray._reconstruct,  # the function
        np.ndarray,                        # the class
        np.dtype                            # the type
    ])
except Exception as e:
    print("Fallback: disabling weights_only")
    import torch.serialization
    torch.serialization.default_weights_only = False
    exit(1)
# ✅ 2. FORCE weights_only=False globally
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load
  
import cv2
import srt
import spacy_stanza
from spacy import displacy
import cairosvg
import io
from PIL import Image

# 2. Setup spacy-stanza
nlp = spacy_stanza.load_pipeline("en", processors='tokenize,pos,lemma,depparse', use_gpu=True)

VIDEO_PATH = "data/40m_act_24_S06E01_30fps_subtitled.mp4"
SRT_PATH = "data/24_S06E01.srt"
OUTPUT_PATH = "output/24_S06E01_Linguistic_Analysis.mp4"

def render_dep_tree(text, width, max_height):
    doc = nlp(text)

    options = {
        "compact": True,
        "color": "white",
        "bg": "#00000000",
        "font": "Arial"
    }

    svg_data = displacy.render(doc, style="dep", jupyter=False, options=options)

    # 🔥 Render at correct scale directly
    png_data = cairosvg.svg2png(
        bytestring=svg_data.encode('utf-8'),
        output_width=width,
        output_height=max_height  # ← KEY FIX
    )

    img = Image.open(io.BytesIO(png_data)).convert("RGBA")
    return np.array(img)

def process_video():
    with open(SRT_PATH, 'r', encoding='utf-8') as f:
        subs = list(srt.parse(f.read()))

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    current_sub_idx = 0
    cached_tree = None
    last_rendered_text = None
    
    print("Starting video synthesis with caching...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        if current_sub_idx < len(subs):
            sub = subs[current_sub_idx]
            
            if sub.start.total_seconds() <= frame_time <= sub.end.total_seconds():
                clean_text = sub.content.replace('\n', ' ')
                
                # ONLY render if the text has changed
                if clean_text != last_rendered_text:
                    MAX_OVERLAY_HEIGHT = int(h * 0.35)  # 35% of screen
                    cached_tree = render_dep_tree(
                        clean_text,
                        width=w,
                        max_height=MAX_OVERLAY_HEIGHT
                    )
                    last_rendered_text = clean_text
                
                # Overlay logic
                overlay_h, overlay_w = cached_tree.shape[:2]

                # ✅ Now guaranteed to fit
                alpha = cached_tree[:, :, 3] / 255.0

                for c in range(3):
                    frame[0:overlay_h, 0:overlay_w, c] = (
                        alpha * cached_tree[:, :, c] +
                        (1 - alpha) * frame[0:overlay_h, 0:overlay_w, c]
                    )
            elif frame_time > sub.end.total_seconds():
                current_sub_idx += 1
                cached_tree = None # Clear cache for next segment

        out.write(frame)
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 500 == 0:
            print(f"Processed {int(cap.get(cv2.CAP_PROP_POS_FRAMES))} frames...")

    cap.release()
    out.release()
    import subprocess
    import os

    print("Embedding subtitles into final video...")

    temp_output = OUTPUT_PATH.replace(".mp4", "_temp.mp4")

    # Rename current output temporarily
    os.rename(OUTPUT_PATH, temp_output)

    # Run ffmpeg to add subtitles
    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-i", temp_output,
        "-i", SRT_PATH,
        "-c", "copy",
        "-c:s", "mov_text",
        OUTPUT_PATH
    ]

    subprocess.run(cmd, check=True)

    # Remove temp file
    os.remove(temp_output)

    print(f"Final video with subtitles: {OUTPUT_PATH}")

if __name__ == "__main__":
    process_video()