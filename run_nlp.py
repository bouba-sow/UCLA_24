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

VIDEO_PATH = "/store/scratch/bsow/Documents/UCLA_24/data/40m_act_24_S06E01_30fps_subtitled_marked.mp4"
SRT_PATH = "/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01.srt"
OUTPUT_PATH = "24_S06E01_Linguistic_Analysis.mp4"

def render_dep_tree(text, width):
    """Converts a sentence into a PNG image of its dependency tree."""
    doc = nlp(text)
    options = {"compact": True, "color": "white", "bg": "#00000000", "font": "Arial"}
    svg_data = displacy.render(doc, style="dep", jupyter=False, options=options)
    
    # Scale width to match video width
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'), output_width=width)
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
                    cached_tree = render_dep_tree(clean_text, width=w)
                    last_rendered_text = clean_text
                
                # Overlay logic
                overlay_h, overlay_w = cached_tree.shape[:2]

                # ✅ Ensure overlay fits inside frame
                max_h, max_w = frame.shape[:2]

                scale = min(max_w / overlay_w, max_h / overlay_h, 1.0)

                if scale < 1.0:
                    new_w = int(overlay_w * scale)
                    new_h = int(overlay_h * scale)
                    cached_tree = cv2.resize(cached_tree, (new_w, new_h))
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
    print(f"Success! Output: {OUTPUT_PATH}")

if __name__ == "__main__":
    process_video()