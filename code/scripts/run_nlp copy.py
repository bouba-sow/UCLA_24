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

VIDEO_PATH = "data/40m_act_24_S06E01_30fps_subtitled_marked.mp4"
SRT_PATH = "data/24_S06E01.srt"
OUTPUT_PATH = "24_S06E01_Linguistic_Analysis.mp4"

def render_dep_tree(text, width, max_height):
    """Converts a sentence into a readable PNG dependency tree."""
    doc = nlp(text)
    token_count = len([t for t in doc if not t.is_space])

    # Large trees become blurry when rasterized directly to the target width.
    # Render bigger first, then downsample with high-quality filters.
    supersample = 3 if token_count <= 18 else 4
    target_width = width
    render_width = width * supersample

    options = {
        "compact": token_count >= 20,
        "color": "white",
        "bg": "#00000000",
        "font": "Arial",
        "distance": 120 if token_count <= 16 else 100 if token_count <= 24 else 85,
    }
    svg_data = displacy.render(doc, style="dep", jupyter=False, options=options)

    # Make relation labels and POS tags more readable in video overlays.
    word_px = 24 if token_count <= 12 else 22 if token_count <= 20 else 20
    rel_px = 20 if token_count <= 12 else 18 if token_count <= 20 else 16
    outline_px = 1.4 if token_count <= 20 else 1.2
    svg_style = f"""
    .displacy-word {{
        font-size: {word_px}px !important;
        fill: #ffffff !important;
        paint-order: stroke;
        stroke: rgba(0, 0, 0, 0.75);
        stroke-width: {outline_px}px;
    }}
    .displacy-tag, .displacy-label {{
        font-size: {rel_px}px !important;
        fill: #00e5ff !important;
        font-weight: 700;
        paint-order: stroke;
        stroke: rgba(0, 0, 0, 0.80);
        stroke-width: {outline_px}px;
    }}
    .displacy-arc {{
        stroke-width: 2.6px !important;
    }}
    .displacy-arrowhead {{
        fill: #00e5ff !important;
    }}
    """
    svg_open_end = svg_data.find(">")
    if svg_open_end != -1:
        svg_data = (
            svg_data[:svg_open_end + 1]
            + f"<defs><style><![CDATA[{svg_style}]]></style></defs>"
            + svg_data[svg_open_end + 1:]
        )

    png_data = cairosvg.svg2png(
        bytestring=svg_data.encode("utf-8"),
        output_width=render_width,
        dpi=384,
    )
    img = Image.open(io.BytesIO(png_data)).convert("RGBA")

    # Fit overlay into a reasonable top band to prevent giant trees from smearing.
    if img.height > max_height:
        fit_ratio = max_height / float(img.height)
        fit_width = max(1, int(img.width * fit_ratio))
        img = img.resize((fit_width, max_height), Image.Resampling.LANCZOS)

    if img.width != target_width:
        fit_ratio = target_width / float(img.width)
        fit_height = max(1, int(img.height * fit_ratio))
        img = img.resize((target_width, fit_height), Image.Resampling.LANCZOS)

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
                    cached_tree = render_dep_tree(clean_text, width=w, max_height=int(h * 0.60))
                    last_rendered_text = clean_text
                
                # Overlay logic
                overlay_h, overlay_w = cached_tree.shape[:2]
                if overlay_w > w:
                    # Simple failsafe resize
                    cached_tree = cv2.resize(cached_tree, (w, int(overlay_h * (w / overlay_w))))
                    overlay_h, overlay_w = cached_tree.shape[:2]

                # Clamp/crop to frame bounds to avoid blend shape mismatches.
                draw_h = min(overlay_h, h)
                draw_w = min(overlay_w, w)
                tree_roi = cached_tree[:draw_h, :draw_w, :]

                # Blend into the top area
                alpha = tree_roi[:, :, 3] / 255.0
                for c in range(3):
                    frame[0:draw_h, 0:draw_w, c] = (
                        alpha * tree_roi[:, :, c] + (1 - alpha) * frame[0:draw_h, 0:draw_w, c]
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