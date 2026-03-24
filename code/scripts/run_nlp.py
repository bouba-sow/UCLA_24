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
import re
from collections import Counter
from PIL import Image

# 2. Setup spacy-stanza
nlp = spacy_stanza.load_pipeline("en", processors='tokenize,pos,lemma,depparse', use_gpu=True)

VIDEO_PATH = "data/40m_act_24_S06E01_30fps_subtitled_marked.mp4"
SRT_PATH = "data/24_S06E01.srt"
OUTPUT_PATH = "24_S06E01_Linguistic_Analysis.mp4"

# Overlay placement tuning.
TREE_TOP_OFFSET_PX = 90     # leave space for character/concept labels at top
TREE_LEFT_MARGIN_PX = 10
TREE_MAX_WIDTH_RATIO = 0.95
TREE_MAX_HEIGHT_RATIO = 0.58
LEX_PANEL_MARGIN_PX = 12
LEX_PANEL_GAP_FROM_TREE_PX = 10
LEX_PANEL_HEIGHT_PX = 88
LEX_CELL_MIN_WIDTH_PX = 24
LEX_CELL_GOOD_WIDTH_PX = 44

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def build_subtitle_frequency_profile(subs):
    """Builds corpus word-frequency stats from the subtitle file."""
    counts = Counter()
    for sub in subs:
        text = sub.content.replace("\n", " ").lower()
        counts.update(TOKEN_PATTERN.findall(text))
    total_tokens = sum(counts.values()) or 1
    return counts, total_tokens

def token_frequency_score(token, freq_counts, total_tokens):
    """
    Returns a simple Zipf-like score (log10 per million + 1) from subtitle corpus.
    This is a fallback when external corpora are not available.
    """
    raw_count = freq_counts.get(token.lower(), 0)
    ppm = (raw_count / total_tokens) * 1_000_000.0
    return np.log10(ppm + 1.0)

def extract_lexical_features(doc, freq_counts, total_tokens):
    """Extracts lexical rows: word, length, frequency score."""
    rows = []
    for token in doc:
        txt = token.text.strip()
        if not txt:
            continue
        match = TOKEN_PATTERN.search(txt)
        if not match:
            continue
        clean = match.group(0)
        rows.append(
            {
                "word": clean,
                "length": len(clean),
                "freq": token_frequency_score(clean, freq_counts, total_tokens),
            }
        )
    return rows

def boxes_intersect(box_a, box_b):
    """Returns True if two (x1, y1, x2, y2) boxes intersect."""
    return not (
        box_a[2] <= box_b[0]
        or box_a[0] >= box_b[2]
        or box_a[3] <= box_b[1]
        or box_a[1] >= box_b[3]
    )

def draw_lexical_panel(frame, lexical_rows, tree_bbox=None):
    """Draws a horizontal lexical table: word / len / freq."""
    if not lexical_rows:
        return

    h, w = frame.shape[:2]
    x1 = LEX_PANEL_MARGIN_PX
    x2 = w - LEX_PANEL_MARGIN_PX
    panel_w = x2 - x1
    if panel_w <= 120:
        return

    panel_h = LEX_PANEL_HEIGHT_PX
    if tree_bbox is not None:
        y1 = tree_bbox[3] + LEX_PANEL_GAP_FROM_TREE_PX
    else:
        y1 = h - panel_h - LEX_PANEL_MARGIN_PX
    y2 = y1 + panel_h

    # Keep panel in frame if tree is too low.
    if y2 > h - LEX_PANEL_MARGIN_PX:
        y2 = h - LEX_PANEL_MARGIN_PX
        y1 = y2 - panel_h
    if y1 < LEX_PANEL_MARGIN_PX:
        y1 = LEX_PANEL_MARGIN_PX
        y2 = y1 + panel_h

    # Semi-transparent dark panel.
    panel_region = frame[y1:y2, x1:x2].copy()
    dark_bg = np.zeros_like(panel_region)
    frame[y1:y2, x1:x2] = cv2.addWeighted(panel_region, 0.40, dark_bg, 0.60, 0.0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    label_w = 44
    left_pad = 6
    right_pad = 6
    data_w = max(1, panel_w - label_w - left_pad - right_pad)
    max_visible = max(1, data_w // LEX_CELL_MIN_WIDTH_PX)
    target_visible = min(len(lexical_rows), max_visible)
    cell_w = max(LEX_CELL_MIN_WIDTH_PX, data_w // target_visible)
    good_visible = max(1, data_w // LEX_CELL_GOOD_WIDTH_PX)

    rows_to_draw = lexical_rows[:target_visible]
    truncated_count = max(0, len(lexical_rows) - target_visible)

    y_word = y1 + 24
    y_len = y1 + 50
    y_freq = y1 + 76
    label_color = (0, 255, 255)
    data_color = (230, 230, 230)

    cv2.putText(frame, "word", (x1 + left_pad, y_word), cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "len",  (x1 + left_pad, y_len),  cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "freq", (x1 + left_pad, y_freq), cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1, cv2.LINE_AA)

    data_x_start = x1 + label_w
    word_char_limit = 10 if target_visible <= good_visible else 6
    word_font = 0.38 if target_visible <= good_visible else 0.33

    for idx, row in enumerate(rows_to_draw):
        x = data_x_start + idx * cell_w
        word_text = row["word"][:word_char_limit]
        cv2.putText(frame, word_text, (x, y_word), cv2.FONT_HERSHEY_SIMPLEX, word_font, data_color, 1, cv2.LINE_AA)
        cv2.putText(frame, str(row["length"]), (x, y_len), cv2.FONT_HERSHEY_SIMPLEX, 0.36, data_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{row['freq']:.2f}", (x, y_freq), cv2.FONT_HERSHEY_SIMPLEX, 0.36, data_color, 1, cv2.LINE_AA)

    if truncated_count > 0:
        cv2.putText(
            frame,
            f"+{truncated_count} more",
            (x2 - 110, y_freq),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

def render_dep_tree(doc, max_width, max_height):
    """Converts a sentence into a readable PNG dependency tree."""
    token_count = len([t for t in doc if not t.is_space])

    # Large trees become blurry when rasterized directly to the target width.
    # Render bigger first, then downsample with high-quality filters.
    supersample = 3 if token_count <= 18 else 4

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
        scale=supersample,
        dpi=384,
    )
    img = Image.open(io.BytesIO(png_data)).convert("RGBA")

    # Keep natural size for short trees; shrink only if out of bounds.
    width_ratio = max_width / float(img.width) if img.width > max_width else 1.0
    height_ratio = max_height / float(img.height) if img.height > max_height else 1.0
    fit_ratio = min(width_ratio, height_ratio)
    if fit_ratio < 1.0:
        fit_width = max(1, int(img.width * fit_ratio))
        fit_height = max(1, int(img.height * fit_ratio))
        img = img.resize((fit_width, fit_height), Image.Resampling.LANCZOS)

    return np.array(img)

def process_video():
    with open(SRT_PATH, 'r', encoding='utf-8') as f:
        subs = list(srt.parse(f.read()))
    freq_counts, total_tokens = build_subtitle_frequency_profile(subs)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    current_sub_idx = 0
    cached_tree = None
    cached_lexical_rows = []
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
                    doc = nlp(clean_text)
                    cached_tree = render_dep_tree(
                        doc,
                        max_width=int(w * TREE_MAX_WIDTH_RATIO),
                        max_height=int(h * TREE_MAX_HEIGHT_RATIO),
                    )
                    cached_lexical_rows = extract_lexical_features(doc, freq_counts, total_tokens)
                    last_rendered_text = clean_text
                
                # Overlay logic
                overlay_h, overlay_w = cached_tree.shape[:2]
                if overlay_w > w:
                    # Simple failsafe resize
                    cached_tree = cv2.resize(cached_tree, (w, int(overlay_h * (w / overlay_w))))
                    overlay_h, overlay_w = cached_tree.shape[:2]

                # Clamp/crop to frame bounds to avoid blend shape mismatches.
                overlay_x = TREE_LEFT_MARGIN_PX
                overlay_y = TREE_TOP_OFFSET_PX
                if overlay_y >= h or overlay_x >= w:
                    continue

                draw_h = min(overlay_h, h - overlay_y)
                draw_w = min(overlay_w, w - overlay_x)
                tree_roi = cached_tree[:draw_h, :draw_w, :]
                tree_bbox = (overlay_x, overlay_y, overlay_x + draw_w, overlay_y + draw_h)

                # Blend into the top area
                alpha = tree_roi[:, :, 3] / 255.0
                for c in range(3):
                    frame[overlay_y:overlay_y + draw_h, overlay_x:overlay_x + draw_w, c] = (
                        alpha * tree_roi[:, :, c]
                        + (1 - alpha) * frame[overlay_y:overlay_y + draw_h, overlay_x:overlay_x + draw_w, c]
                    )
                draw_lexical_panel(frame, cached_lexical_rows, tree_bbox=tree_bbox)
            elif frame_time > sub.end.total_seconds():
                current_sub_idx += 1
                cached_tree = None # Clear cache for next segment
                cached_lexical_rows = []

        out.write(frame)
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 500 == 0:
            print(f"Processed {int(cap.get(cv2.CAP_PROP_POS_FRAMES))} frames...")

    cap.release()
    out.release()
    print(f"Success! Output: {OUTPUT_PATH}")

if __name__ == "__main__":
    process_video()