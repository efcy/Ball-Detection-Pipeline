import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.widgets import Slider, TextBox
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── load one test image ──────────────────────────────────────────────────────
IMAGE_PATH = "C:/Users/anina/Desktop/Ball-Detection-Pipeline/test_images/28588257.jpg"
img   = Image.open(IMAGE_PATH)
ycbcr = img.convert('YCbCr')
arr   = np.array(ycbcr)
img_y = arr[:, :, 0].astype(float)
img_u = arr[:, :, 1].astype(float)
img_v = arr[:, :, 2].astype(float)
img_rgb = np.array(img)

# ── helper: print actual UV stats for a clicked pixel ────────────────────────
def print_pixel_stats(event):
    if event.inaxes != ax_img:
        return
    x, y = int(event.xdata), int(event.ydata)
    Y, U, V = img_y[y, x], img_u[y, x], img_v[y, x]
    angle  = np.degrees(np.arctan2(V - 128, U - 128))
    chroma = np.hypot(U - 128, V - 128)
    print(f"Pixel ({x},{y})  Y={Y:.0f}  U={U:.0f}  V={V:.0f}  "
          f"angle={angle:.1f}°  chroma={chroma:.1f}")

# ── classifier ──────────────────────────────────────────────────────────────
def classify_green(y, u, v, bW, bB, bO, cC_deg, cW_deg):
    cC    = np.radians(cC_deg)
    cW    = np.radians(cW_deg)
    alpha = (bW - bB) / (255.0 - bO)
    thresh = np.clip(bB + alpha * (y - bO), bB, 255)
    chroma = np.hypot(u - 128, v - 128)
    no_col = chroma < thresh
    angle  = np.arctan2(v - 128, u - 128)
    diff   = np.arctan2(np.sin(angle - cC), np.cos(angle - cC))
    is_ch  = np.abs(diff) < cW
    return np.logical_and(~no_col, is_ch)

# ── figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 8), facecolor='#1a1a2e')
fig.canvas.manager.set_window_title("Green Detector – YCbCr Tuner")

# Two image panels occupy the top 60% of the figure
ax_img  = fig.add_axes([0.03, 0.38, 0.44, 0.57])
ax_mask = fig.add_axes([0.53, 0.38, 0.44, 0.57])

for ax in (ax_img, ax_mask):
    ax.set_facecolor('#0d0d1a')
    for spine in ax.spines.values():
        spine.set_edgecolor('#3d3d6b')

ax_img.imshow(img_rgb)
ax_img.set_title("Original  ·  click pixel → YUV info",
                 color='#a0a0d0', fontsize=10, pad=6)
ax_img.tick_params(colors='#3d3d6b')

mask_display = ax_mask.imshow(np.zeros_like(img_y), cmap='Greens', vmin=0, vmax=1)
ax_mask.set_title("Green mask", color='#a0a0d0', fontsize=10, pad=6)
ax_mask.tick_params(colors='#3d3d6b')

fig.canvas.mpl_connect('button_press_event', print_pixel_stats)

# ── slider definitions ───────────────────────────────────────────────────────
#  (key,             label,           col, row,  lo,   hi,  init)
SLIDER_DEFS = [
    ("bW",  "White response  bW",     0,   2,    0,  150,   55),
    ("bB",  "Black response  bB",     0,   1,    0,   80,   40),
    ("bO",  "Offset  bO",             0,   0,    0,   80,   40),
    ("cC",  "Angle center (°)",       1,   1, -180,  180, -117),
    ("cW",  "Angle width  ±(°)",      1,   0,    1,   60,   84),
]

# Grid parameters for slider rows (bottom-anchored)
ROW_H   = 0.075   # height of each row slot
ROW_GAP = 0.005   # gap between rows
BASE_Y  = 0.035   # bottom of row 0

# Two column groups
COL_X   = [0.03, 0.53]   # left edge of each column group
SL_W    = 0.30            # slider width
TB_W    = 0.055           # textbox width
TB_GAP  = 0.008           # gap between slider and textbox
SL_H    = 0.030           # slider height
TB_H    = 0.030           # textbox height

# Matplotlib slider / textbox colours
SL_COLOR     = '#2a2a4a'
SL_HANDLE    = '#6666cc'
TB_COLOR     = '#12122a'

sliders   = {}
textboxes = {}
labels    = {}

for key, label, col, row, lo, hi, init in SLIDER_DEFS:
    cx   = COL_X[col]
    cy   = BASE_Y + row * (ROW_H + ROW_GAP)

    # ── label (axes text) ──────────────────────────────────────────────────
    ax_lbl = fig.add_axes([cx, cy + SL_H + 0.004, SL_W + TB_GAP + TB_W, 0.022])
    ax_lbl.set_axis_off()
    ax_lbl.text(0, 0.5, label,
                va='center', ha='left',
                color='#8888cc', fontsize=8.5, fontweight='semibold',
                transform=ax_lbl.transAxes)
    labels[key] = ax_lbl

    # ── slider ────────────────────────────────────────────────────────────
    ax_s = fig.add_axes([cx, cy, SL_W, SL_H], facecolor=SL_COLOR)
    for spine in ax_s.spines.values():
        spine.set_edgecolor('#3d3d6b')
    sl = Slider(ax_s, '', lo, hi, valinit=init,
                color=SL_HANDLE, track_color=SL_COLOR,
                handle_style={'facecolor': SL_HANDLE, 'edgecolor': '#9090ee',
                              'size': 12})
    sl.label.set_visible(False)
    sl.valtext.set_visible(False)   # we show value in our own textbox
    sliders[key] = sl

    # ── textbox (right of slider) ──────────────────────────────────────────
    ax_t = fig.add_axes([cx + SL_W + TB_GAP, cy, TB_W, TB_H],
                        facecolor=TB_COLOR)
    for spine in ax_t.spines.values():
        spine.set_edgecolor('#3d3d6b')
    tb = TextBox(ax_t, '', initial=f"{init:.1f}",
                 color=TB_COLOR, hovercolor='#1e1e3e',
                 label_pad=0)
    tb.text_disp.set_color('#ccccff')
    tb.text_disp.set_fontsize(9)
    textboxes[key] = tb

# ── update logic ─────────────────────────────────────────────────────────────

def _set_tb_text(key, val):
    """Update textbox display without triggering its on_submit callbacks."""
    textboxes[key].text_disp.set_text(f"{val:.1f}")

def redraw(print_params=False):
    bW     = sliders["bW"].val
    bB     = sliders["bB"].val
    bO     = sliders["bO"].val
    cC_deg = sliders["cC"].val
    cW_deg = sliders["cW"].val
    mask   = classify_green(img_y, img_u, img_v, bW, bB, bO, cC_deg, cW_deg)
    mask_display.set_data(mask.astype(float))
    frac = mask.mean()
    ax_mask.set_title(f"Green mask  ·  {frac*100:.1f}% pixels",
                      color='#a0a0d0', fontsize=10, pad=6)
    fig.canvas.draw_idle()
    if print_params:
        print(f"  ColorClassifier({bW:.0f}, {bB:.0f}, {bO:.0f}, "
              f"np.radians({cC_deg:.1f}), np.radians({cW_deg:.1f}))"
              f"  ←  center={cC_deg:.1f}°  width=±{cW_deg:.1f}°")

def on_slider_changed(key):
    def _cb(val):
        _set_tb_text(key, val)   # cheap: just update the text artist
        redraw()                 # no print while dragging
    return _cb

def on_slider_released(key):
    """Print params once when the mouse button is released."""
    def _cb(event):
        if event.inaxes == sliders[key].ax:
            redraw(print_params=True)
    return _cb

def on_text_submit(key):
    def _cb(text):
        try:
            val = float(text)
        except ValueError:
            return
        sl  = sliders[key]
        val = float(np.clip(val, sl.valmin, sl.valmax))
        sl.set_val(val)          # triggers on_slider_changed → _set_tb_text
        redraw(print_params=True)
    return _cb

for key in sliders:
    sliders[key].on_changed(on_slider_changed(key))
    textboxes[key].on_submit(on_text_submit(key))

fig.canvas.mpl_connect('button_release_event',
                       lambda e: [on_slider_released(k)(e) for k in sliders])

# ── separator line ────────────────────────────────────────────────────────────
ax_sep = fig.add_axes([0.03, 0.355, 0.94, 0.003])
ax_sep.set_facecolor('#3d3d6b')
ax_sep.set_axis_off()

redraw(print_params=True)
plt.show()