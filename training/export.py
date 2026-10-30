"""Schrijft de gewichten en tabellen van gbref.GB als binaire bestanden + data.asm voor de ROM."""
import sys
from pathlib import Path
import numpy as np
from gbfont import FONT
import gbref

import os
OUT = Path(os.environ.get("GB_OUT", Path(__file__).resolve().parent.parent / "rom" / "build"))
S = gbref.tr.S
PITCH, OFF = (6, 0) if S == 16 else (3, 6)   # stip + tussenruimte in px, en marge in het canvas (28x28: 84 px)
gb = gbref.GB()
asm = []

def sec(name, bank, blob):
    (OUT / f"{name}.bin").write_bytes(bytes(blob))
    where = f"ROMX[$4000], BANK[{bank}]" if bank else "ROM0, ALIGN[8]"
    asm.append(f'SECTION "{name}", {where}\n{name}::\n    INCBIN "build/{name}.bin"\n')

def i16(a):  # little-endian int16, lo/hi naast elkaar zoals de accumulator in WRAM
    return np.asarray(a, np.int64).astype("<i2").tobytes()

# gewichten kolomsgewijs: kolom i (= invoer i) is W[:, i], zoveel hele kolommen per bank als er passen.
# Kolommen beginnen op een veelvoud van 128 bytes, zodat de ROM ze in blokken van 128 kan aflopen.
H = gb.W1.shape[0]; NB = H // 128; assert H % 128 == 0
bank = 1; banks = {}
for li, W in enumerate([gb.W1, gb.W2, gb.W3]):
    n = W.shape[0]; cpb = 16384 // n; Wt = W.T.astype(np.int8); banks[li] = (bank, cpb)
    for k in range(0, Wt.shape[0], cpb):
        sec(f"W{li + 1}_{k // cpb}", bank, Wt[k:k + cpb].tobytes()); bank += 1
# bias van laag 1 per (categorie, stap): 2H bytes, opgevuld tot een macht van 2
BSTR = 1 << int(np.ceil(np.log2(2 * H))); BPER = 16384 // BSTR; BIAS1_BANK = bank
B1 = b"".join(i16(gb.BIAS1[c, i]).ljust(BSTR, b"\0") for c in range(8) for i in range(16))
for k in range(0, len(B1), 16384):
    sec(f"BIAS1_{k // 16384}", bank, B1[k:k + 16384]); bank += 1
u8 = lambda a: (np.asarray(a, np.int64) & 0xFF).astype(np.uint8).tobytes()
byte_idx = np.arange(256); signed = np.where(byte_idx < 128, byte_idx, byte_idx - 256)
sec("B2", 0, i16(gb.B2)); sec("B3", 0, i16(gb.B3))
sec("TanhTab", 0, u8(gb.TANH[np.clip(signed, -128, 127) + 128]))           # index = byte van acc >> K3
sec("GaussTab", 0, u8(gb.GAUSS))
sec("VTab", 0, u8(np.clip(((signed + 4) >> 3) + 4, gbref.VMIN, gbref.VMAX)))  # x -> v van laag 1
sec("ToneTab", 0, u8(gb.TONE[signed + 128]))
sec("StepA", 0, u8(gb.A)); sec("StepB", 0, u8(gb.B))
# ---------- scherm (UI) ----------
# Tegels 0-143: canvas (12x12 tegels, 96x96 px = 16x16 stippen van 5 px met 1 px tussenruimte), in beide VRAM-banken.
# 144-163: kopregel, 164-183: voetregel (de ROM tekent er tekst in), 184+: vaste tegels van het scherm.
from PIL import Image
CX, CY, CW = 32, 24, 96            # canvas op tegelrand: kolom 4, rij 3
img = np.full((144, 160), 3, np.uint8)                       # palet 1: 0 papier .. 3 donker
x0, y0, x1, y1, rad = CX - 5, CY - 5, CX + CW + 4, CY + CW + 4, 3
for y in range(y0, y1 + 1):
    for x in range(x0, x1 + 1):
        dx = max(x0 + rad - x, 0, x - (x1 - rad)); dy = max(y0 + rad - y, 0, y - (y1 - rad))
        if dx * dx + dy * dy <= rad * rad + 1: img[y, x] = 0
def tile_bytes(cell):
    out = bytearray()
    for r in cell:
        lo = hi = 0
        for x, v in enumerate(r):
            if v & 1: lo |= 0x80 >> x
            if v & 2: hi |= 0x80 >> x
        out += bytes([lo, hi])
    return bytes(out)
static, ids = [], {}
mapA = np.zeros((18, 32), np.uint8); mapB = mapA.copy(); attA = mapA.copy(); attB = mapA.copy()
for r in range(18):
    for c in range(20):
        if r == 1: mapA[r, c] = mapB[r, c] = 144 + c; continue
        if r == 16: mapA[r, c] = mapB[r, c] = 164 + c; continue
        if 3 <= r < 15 and 4 <= c < 16:
            mapA[r, c] = mapB[r, c] = (r - 3) * 12 + (c - 4); attA[r, c] = 1; attB[r, c] = 1 | 8; continue
        tb = tile_bytes(img[r * 8:r * 8 + 8, c * 8:c * 8 + 8])
        if tb not in ids: ids[tb] = 184 + len(static); static.append(tb)
        mapA[r, c] = mapB[r, c] = ids[tb]; attA[r, c] = attB[r, c] = 1
assert 184 + len(static) <= 256
sec("StaticTiles", 0, b"".join(static))
sec("MapA", 0, mapA.tobytes()); sec("MapB", 0, mapB.tobytes()); sec("AttrA", 0, attA.tobytes()); sec("AttrB", 0, attB.tobytes())
# canvas-kolomtabel: per tegelkolom de stippen die erin vallen: n, dan n x (stipkolom, masker)
col = bytearray()
for tc in range(12):
    ent = {}
    for b in range(8):
        px = tc * 8 + b
        q = px - OFF
        if q < 0 or q >= S * PITCH or q % PITCH == PITCH - 1: continue   # marge of tussenruimte
        ent.setdefault(q // PITCH, 0); ent[q // PITCH] |= 0x80 >> b
    col += bytes([len(ent)]) + b"".join(bytes([k, m]) for k, m in sorted(ent.items()))
sec("ColTab", 0, col)
# rijtabel: per pixelrij van het canvas de offset (stiprij * S) in wTone, of $FFFF voor een lege rij
row = []
for py in range(96):
    q = py - OFF
    row.append(0xFFFF if q < 0 or q >= S * PITCH or q % PITCH == PITCH - 1 else (q // PITCH) * S)
sec("RowTab", 0, np.array(row, "<u2").tobytes())
# proportioneel font: per teken 8 bytes (breedte, 7 rijen links uitgelijnd), ASCII 32-127
extra = {"{": "...#|..##|.###|####|.###|..##|...#", "}": "#...|##..|###.|####|###.|##..|#...",
         "~": "..###..|.#####.|#######|#######|#######|.#####.|..###..", "^": "...|.#.|#.#|###|#.#|#.#|...",
         "|": ".....|.....|#####|#####|#####|.....|.....", " ": "..|..|..|..|..|..|.."}
font = bytearray()
for ch in range(32, 128):
    g = extra.get(chr(ch)) or FONT.get(chr(ch))
    if g is None: font += bytes(8); continue
    rows = g.split("|")
    if chr(ch) not in extra:                          # lege kolommen links en rechts weg
        cols = [x for x in range(5) if any(r[x] == "#" for r in rows)] or [0]
        rows = [r[min(cols):max(cols) + 1] for r in rows]
    w = len(rows[0]); font += bytes([w])
    for r in rows:
        font.append(sum(0x80 >> x for x, p in enumerate(r) if p == "#"))
sec("Font", 0, font)
def rgb(c): return (c[0] >> 3) | (c[1] >> 3) << 5 | (c[2] >> 3) << 10
PAPER, LIGHT, MID, DARK = (224, 248, 208), (136, 192, 112), (52, 104, 86), (8, 24, 32)
pal = [DARK, MID, LIGHT, PAPER] + [PAPER, LIGHT, MID, DARK]   # palet 0: UI, palet 1: canvas
sec("Palettes", 0, b"".join(rgb(c).to_bytes(2, "little") for c in pal))
Image.fromarray(np.array([PAPER, LIGHT, MID, DARK], np.uint8)[img]).resize((480, 432), Image.NEAREST).save(OUT / "static_preview.png")
tiles = b""
(OUT / "data.asm").write_text("\n".join(asm))
(OUT / "consts.inc").write_text(f"DEF K1 EQU {gb.K1}\nDEF K2 EQU {gb.K2}\nDEF K3 EQU {gb.K3}\nDEF STATIC_LEN EQU {len(static) * 16}\n"
    f"DEF NH EQU {H}\nDEF NB EQU {NB}\nDEF NB16 EQU {H // 16}\nDEF DPIX EQU {gbref.tr.D}\nDEF NB16D EQU {gbref.tr.D // 16}\nDEF W1_BANK EQU {banks[0][0]}\nDEF W2_BANK EQU {banks[1][0]}\nDEF W3_BANK EQU {banks[2][0]}\n"
    f"DEF CPB_H EQU {banks[0][1]}\nDEF CPB_D EQU {banks[2][1]}\nDEF BIAS1_BANK EQU {BIAS1_BANK}\n"
    f"DEF BIAS_SHIFT EQU {int(np.log2(BPER))}\nDEF BIAS_PAGES_LOG2 EQU {int(np.log2(BSTR // 256))}\n")
print("klaar:", len(asm), "secties, K", gb.K1, gb.K2, gb.K3, "breedte", H, "laatste bank", bank - 1)
