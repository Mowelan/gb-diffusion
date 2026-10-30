"""Hoe goed zijn de tekeningen van de Game Boy-rekenregels? Per categorie N plaatjes via gbref:
herkend (classifier), FID op classifier-kenmerken tegen echte testtekeningen (lager = beter), losse spikkels."""
import sys, numpy as np, gbref, judge
from prep import CATS

import os
SEED0 = int(os.environ.get("SEED0", 50000))
REAL = None
def real(c, n):
    global REAL
    if REAL is None:
        d = np.load(judge.HERE / "data" / gbref.tr.DATA); REAL = (d["xt"].reshape(-1, gbref.tr.D) / 127.5 - 1, d["yt"])
    return REAL[0][REAL[1] == c][:n]

def to_ink(x):  # zoals op het scherm: 4 tinten, terug naar -1..1
    t = np.clip(((x + 32) * 3 + 32) >> 6, 0, 3); return t / 1.5 - 1

def evaluate(gb, n=300, tag=""):
    m = judge.load(); rows = []
    for c in range(8):
        imgs = np.array([to_ink(gb.sample(c, SEED0 + c * 1000 + k)[-1]) for k in range(n)])
        acc, _ = judge.score(m, imgs, c)
        rows.append((acc, judge.fid(judge.feats(m, imgs), judge.feats(m, to_ink(np.round(real(c, 2000) * 32).astype(int)))), judge.speckle(imgs)))
    r = np.array(rows)
    print(f"{tag}: herkend {r[:,0].mean():.3f}  FID {r[:,1].mean():.1f}  spikkels {r[:,2].mean():.3f}")
    print("   ", "  ".join(f"{CATS[c]} {rows[c][1]:.1f}" for c in range(8)))
    return r

if __name__ == "__main__":
    evaluate(gbref.GB(*sys.argv[1:2]), tag=sys.argv[1] if len(sys.argv) > 1 else "huidig")
