"""Meer data: 120.000 tekeningen per categorie, 16x16. De laatste 10.000 per categorie blijven apart (test)."""
import numpy as np
from pathlib import Path
import sys
from prep import CATS, shrink, HERE
S = int(sys.argv[1]) if len(sys.argv) > 1 else 16   # 28 = oorspronkelijke bitmaps, niet geschaald
N = 120000
X, Y = [], []
for i, c in enumerate(CATS):
    b = (HERE / "data" / f"{c}.big").read_bytes(); hl = int.from_bytes(b[8:10], "little"); o = 10 + hl
    print(c, b[10:10 + hl].decode().strip())
    a = np.frombuffer(b[o:o + N * 784], np.uint8).reshape(N, 28, 28)
    X.append(a if S == 28 else shrink(a, S)); Y.append(np.full(N, i, np.uint8))
X, Y = np.stack(X), np.stack(Y)
np.savez_compressed(HERE / "data" / ("qd16_big.npz" if S == 16 else f"qd{S}_big.npz"), x=X[:, :-10000].reshape(-1, S, S), y=Y[:, :-10000].reshape(-1),
                    xt=X[:, -10000:].reshape(-1, S, S), yt=Y[:, -10000:].reshape(-1))
print("klaar")
