"""Quick, Draw! (Google, CC BY 4.0) naar kleine 4-tinten-plaatjes voor de Game Boy.
Leest de eerste 30.000 tekeningen per categorie (28x28) en schaalt ze naar SIZE x SIZE."""
import ast, sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
CATS = ["cat", "house", "tree", "fish", "car", "apple", "sun", "smiley_face"]
N = 30000

def load(cat):
    b = (HERE / "data" / f"{cat}.part").read_bytes()
    hl = int.from_bytes(b[8:10], "little"); o = 10 + hl
    return np.frombuffer(b[o:o + N * 784], np.uint8).reshape(N, 28, 28)

def shrink(a, size):
    return np.stack([np.asarray(Image.fromarray(x).resize((size, size), Image.BOX)) for x in a])

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    X, Y = [], []
    for i, c in enumerate(CATS):
        X.append(shrink(load(c), size)); Y.append(np.full(N, i, np.uint8))
    np.savez_compressed(HERE / "data" / f"qd{size}.npz", x=np.concatenate(X), y=np.concatenate(Y))
    # contactvel: 8 categorieen x 12 voorbeelden, 4 tinten
    rows = []
    for i in range(len(CATS)):
        xs = X[i][:12] // 64 * 85
        rows.append(np.concatenate(list(xs), 1))
    Image.fromarray((255 - np.concatenate(rows, 0)).astype(np.uint8)).resize((12 * size * 4, 8 * size * 4), Image.NEAREST).save(HERE / f"sheet{size}.png")
    print("klaar", size)
