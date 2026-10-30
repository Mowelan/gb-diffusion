"""Meetlat: een kleine CNN die herkent welke categorie een 16x16-tekening is (getraind op echte tekeningen).
score(plaatjes, categorie) = deel dat als de bedoelde categorie wordt herkend."""
from pathlib import Path
import os
import numpy as np, torch, torch.nn as nn
HERE = Path(__file__).resolve().parent
S = int(os.environ.get("GB_S", 16))
DATA = "qd16_big.npz" if S == 16 else f"qd{S}_big.npz"
JUDGE = "judge.pt" if S == 16 else f"judge{S}.pt"
dev = "mps" if torch.backends.mps.is_available() else "cpu"

def make():
    return nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                         nn.Linear(64 * (S // 4) ** 2, 128), nn.ReLU(), nn.Linear(128, 8))

def load():
    m = make().to(dev); m.load_state_dict(torch.load(HERE / "model" / JUDGE)); m.eval(); return m

def score(m, imgs, c):
    """imgs: N x 256 in -1..1 (inkt = 1). Geeft (herkend-percentage, gemiddelde zekerheid)."""
    with torch.no_grad():
        p = torch.softmax(m(torch.tensor(np.asarray(imgs), dtype=torch.float32).view(-1, 1, S, S).to(dev)), 1).cpu().numpy()
    return float((p.argmax(1) == c).mean()), float(p[:, c].mean())

if __name__ == "__main__":
    d = np.load(HERE / "data" / DATA)
    X = torch.tensor(d["x"], dtype=torch.float32).view(-1, 1, S, S) / 127.5 - 1; Y = torch.tensor(d["y"]).long()
    Xt = torch.tensor(d["xt"], dtype=torch.float32).view(-1, 1, S, S) / 127.5 - 1; Yt = torch.tensor(d["yt"]).long()
    m = make().to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for it in range(15000):
        i = torch.randint(0, len(X), (256,))
        loss = nn.functional.cross_entropy(m(X[i].to(dev)), Y[i].to(dev)); opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); torch.save(m.state_dict(), HERE / "model" / JUDGE)
    with torch.no_grad():
        pr = torch.cat([m(Xt[k:k + 4000].to(dev)).argmax(1).cpu() for k in range(0, len(Xt), 4000)])
    print("echte testtekeningen herkend:", f"{(pr == Yt).float().mean():.3f}", "per categorie:", [f"{(pr[Yt == c] == c).float().mean():.2f}" for c in range(8)])

def feats(m, imgs):
    with torch.no_grad():
        return m[:-1](torch.tensor(np.asarray(imgs), dtype=torch.float32).view(-1, 1, S, S).to(dev)).cpu().numpy().astype(np.float64)

def fid(f1, f2):
    """Fréchet-afstand tussen twee verzamelingen kenmerken (lager = meer als echte tekeningen)."""
    from numpy.linalg import eigh
    m1, m2 = f1.mean(0), f2.mean(0); c1, c2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    w, v = eigh(c1); s1 = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    w2 = np.linalg.eigvals(s1 @ c2 @ s1)
    return float(((m1 - m2) ** 2).sum() + np.trace(c1) + np.trace(c2) - 2 * np.sqrt(np.clip(w2.real, 0, None)).sum())

def speckle(imgs):
    """Deel van de inktpixels zonder inkt-buur (losse spikkels)."""
    a = (np.asarray(imgs).reshape(-1, S, S) > 0)
    p = np.pad(a, ((0, 0), (1, 1), (1, 1)))
    nb = sum(np.roll(np.roll(p, dy, 1), dx, 2) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)[:, 1:-1, 1:-1]
    return float((a & (nb == 0)).sum() / max(a.sum(), 1))
