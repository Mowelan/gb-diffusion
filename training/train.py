"""Klein diffusiemodel (MLP) voor 16x16 Quick, Draw!-tekeningen, ontworpen voor de Game Boy:
- vast schema van T stappen, dus tijd en categorie zijn alleen een bias-tabel in laag 1,
- het netwerk voorspelt het schone plaatje x0; de DDIM-stap rekent daarmee terug."""
import sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
A = sys.argv[1:] if Path(sys.argv[0]).name == "train.py" else []
S = int(__import__("os").environ.get("GB_S", 16))          # zijde van het plaatje
T, H, NC, D = 16, int(A[0]) if A else int(__import__("os").environ.get("GB_H", 256)), 8, S * S
DATA = "qd16_big.npz" if S == 16 else f"qd{S}_big.npz"
STEPS = int(A[1]) if len(A) > 1 else 20000
dev = "mps" if torch.backends.mps.is_available() else "cpu"

# cosinus-schema: abar[t] = deel signaal dat overblijft na t ruisstappen (t=1..T), abar[0]=1
s = np.arange(T + 1) / T
abar = np.cos((s + 0.008) / 1.008 * np.pi / 2) ** 2; abar = abar / abar[0]; abar[-1] = 0.0
ABAR = torch.tensor(abar, dtype=torch.float32)

class Net(nn.Module):
    def __init__(s):
        super().__init__()   # categorie NC = 'geen categorie' (voor guidance)
        s.l1 = nn.Linear(D, H); s.cb = nn.Embedding(NC + 1, H); s.tb = nn.Embedding(T + 1, H)
        s.l2 = nn.Linear(H, H); s.l3 = nn.Linear(H, D)
    def forward(s, x, t, c):
        h = torch.relu(s.l1(x) + s.cb(c) + s.tb(t))
        h = torch.relu(s.l2(h))
        return torch.tanh(s.l3(h))

def sample(net, c, seed=0, w=0.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(c), D, generator=g).to(dev); traj = [x.cpu()]
    for t in range(T, 0, -1):
        tt = torch.full((len(c),), t, device=dev)
        x0 = net(x, tt, c)
        if w: x0 = (x0 + w * (x0 - net(x, tt, torch.full_like(c, NC)))).clamp(-1, 1)
        a, ap = ABAR[t].item(), ABAR[t - 1].item()
        eps = (x - a ** .5 * x0) / max(1 - a, 1e-8) ** .5
        x = ap ** .5 * x0 + (1 - ap) ** .5 * eps; traj.append(x.cpu())
    return traj

if __name__ == "__main__":
    d = np.load(HERE / "data" / "qd16.npz")
    X = torch.tensor(d["x"], dtype=torch.float32).view(-1, D) / 127.5 - 1; Y = torch.tensor(d["y"]).long()
    X, Y = X.to(dev), Y.to(dev)
    net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS)
    t0 = time.time()
    for it in range(STEPS):
        i = torch.randint(0, len(X), (512,), device=dev)
        x0, c = X[i], Y[i]; t = torch.randint(1, T + 1, (512,), device=dev)
        a = ABAR.to(dev)[t].unsqueeze(1)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * torch.randn_like(x0)
        c = torch.where(torch.rand(512, device=dev) < 0.1, torch.full_like(c, NC), c)
        loss = ((net(xt, t, c) - x0) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 2000 == 0: print(it, round(loss.item(), 4), f"{time.time() - t0:.0f}s", flush=True)
    torch.save(net.state_dict(), HERE / "model" / f"mlp{H}.pt")
    print("params", sum(p.numel() for p in net.parameters()))
