"""Zelfde Game Boy-MLP als train.py, maar op 4x zoveel tekeningen, langer, met EMA."""
import sys, time, copy
import numpy as np, torch
import train as tr

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
OUT = sys.argv[2] if len(sys.argv) > 2 else "mlp_big.pt"
d = np.load(tr.HERE / "data" / tr.DATA)
X = (torch.tensor(d["x"], dtype=torch.float32).view(-1, tr.D) / 127.5 - 1).to(tr.dev); Y = torch.tensor(d["y"]).long().to(tr.dev)
net = tr.Net().to(tr.dev); ema = copy.deepcopy(net)
opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS, pct_start=0.05)
AB = tr.ABAR.to(tr.dev); t0 = time.time()
for it in range(STEPS):
    i = torch.randint(0, len(X), (1024,), device=tr.dev); x0, c = X[i], Y[i]
    t = torch.randint(1, tr.T + 1, (1024,), device=tr.dev); a = AB[t].unsqueeze(1)
    xt = a.sqrt() * x0 + (1 - a).sqrt() * torch.randn_like(x0)
    c = torch.where(torch.rand(1024, device=tr.dev) < 0.1, torch.full_like(c, tr.NC), c)
    loss = ((net(xt, t, c) - x0) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        for pe, pn in zip(ema.parameters(), net.parameters()): pe.lerp_(pn, 5e-4 if it > 2000 else 1.0)
    if it % 50000 == 0: print(it, round(loss.item(), 4), f"{time.time() - t0:.0f}s", flush=True)
torch.save(ema.state_dict(), tr.HERE / "model" / OUT)
