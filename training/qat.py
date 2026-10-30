"""Quantization-aware distillatie: de Game Boy-leerling (MLP 256) leert de guided uitkomst van de U-Net-leraar,
terwijl de forward precies rekent zoals de ROM (int8-gewichten, invoer in kwartstappen, 4-bit activaties,
tanh-tabel, toestand int8). Afronden gaat met straight-through gradiënten.
De helft van de trainingsvoorbeelden komt uit trajecten die de leerling zelf (gequantiseerd) tekent."""
import sys, time
import numpy as np, torch
import train as tr, gbref

W_ARG = sys.argv[1] if len(sys.argv) > 1 else "2.0"   # een getal, of 8 getallen met komma (per categorie)
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
OUT = sys.argv[3] if len(sys.argv) > 3 else "qat.pt"
TEACH = sys.argv[5] if len(sys.argv) > 5 else "unet"
POOL = (sys.argv[4] if len(sys.argv) > 4 else "1") == "1"   # eigen trajecten meenemen
dev = tr.dev

def ste(x, q): return x + (q - x).detach()

def p2(x): return 2.0 ** torch.ceil(torch.log2(x))

class QStudent(torch.nn.Module):
    def __init__(s, net, a1, a2):
        super().__init__(); s.net = net; s.a1, s.a2 = a1, a2
    def qw(s, w, extra=0):
        sc = p2(w.detach().abs().max() / 127) * 2 ** extra; return ste(w, torch.clamp(torch.round(w / sc), -127, 127) * sc), sc
    def forward(s, x, t, c):                       # x: al gequantiseerd (veelvoud van 1/32)
        n = s.net
        xq = torch.round(x * 32)
        v = torch.clamp(torch.floor((xq + 4) / 8) + 4, -24, 23)
        xin = ste(x, (v - 4) / 4)
        W1, s1 = s.qw(n.l1.weight, gbref.W1X); W2, s2 = s.qw(n.l2.weight); W3, s3 = s.qw(n.l3.weight)
        pre = xin @ W1.T + n.l1.bias + n.cb(c) + n.tb(t)
        h = torch.clamp(pre / s.a1, 0, 15); h = ste(h, torch.floor(h)) * s.a1
        pre = h @ W2.T + n.l2.bias
        h = torch.clamp(pre / s.a2, 0, 15); h = ste(h, torch.floor(h)) * s.a2
        pre = h @ W3.T + n.l3.bias
        u3 = s3 * s.a2; k3 = max(0.0, float(torch.ceil(torch.log2(4.0 / u3 / 128)))); st = u3 * 2 ** k3
        idx = torch.clamp(torch.floor(pre / st), -128, 127)
        xq0 = torch.round(32 * torch.tanh(idx * st)) / 32
        return ste(torch.tanh(pre), xq0)

A = [int(round(((1 - tr.ABAR[t - 1].item()) / max(1 - tr.ABAR[t].item(), 1e-8)) ** .5 * 128)) for t in range(tr.T, 0, -1)]
B = [int(round((tr.ABAR[t - 1].item() ** .5 - ((1 - tr.ABAR[t - 1].item()) / max(1 - tr.ABAR[t].item(), 1e-8)) ** .5 * tr.ABAR[t].item() ** .5) * 128)) for t in range(tr.T, 0, -1)]

def gb_noise(n, g):  # zoals de ROM: gaussische waarden afgerond op 1/32
    return torch.clamp(torch.round(torch.randn(n, tr.D, generator=g) * 32), -127, 127).to(dev) / 32

@torch.no_grad()
def rollout(q, n, g):
    """Gequantiseerd tekenen; geeft per stap (x, t, c)."""
    c = torch.randint(0, 8, (n,), generator=g).to(dev); x = gb_noise(n, g); out = []
    for i in range(tr.T):
        t = torch.full((n,), tr.T - i, device=dev); out.append((x, t, c))
        x0 = q(x, t, c)
        y = A[i] * torch.round(x * 32) + B[i] * torch.round(x0 * 32)
        x = torch.clamp(torch.floor(y / 128), -128, 127) / 32
    return out

if __name__ == "__main__":
    import gbref
    INIT = sys.argv[6] if len(sys.argv) > 6 else "gb_student.pt"
    old = gbref.GB(INIT)                               # schalen geijkt op het startmodel
    sd = {k: v for k, v in torch.load(tr.HERE / "model" / INIT, map_location="cpu").items() if k != "_cal"}
    s1 = 2.0 ** (np.ceil(np.log2(sd["l1.weight"].abs().max().item() / 127)) + gbref.W1X); s2 = 2.0 ** np.ceil(np.log2(sd["l2.weight"].abs().max().item() / 127))
    a1 = s1 / 4 * 2 ** old.K1; a2 = s2 * a1 * 2 ** old.K2
    print("a1", a1, "a2", a2, flush=True)
    net = tr.Net().to(dev); net.load_state_dict(sd); q = QStudent(net, a1, a2)
    if TEACH == "unet":
        import teacher2
        teach = teacher2.UNet(C=32).to(dev); teach.load_state_dict(torch.load(tr.HERE / "model" / "teacher2.pt", map_location=dev))
    else:
        teach = tr.Net().to(dev); teach.load_state_dict(torch.load(tr.HERE / "model" / TEACH, map_location=dev))
    teach.eval()
    d = np.load(tr.HERE / "data" / tr.DATA)
    X = (torch.tensor(d["x"], dtype=torch.float32).view(-1, tr.D) / 127.5 - 1).to(dev); Y = torch.tensor(d["y"]).long().to(dev)
    WC = torch.tensor([float(v) for v in W_ARG.split(",")] * (8 if "," not in W_ARG else 1), device=dev)
    net.load_state_dict({k: v for k, v in torch.load(tr.HERE / "model" / INIT, map_location="cpu").items() if k != "_cal"})
    opt = torch.optim.Adam(net.parameters(), 3e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    AB = tr.ABAR.to(dev); g = torch.Generator().manual_seed(0); pool = None; t0 = time.time()
    for it in range(STEPS):
        if it % 250 == 0:                                  # nieuwe eigen trajecten
            r = rollout(q, 256, g)
            pool = [torch.cat([e[k] for e in r]) for k in range(3)]
        i = torch.randint(0, len(X), (128,), device=dev); c1 = Y[i]
        t1 = torch.randint(1, tr.T + 1, (128,), device=dev); a = AB[t1].unsqueeze(1)
        x1 = torch.clamp(torch.round((a.sqrt() * X[i] + (1 - a).sqrt() * torch.randn_like(X[i])) * 32), -128, 127) / 32
        if POOL:
            j = torch.randint(0, len(pool[0]), (128,), device=dev)
            x = torch.cat([x1, pool[0][j]]); t = torch.cat([t1, pool[1][j]]); c = torch.cat([c1, pool[2][j]])
        else:
            x, t, c = x1, t1, c1
        with torch.no_grad():
            pc = teach(x, t, c); pu = teach(x, t, torch.full_like(c, tr.NC))
            tgt = (pc + WC[c].unsqueeze(1) * (pc - pu)).clamp(-1, 1)
        loss = ((q(x, t, c) - tgt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 5000 == 0: print(it, round(loss.item(), 4), f"{time.time() - t0:.0f}s", flush=True)
    out = {k: v.cpu() for k, v in net.state_dict().items()}; out["_cal"] = torch.tensor([a1, a2], dtype=torch.float64)
    torch.save(out, tr.HERE / "model" / OUT); print("opgeslagen", OUT)
