"""Exacte rekenregels van de ROM (bit voor bit), plus export van alle tabellen.

Toestand x: int8, schaal 1/32. Alles in 16-bits sommen, en elke vermenigvuldiging is een opzoeking in
een producttabel T[v][w] = v * w (v = -24..23, w = int8) die de ROM bij het opstarten in WRAM-banken 1-6 zet.
  laag 1:  v_i = clamp(((x_i + 4) >> 3) + 4, -24, 23)      papier (x = -32) wordt v = 0 en slaat de ROM over
           acc_j = BIAS1[c][stap][j] + sum_i T[v_i][W1[j][i]];  h1 = clamp(acc >> K1, 0, 15)
  laag 2:  acc_j = B2[j] + sum_i T[h1_i][W2[j][i]];  h2 = clamp(acc >> K2, 0, 15)
  laag 3:  acc_j = B3[j] + sum_i T[h2_i][W3[j][i]];  x0 = TANH[clamp(acc >> K3, -128, 127)]
  DDIM:    x = clamp((A[stap] * x + B[stap] * x0) >> 7, -128, 127)
Begin-ruis: xorshift16 (7, 9, 8) vanaf het zaad, x_j = GAUSS[r & 255]."""
import sys
from pathlib import Path
from statistics import NormalDist
import numpy as np, torch
import train as tr

HERE = Path(__file__).resolve().parent
VMIN, VMAX = -24, 23
W1X = int(__import__("os").environ.get("GB_W1X", 0))   # 28x28: 1

def wrap16(a):
    a = np.asarray(a, np.int64); return ((a + 32768) & 0xFFFF) - 32768

class GB:
    def __init__(self, path=None):          # v1: gb_student.pt, v2: gb_v2.pt; GB_MODEL kiest
        path = path or __import__("os").environ.get("GB_MODEL", "gb_v2.pt")
        sd = torch.load(HERE / "model" / path, map_location="cpu")
        self.cal = sd.pop("_cal", None)        # vaste activatieschalen uit quantization-aware training
        g = lambda k: sd[k].double().numpy()
        def qw(w, extra=0):                  # extra: grovere stap, zodat de sommen in 16 bits blijven
            e = int(np.ceil(np.log2(np.abs(w).max() / 127))) + extra; sc = 2.0 ** e
            return np.clip(np.round(w / sc), -127, 127).astype(np.int64), sc
        W1, s1 = qw(g("l1.weight"), W1X); W2, s2 = qw(g("l2.weight")); W3, s3 = qw(g("l3.weight"))
        self.W1, self.W2, self.W3 = W1, W2, W3
        # activatieschalen ijken op een float-run (machten van 2)
        net = tr.Net(); net.load_state_dict(sd); net.eval(); A1, A2, A3 = [], [], []
        with torch.no_grad():
            c = torch.arange(800) % 8; x = torch.randn(800, tr.D)
            for t in range(tr.T, 0, -1):
                tt = torch.full((800,), t)
                h1 = torch.relu(net.l1(x) + net.cb(c) + net.tb(tt)); h2 = torch.relu(net.l2(h1)); o = net.l3(h2)
                A1.append(h1); A2.append(h2); A3.append(o)
                x0 = torch.tanh(o); a, ap = tr.ABAR[t].item(), tr.ABAR[t - 1].item()
                x = ap ** .5 * x0 + (1 - ap) ** .5 * (x - a ** .5 * x0) / max(1 - a, 1e-8) ** .5
        a1 = 2.0 ** np.round(np.log2(np.percentile(torch.cat(A1).numpy(), 99.5) / 15))
        a2 = 2.0 ** np.round(np.log2(np.percentile(torch.cat(A2).numpy(), 99.5) / 15))
        if self.cal is not None: a1, a2 = float(self.cal[0]), float(self.cal[1])
        u1 = s1 / 4                                             # eenheid van acc1: v loopt in stappen van 1/4
        self.K1 = int(round(np.log2(a1 / u1)))
        b = (g("l1.bias")[None, None] + g("cb.weight")[:8, None] + g("tb.weight")[None, 1:])   # [c][t-1][j]
        b = b[:, ::-1]                                           # op volgorde van de stappen (t = T..1)
        self.BIAS1 = np.round(b / u1).astype(np.int64) - 4 * W1.sum(1)[None, None]   # v = 4*x + 4
        u2 = s2 * a1; self.K2 = int(round(np.log2(a2 / u2))); self.B2 = np.round(g("l2.bias") / u2).astype(np.int64)
        u3 = s3 * a2; self.B3 = np.round(g("l3.bias") / u3).astype(np.int64)
        self.K3 = max(0, int(np.ceil(np.log2(4.0 / u3 / 128))))
        self.TANH = np.round(32 * np.tanh(np.arange(-128, 128) * u3 * 2 ** self.K3)).astype(np.int64)
        self.A, self.B = [], []
        for t in range(tr.T, 0, -1):
            a, ap = tr.ABAR[t].item(), tr.ABAR[t - 1].item()
            k1 = ((1 - ap) / max(1 - a, 1e-8)) ** .5; k2 = ap ** .5 - k1 * a ** .5
            self.A.append(int(round(k1 * 128))); self.B.append(int(round(k2 * 128)))
        nd = NormalDist()
        self.GAUSS = np.array([int(np.clip(round(32 * nd.inv_cdf((i + .5) / 256)), -127, 127)) for i in range(256)])
        self.TONE = np.array([min(3, max(0, ((x + 32) * 3 + 32) >> 6)) for x in range(-128, 128)])
        self.max = [0, 0, 0, 0]

    def T(self, v, w):
        return v * w

    def noise(self, seed):
        r = seed & 0xFFFF or 1; x = np.zeros(tr.D, np.int64)
        for j in range(tr.D):
            r ^= (r << 7) & 0xFFFF; r ^= r >> 9; r ^= (r << 8) & 0xFFFF; x[j] = self.GAUSS[r & 0xFF]
        return x

    def layer(self, W, bias, v):
        acc = bias + (W * v[None, :]).sum(1)
        assert np.abs(acc).max() < 32768     # tussensommen mogen overlopen: in 16 bits rekent de ROM modulo 2^16
        return acc

    def step(self, x, c, i):
        v = np.clip(((x + 4) >> 3) + 4, VMIN, VMAX)
        acc = self.layer(self.W1, self.BIAS1[c, i], v); self.max[0] = max(self.max[0], abs(acc).max())
        h = np.clip(acc >> self.K1, 0, 15)
        acc = self.layer(self.W2, self.B2, h); self.max[1] = max(self.max[1], abs(acc).max())
        h = np.clip(acc >> self.K2, 0, 15)
        acc = self.layer(self.W3, self.B3, h); self.max[2] = max(self.max[2], abs(acc).max())
        x0 = self.TANH[np.clip(acc >> self.K3, -128, 127) + 128]
        y = self.A[i] * x + self.B[i] * x0; self.max[3] = max(self.max[3], abs(y).max())
        return np.clip(y >> 7, -128, 127)

    def sample(self, c, seed):
        x = self.noise(seed); traj = [x]
        for i in range(tr.T):
            x = self.step(x, c, i); traj.append(x)
        return traj

if __name__ == "__main__":
    from PIL import Image
    import show
    gb = GB(); rows = []
    for c in range(8):
        rows.append(np.concatenate([np.pad(show.tile(gb.sample(c, c * 1000 + k + 1)[-1] / 32.0), ((2, 2), (2, 2), (0, 0)), constant_values=255) for k in range(10)], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(HERE / "gbref.png")
    print("K1 K2 K3", gb.K1, gb.K2, gb.K3, "max", gb.max, "A", gb.A, "B", gb.B)
    print("v-bereik bias1", gb.BIAS1.min(), gb.BIAS1.max(), "B2", gb.B2.min(), gb.B2.max(), "B3", gb.B3.min(), gb.B3.max())
