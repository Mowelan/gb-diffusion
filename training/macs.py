"""Rekenwerk per tekening volgens de ROM-regels: de ROM slaat invoer 0 over, dus per laag telt
(aantal niet-nul invoeren) x (aantal uitgangen). Snelheid volgt uit de gemeten ROM-tijd van gb_v2."""
import sys, os, numpy as np, gbref

def count(gb, seeds=40):
    tot = []
    for k in range(seeds):
        c = k % 8; x = gb.noise(70000 + k); m = 0
        for i in range(16):
            v = np.clip(((x + 4) >> 3) + 4, gbref.VMIN, gbref.VMAX); H = gb.W1.shape[0]
            acc = gb.BIAS1[c, i] + gb.W1 @ v; h1 = np.clip(acc >> gb.K1, 0, 15)
            acc = gb.B2 + gb.W2 @ h1; h2 = np.clip(acc >> gb.K2, 0, 15)
            m += np.count_nonzero(v) * H + np.count_nonzero(h1) * H + np.count_nonzero(h2) * gb.W3.shape[0]
            x = gb.step(x, c, i)
        tot.append(m)
    return np.mean(tot)

if __name__ == "__main__":
    for p in sys.argv[1:]:
        gb = gbref.GB(p); print(p, "breedte", gb.W1.shape[0], "MACs per tekening", int(count(gb)), "max acc", gb.max, flush=True)
