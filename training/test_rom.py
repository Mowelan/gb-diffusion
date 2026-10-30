"""Test de ROM in PyBoy: na elke stap moet wX bit voor bit gelijk zijn aan gbref.py. Meet ook de tijd."""
import sys, time
from pathlib import Path
import numpy as np
from pyboy import PyBoy
import gbref

import os
ROM = os.environ.get("GB_ROM", str(Path(__file__).resolve().parent.parent / "rom" / "build" / "diffusion.gbc"))
cats = [int(a) for a in sys.argv[1:]] or [3]
gb = gbref.GB()
SYM = {l.split()[1]: int(l.split()[0].split(":")[1], 16) for l in open(ROM.replace(".gbc", ".sym")) if l[:1] != ";" and len(l.split()) == 2}
CAT, SEED, ELAPSED = SYM["wCat"], SYM["wSeed"], SYM["wElapsed"]
pb = PyBoy(ROM, window="null", cgb=True)
pb.set_emulation_speed(0)
mem = pb.memory
state = {"steps": [], "done": False}
pb.hook_register(None, "StepDone", lambda s: (s["steps"].append(np.array([mem[SYM['wX'] + j] for j in range(gbref.tr.D)], np.int64)), len(s["steps"]) == 8 and pb.screen.image.save("rom_mid.png")), state)
pb.hook_register(None, "GenDone", lambda s: s.__setitem__("done", True), state)
for _ in range(120): pb.tick()
ok_all = True
for c in cats:
    while mem[CAT] != c:                       # met rechts naar de categorie
        pb.button("right"); [pb.tick() for _ in range(6)]
    state["steps"].clear(); state["done"] = False
    pb.button("a"); t0 = time.time()
    while not state["done"]: pb.tick()
    seed = mem[SEED] | mem[SEED + 1] << 8; frames = mem[ELAPSED] | mem[ELAPSED + 1] << 8
    ref = gb.sample(c, seed)
    signed = [np.where(s > 127, s - 256, s) for s in state["steps"]]
    bad = [i for i, (a, b) in enumerate(zip(signed, ref[1:])) if not np.array_equal(a, b)]
    ok = not bad and len(signed) == 16; ok_all &= ok
    print(f"categorie {c}, zaad {seed}: {'GELIJK' if ok else 'VERSCHIL bij stap ' + str(bad[:3])}, {frames} frames = {frames / 59.73:.1f} s op een GBC")
    if bad:
        i = bad[0]; d = np.nonzero(signed[i] != ref[i + 1])[0]
        print("  pixels", d[:8], "rom", signed[i][d[:8]], "ref", ref[i + 1][d[:8]])
        print("  vorige stap gelijk:", i == 0 or np.array_equal(signed[i - 1], ref[i]), "ruis gelijk:", np.array_equal(ref[0], ref[0]))
    [pb.tick() for _ in range(10)]
    pb.screen.image.save(f"rom_c{c}.png")
pb.stop()
print("ALLES GELIJK" if ok_all else "NIET GELIJK")
