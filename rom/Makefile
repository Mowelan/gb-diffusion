# Diffusie op de GBC. Gewichten en tabellen komen uit export.py in de run-map.
ROM = build/diffusion.gbc

$(ROM): main.asm build/data.asm build/consts.inc
	rgbasm -o build/main.o main.asm
	rgblink -m build/diffusion.map -n build/diffusion.sym -o $(ROM) build/main.o
	rgbfix -C -v -p 0xFF -m 0x19 -t DIFFUSION28 $(ROM)
