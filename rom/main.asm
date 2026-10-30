; Diffusie op de Game Boy Color, 28x28: een klein neuraal netwerk tekent 28x28-plaatjes uit ruis, in 16 stappen.
; (De 16x16-versie staat in ../gb-diffusion; dit is dezelfde opzet met lussen over DPIX = 784 pixels.)
;
; Rekenregels: zie gbref.py in de run-map (de ROM moet daar bit voor bit gelijk aan zijn).
; Elke vermenigvuldiging is een opzoeking in de producttabel T[v][w] = v * w (v = -24..23, w = int8).
; Die staat in WRAM-banken 1-6 ($D000-$DFFF): rij r = v + 24, bank 1 + r/8, lage bytes op pagina
; $D0 + (r%8)*2, hoge bytes op de pagina erna. De gewichten staan kolomsgewijs in ROM-banken (64 kolommen
; die op een veelvoud van 128 bytes beginnen, zoveel hele kolommen per bank als er passen), zodat ROM-bank en
; WRAM-bank tegelijk in beeld zijn en de binnenste lus per vermenigvuldiging alleen twee opzoekingen en een
; 16-bits optelling doet. Invoer 0 slaat een kolom over. Breedte H en de bankindeling komen uit consts.inc.

DEF rP1    EQU $FF00
DEF rDIV   EQU $FF04
DEF rIF    EQU $FF0F
DEF rLCDC  EQU $FF40
DEF rLY    EQU $FF44
DEF rKEY1  EQU $FF4D
DEF rVBK   EQU $FF4F
DEF rHDMA1 EQU $FF51
DEF rHDMA2 EQU $FF52
DEF rHDMA3 EQU $FF53
DEF rHDMA4 EQU $FF54
DEF rHDMA5 EQU $FF55
DEF rBCPS  EQU $FF68
DEF rBCPD  EQU $FF69
DEF rSVBK  EQU $FF70
DEF rIE    EQU $FFFF
DEF rROMB0 EQU $2000

DEF LCDC_ON EQU %10010001       ; LCD aan, tegels op $8000, map $9800, BG aan (bit 3 wisselt de map)

DEF PAD_RIGHT EQU $10
DEF PAD_LEFT  EQU $20
DEF PAD_A     EQU $01

INCLUDE "build/consts.inc"      ; K1-K3, STATIC_LEN, NH, NB, banken

; Scherm: canvas 96x96 (16x16 stippen van 5 px, 1 px tussenruimte) op een papieren paneel, kop- en voetregel
; met proportionele tekst. Het canvas staat dubbel: tegels 0-143 in VRAM-bank 0 (map $9800) en bank 1
; (map $9C00). Een nieuwe stap gaat naar de onzichtbare kant en wordt in VBlank omgeschakeld.
DEF HDR_TILES EQU $8000 + 144 * 16
DEF FTR_TILES EQU $8000 + 164 * 16
DEF CANVAS_BUF EQU $D000        ; WRAM-bank 7: de canvastegels zoals ze in VRAM komen (2304 bytes)

; WRAM-bank 0 is altijd zichtbaar: alles wat de laag-lus leest of schrijft staat hier. De rest staat in bank 7,
; die zichtbaar is behalve tijdens de laag-lus (die schakelt banken 1-6 voor de producttabel en zet 7 terug).
SECTION "Acc", WRAM0[$C000]
wAcc:: ds 2 * DPIX              ; tot DPIX sommen, int16 little-endian
SECTION "BufV", WRAM0[$C620]
wV::                            ; invoer laag 1, later uitgang laag 2, later tinten (nooit tegelijk nodig)
wH2::
wTone:: ds DPIX
SECTION "BufH1", WRAM0[$C930]
wH1:: ds NH
SECTION "BufX0", WRAM0[$CB30]
wX0:: ds DPIX                   ; voorspeld schoon plaatje
SECTION "BufHdr", WRAMX[$D900], BANK[7]
wHdr:: ds 320                   ; kopregel als 20 tegels
wFtr:: ds 320                   ; voetregel als 20 tegels
SECTION "BufX", WRAMX[$DB80], BANK[7]
wX::  ds DPIX                   ; toestand, int8 met schaal 1/32

SECTION "Vars", WRAM0[$CE40]
wFrames::  dw
wCat::     db
wSeed::    dw
wStep::    db
wT0::      dw
wElapsed:: dw                   ; frames voor het hele plaatje
wJoyPrev:: db
wShown::   db                   ; zichtbare canvas-bank (0 of 1)
wGens::    db                   ; aantal gemaakte plaatjes (voor de test)
wState::   db                   ; 0 = kiezen, 1 = bezig, 2 = klaar
wStr::     ds 16                ; tekst in opbouw

SECTION "HVars", HRAM
hI::     db                     ; teller in de DDIM-lus
hInLo::  db                     ; laag: invoerpointer, aantal invoeren, ROM-bank, kolommen per bank,
hInHi::  db                     ;       blokken van 128 per kolom, kolomlengte (in deze volgorde, zie LayerN)
hCntLo:: db
hCntHi:: db
hBank::  db
hCPB::   db
hBlocks:: db
hStrLo:: db
hStrHi:: db
hColLo:: db                     ; kolom in de bank
hColHi:: db
hLeft::  db                     ; kolommen over in deze bank
hM::     db
hXp::    dw                     ; DDIM: pointer in wX
hX0p::   dw                     ;       pointer in wX0
hRP::    dw                     ; canvas: begin van de stiprij in wTone
hK::     db
hA::     db
hB::     db
hTX::    db                     ; tekst: x in pixels
hTC::    db                     ; tekst: kleur 0-3
hTBase:: dw                     ; tekst: buffer (20 tegels)
hGW::    db                     ; glyph: breedte
hGPtr::  dw                     ; glyph: rij-pointer
hRow::   db
hLo::    db
hHi::    db
hN::     db
hDst::   dw
hPY::    db                     ; canvas: pixelrij
hSub::   db                     ; canvas: rij binnen een stip (0-5)

SECTION "VBlank", ROM0[$40]
    jp VBlankHandler

SECTION "Header", ROM0[$100]
    nop
    jp Start
    ds $150 - @, 0

SECTION "Main", ROM0
Start:
    cp $11                      ; A = $11 bij het opstarten van een GBC
    jr z, .cgb
.geenGBC:
    halt
    jr .geenGBC
.cgb:
    di
    ld sp, $D000
    ld a, 1                     ; double speed
    ldh [rKEY1], a
    ld a, $30
    ldh [rP1], a
    stop
.wacht:
    ldh a, [rLY]
    cp 144
    jr c, .wacht
    xor a
    ldh [rLCDC], a
    ld a, 1                     ; VRAM-bank 1: canvastegels leeg, attributen map B
    ldh [rVBK], a
    ld hl, $8000
    ld bc, $1800
    xor a
    call Memset
    ld hl, AttrA
    ld de, $9800
    ld bc, 18 * 32
    call Memcpy
    ld hl, AttrB
    ld de, $9C00
    ld bc, 18 * 32
    call Memcpy
    xor a                       ; VRAM-bank 0: tegels, maps
    ldh [rVBK], a
    ld hl, $8000
    ld bc, $1800
    xor a
    call Memset
    ld hl, StaticTiles
    ld de, $8000 + 184 * 16
    ld bc, STATIC_LEN
    call Memcpy
    ld hl, MapA
    ld de, $9800
    ld bc, 18 * 32
    call Memcpy
    ld hl, MapB
    ld de, $9C00
    ld bc, 18 * 32
    call Memcpy
    ld a, $80                   ; BG-paletten 0 en 1, automatisch ophogen
    ldh [rBCPS], a
    ld hl, Palettes
    ld b, 16
.pal:
    ld a, [hl+]
    ldh [rBCPD], a
    dec b
    jr nz, .pal

    call BuildProducts
    ld a, 7                     ; canvasbuffer leeg
    ldh [rSVBK], a
    ld hl, CANVAS_BUF
    ld bc, 2304
    xor a
    call Memset

    ld hl, $C000                ; WRAM-bank 0 leeg (behalve de stapel)
    ld bc, $0F00
    xor a
    call Memset
    ld a, 3                     ; begin bij "fish"
    ld [wCat], a
    call DrawHeader
    call DrawFooter
    call CopyStripsNow
    ld a, LCDC_ON
    ldh [rLCDC], a
    xor a
    ldh [rIF], a
    ld a, 1
    ldh [rIE], a
    ei

MainLoop:
    call WaitFrame
    call ReadJoy
    ld b, a
    ld a, [wJoyPrev]
    cpl
    and b                       ; alleen nieuw ingedrukte knoppen
    ld c, a
    ld a, b
    ld [wJoyPrev], a
    ld a, c
    and PAD_LEFT | PAD_RIGHT
    jr z, .geenPijl
    ld a, [wCat]
    bit 4, c
    jr z, .links
    inc a
    jr .zet
.links:
    dec a
.zet:
    and 7
    ld [wCat], a
    call DrawFooter
    call WaitFrame
    call CopyStripsNow
.geenPijl:
    bit 0, c
    call nz, Generate
    jr MainLoop

; Eén plaatje maken: ruis, dan 16 stappen, elke stap in beeld.
Generate:
    ld a, [wFrames]
    ld e, a
    ldh a, [rDIV]
    ld d, a
    or e
    jr nz, .zaadOk
    inc e
.zaadOk:
    ld a, e
    ld [wSeed], a
    ld a, d
    ld [wSeed + 1], a
    call Noise
    xor a
    ld [wStep], a
    ld a, 1
    ld [wState], a
    call Show
    ld a, [wFrames]
    ld [wT0], a
    ld a, [wFrames + 1]
    ld [wT0 + 1], a
.stap:
    call Step
    ld a, [wStep]
    inc a
    ld [wStep], a
    cp 16
    jr nz, .tonen
    ld a, [wT0]                 ; klaar: verstreken frames vastleggen voor de laatste weergave
    ld e, a
    ld a, [wT0 + 1]
    ld d, a
    ld a, [wFrames]
    sub e
    ld [wElapsed], a
    ld a, [wFrames + 1]
    sbc d
    ld [wElapsed + 1], a
    ld a, 2
    ld [wState], a
.tonen:
    call Show
StepDone::
    nop
    ld a, [wStep]
    cp 16
    jr c, Generate.stap
    ld a, [wGens]
    inc a
    ld [wGens], a
GenDone::
    nop
    ret

; Toestand in beeld: canvas renderen, kop en voet tekenen, alles in VBlanks naar VRAM, dan omschakelen.
Show:
    call RenderCanvas
    call DrawHeader
    call DrawFooter
    ld a, [wShown]
    xor 1
    ld b, a                     ; onzichtbare bank
    ld c, 0                     ; deel 0, 1, 2 van 768 bytes
.deel:
    push bc
    call WaitFrame
    pop bc
    ld a, b
    ldh [rVBK], a
    ld a, 7
    ldh [rSVBK], a
    ld a, c                     ; bron en doel: + c * 768 = c * 3 pagina's
    add a
    add c
    ld d, a
    add HIGH(CANVAS_BUF)
    ldh [rHDMA1], a
    xor a
    ldh [rHDMA2], a
    ld a, d
    add $80
    ldh [rHDMA3], a
    xor a
    ldh [rHDMA4], a
    ld a, 768 / 16 - 1
    ldh [rHDMA5], a
    inc c
    ld a, c
    cp 3
    jr nz, .deel
    push bc
    call WaitFrame
    pop bc
    ld a, b                     ; omschakelen: bank 1 hoort bij map $9C00
    ld [wShown], a
    ldh a, [rLCDC]
    res 3, a
    bit 0, b
    jr z, .map
    set 3, a
.map:
    ldh [rLCDC], a
    call CopyStripsNow
    ret

; kop- en voetregel met GDMA naar VRAM (in VBlank aanroepen)
CopyStripsNow:
    ld a, 7
    ldh [rSVBK], a
    xor a
    ldh [rVBK], a
    ld a, HIGH(wHdr)
    ldh [rHDMA1], a
    ld a, LOW(wHdr)
    ldh [rHDMA2], a
    ld a, HIGH(HDR_TILES)
    ldh [rHDMA3], a
    ld a, LOW(HDR_TILES)
    ldh [rHDMA4], a
    ld a, 640 / 16 - 1          ; kop en voet liggen aaneen, in WRAM en in VRAM
    ldh [rHDMA5], a
    ret

; tint per stip, dan 96 pixelrijen van 12 tegels naar CANVAS_BUF (WRAM-bank 7); RowTab zegt per pixelrij welke
; stiprij erin valt (of een lege rij), ColTab per tegelkolom welke stipkolommen
RenderCanvas:
    ld a, 7
    ldh [rSVBK], a
    ld hl, wX
    ld bc, wTone
    ld d, HIGH(ToneTab)
.tint:
    ld e, [hl]
    ld a, [de]
    ld [bc], a
    inc bc
    inc hl
    ld a, l
    cp LOW(wX + DPIX)
    jr nz, .tint
    ld a, h
    cp HIGH(wX + DPIX)
    jr nz, .tint
    xor a
    ldh [hPY], a
.rij:
    ldh a, [hPY]                ; doel = buf + (py / 8) * 192 + (py % 8) * 2
    ld b, a
    and 7
    add a
    ld l, a
    ld h, 0
    ld a, b
    srl a
    srl a
    srl a
    ld e, a
    ld d, 0
    push hl
    ld h, d
    ld l, e
    add hl, hl                  ; *192 = *64 * 3
    add hl, de
    add hl, hl
    add hl, hl
    add hl, hl
    add hl, hl
    add hl, hl
    add hl, hl
    pop de
    add hl, de
    ld de, CANVAS_BUF
    add hl, de
    push hl
    ldh a, [hPY]                ; rij-offset in wTone, of $FFFF
    ld e, a
    ld d, 0
    ld hl, RowTab
    add hl, de
    add hl, de
    ld a, [hl+]
    ld h, [hl]
    ld l, a
    ld a, h
    inc a
    jr z, .leegrij
    ld de, wTone
    add hl, de
    ld a, l
    ldh [hRP], a
    ld a, h
    ldh [hRP + 1], a
    pop hl
    jr .stippen
.leegrij:
    pop hl
    ld b, 12                    ; tussenruimte: lege rij
    ld de, 15
.leeg:
    xor a
    ld [hl+], a
    ld [hl], a
    add hl, de
    dec b
    jr nz, .leeg
    jr .volgende
.stippen:
    ld a, l
    ldh [hDst], a
    ld a, h
    ldh [hDst + 1], a
    ld de, ColTab
    ld c, 12
.tegel:
    xor a
    ldh [hLo], a
    ldh [hHi], a
    ld a, [de]
    inc de
    ld b, a                     ; aantal stippen in deze tegelkolom
.stip:
    ld a, [de]                  ; stipkolom
    inc de
    ld l, a
    ldh a, [hRP]
    add l
    ld l, a
    ldh a, [hRP + 1]
    adc 0
    ld h, a
    ld h, [hl]                  ; tint
    ld a, [de]                  ; masker
    inc de
    ld l, a
    bit 0, h
    jr z, .geenLo
    ldh a, [hLo]
    or l
    ldh [hLo], a
.geenLo:
    bit 1, h
    jr z, .geenHi
    ldh a, [hHi]
    or l
    ldh [hHi], a
.geenHi:
    dec b
    jr nz, .stip
    ldh a, [hDst]
    ld l, a
    ldh a, [hDst + 1]
    ld h, a
    ldh a, [hLo]
    ld [hl+], a
    ldh a, [hHi]
    ld [hl], a
    ld a, l
    add 15
    ldh [hDst], a
    ld a, h
    adc 0
    ldh [hDst + 1], a
    dec c
    jr nz, .tegel
.volgende:
    ldh a, [hPY]
    inc a
    ldh [hPY], a
    cp 96
    jp nz, .rij
    ret

; ---------- tekst ----------
; DrawText: hl = tekst (0 = einde), hTX = x, hTC = kleur, hTBase = buffer. hTX staat daarna achter de tekst.
DrawText:
    ld a, [hl+]
    or a
    ret z
    push hl
    call DrawChar
    pop hl
    jr DrawText

; a = teken (ASCII 32-127)
DrawChar:
    sub 32
    ld l, a
    ld h, 0
    add hl, hl
    add hl, hl
    add hl, hl
    ld de, Font
    add hl, de
    ld a, [hl+]
    ldh [hGW], a
    xor a
    ldh [hRow], a
.rij:
    ld a, [hl+]
    or a
    jr z, .leeg
    push hl
    ld b, a                     ; glyphrij
    ldh a, [hTX]                ; tegel = x / 8, verschuiving = x % 8
    ld c, a
    srl a
    srl a
    srl a
    swap a                      ; de = (x / 8) * 16
    ld d, a
    and $F0
    ld e, a
    ld a, d
    and $0F
    ld d, a
    ldh a, [hRow]
    add a
    add e
    ld e, a
    ld a, d
    adc 0
    ld d, a
    ldh a, [hTBase]
    add e
    ld l, a
    ldh a, [hTBase + 1]
    adc d
    ld h, a                     ; hl = byte in de buffer
    ld a, c
    and 7
    ld c, a                     ; verschuiving
    ld d, b                     ; d = deel in deze tegel, e = deel in de volgende
    ld e, 0
    or a
    jr z, .geschoven
.schuif:
    srl d
    rr e
    dec c
    jr nz, .schuif
.geschoven:
    ld a, d
    push de
    call PutMask
    pop de
    ld a, e
    or a
    jr z, .klaar
    ld bc, 16
    add hl, bc
    ld a, e
    call PutMask
.klaar:
    pop hl
.leeg:
    ldh a, [hRow]
    inc a
    ldh [hRow], a
    cp 7
    jr nz, .rij
    ldh a, [hGW]                ; x += breedte + 1
    ld b, a
    ldh a, [hTX]
    add b
    inc a
    ldh [hTX], a
    ret

; masker a in tegelrij hl (lo, hi) zetten in kleur hTC
PutMask:
    ld b, a
    cpl
    ld c, a
    ldh a, [hTC]
    ld e, a
    ld a, [hl]
    and c
    bit 0, e
    jr z, .lo
    or b
.lo:
    ld [hl+], a
    ld a, [hl]
    and c
    bit 1, e
    jr z, .hi
    or b
.hi:
    ld [hl-], a
    ret

; breedte van tekst hl in pixels -> a
TextWidth:
    ld b, 0
.lus:
    ld a, [hl+]
    or a
    jr z, .einde
    push hl
    sub 32
    ld l, a
    ld h, 0
    add hl, hl
    add hl, hl
    add hl, hl
    ld de, Font
    add hl, de
    ld a, [hl]
    pop hl
    add b
    inc a
    ld b, a
    jr .lus
.einde:
    ld a, b
    or a
    ret z
    dec a
    ret

; Tekst rechts uitlijnen op x = 152: hl = tekst
DrawRight:
    push hl
    call TextWidth
    ld b, a
    ld a, 152
    sub b
    ldh [hTX], a
    pop hl
    jp DrawText

; buffer hl (320 bytes) leegmaken en als doel kiezen
StripBegin:
    ld a, l
    ldh [hTBase], a
    ld a, h
    ldh [hTBase + 1], a
    ld bc, 320
    xor a
    jp Memset

; kop: titel links, rechts de stap of de tijd
DrawHeader:
    ld hl, wHdr
    call StripBegin
    ld a, 8
    ldh [hTX], a
    ld a, 3
    ldh [hTC], a
    ld hl, TitleText
    call DrawText
    ld a, 2
    ldh [hTC], a
    ld a, [wState]
    or a
    ret z
    ld hl, wStr
    cp 1
    jr nz, .tijd
    ld a, [wStep]               ; "07/16"
    call Num2
    ld a, '/'
    ld [hl+], a
    ld a, '1'
    ld [hl+], a
    ld a, '6'
    ld [hl+], a
    jr .zet
.tijd:
    ld a, [wElapsed]            ; tienden = frames * 43 / 256 (59,73 frames per seconde)
    ld e, a
    ld a, [wElapsed + 1]
    ld d, a
    push hl
    ld hl, 0                    ; c:h:l = 24-bits product (boven 25,6 s past het niet in 16 bits)
    ld c, 0
    ld b, 43
.mul:
    add hl, de
    jr nc, .geenc
    inc c
.geenc:
    dec b
    jr nz, .mul
    ld e, h                     ; >> 8: tienden in de
    ld d, c
    ld b, 0                     ; seconden = tienden / 10
.deel:
    ld a, d
    or a
    jr nz, .min10
    ld a, e
    cp 10
    jr c, .rest
.min10:
    ld a, e
    sub 10
    ld e, a
    jr nc, .geenl
    dec d
.geenl:
    inc b
    jr .deel
.rest:
    ld c, a
    pop hl
    ld a, b
    call Num2
    ld a, '.'
    ld [hl+], a
    ld a, c
    add '0'
    ld [hl+], a
    ld a, ' '
    ld [hl+], a
    ld a, 'S'
    ld [hl+], a
.zet:
    xor a
    ld [hl], a
    ld hl, wStr
    ld a, [hl]                  ; voorloopnul weg
    cp '0'
    jr nz, .rechts
    inc hl
.rechts:
    jp DrawRight

; voet: kiezen/klaar = categorie met pijlen + A-knop, bezig = voortgangsbalk
DrawFooter:
    ld hl, wFtr
    call StripBegin
    ld a, [wState]
    cp 1
    jr z, .balk
    ld a, 8
    ldh [hTX], a
    ld a, 2
    ldh [hTC], a
    ld a, 123                   ; pijl links
    call DrawChar
    ldh a, [hTX]
    add 3
    ldh [hTX], a
    ld a, 3
    ldh [hTC], a
    ld a, [wCat]
    add a
    add a
    add a
    ld e, a
    ld d, 0
    ld hl, CatNames
    add hl, de
    call DrawText
    ldh a, [hTX]
    add 3
    ldh [hTX], a
    ld a, 2
    ldh [hTC], a
    ld a, 125                   ; pijl rechts
    call DrawChar
    ld hl, TxtDraw              ; rechts: "DRAW", dan de A-knop ervoor
    call TextWidth
    ld b, a
    ld a, 152
    sub b
    ld c, a
    sub 11
    ldh [hTX], a
    push bc
    ld a, '~'
    call DrawChar
    pop bc
    ld a, c
    sub 9
    ldh [hTX], a
    xor a
    ldh [hTC], a
    push bc
    ld a, '^'
    call DrawChar
    pop bc
    ld a, c
    ldh [hTX], a
    ld a, 3
    ldh [hTC], a
    ld hl, TxtDraw
    jp DrawText
.balk:
    ld c, 0                     ; 16 blokjes onder het canvas
.blok:
    ld a, c
    ld b, a
    add a
    add b
    add a                       ; *6
    add 32
    ldh [hTX], a
    ld a, [wStep]
    dec a
    cp c
    ld a, 2
    jr nc, .aan
    ld a, 1
.aan:
    ldh [hTC], a
    push bc
    ld a, '|'
    call DrawChar
    pop bc
    inc c
    ld a, c
    cp 16
    jr nz, .blok
    ret

; a (0-99) als twee cijfers op hl, hl schuift door
Num2:
    ld b, '0' - 1
.t:
    inc b
    sub 10
    jr nc, .t
    add 10 + '0'
    ld [hl], b
    inc hl
    ld [hl+], a
    ret

WaitFrame:
    ld a, [wFrames]
    ld b, a
.w:
    halt
    ld a, [wFrames]
    cp b
    jr z, .w
    ret

; a = knoppen (bit 7-4: omlaag, omhoog, links, rechts; 3-0: start, select, B, A)
ReadJoy:
    ld a, $20
    ldh [rP1], a
    ldh a, [rP1]
    ldh a, [rP1]
    cpl
    and $0F
    swap a
    ld b, a
    ld a, $10
    ldh [rP1], a
    REPT 6
    ldh a, [rP1]
    ENDR
    cpl
    and $0F
    or b
    ld b, a
    ld a, $30
    ldh [rP1], a
    ld a, b
    ret

VBlankHandler:
    push af
    push hl
    ld hl, wFrames
    inc [hl]
    jr nz, .einde
    inc hl
    inc [hl]
.einde:
    pop hl
    pop af
    reti

Memcpy:
    ld a, [hl+]
    ld [de], a
    inc de
    dec bc
    ld a, b
    or c
    jr nz, Memcpy
    ret

Memset:
    ld d, a
.l:
    ld a, d
    ld [hl+], a
    dec bc
    ld a, b
    or c
    jr nz, .l
    ret

TitleText:  db "GAME BOY DIFFUSION", 0
TxtDraw:    db "DRAW", 0
CatNames:
    db "CAT", 0, 0, 0, 0, 0
    db "HOUSE", 0, 0, 0
    db "TREE", 0, 0, 0, 0
    db "FISH", 0, 0, 0, 0
    db "CAR", 0, 0, 0, 0, 0
    db "APPLE", 0, 0, 0
    db "SUN", 0, 0, 0, 0, 0
    db "SMILEY", 0, 0

; Begin-ruis: xorshift16 (7, 9, 8), x_j = GaussTab[r & 255]
Noise:
    ld a, [wSeed]
    ld e, a
    ld a, [wSeed + 1]
    ld d, a
    ld hl, wX
.lus:
    ld a, d                     ; r ^= r << 7
    rrca
    and $80
    ld b, a
    ld a, e
    srl a
    or b
    xor d
    ld d, a
    ld a, e
    rrca
    and $80
    xor e
    ld e, a
    ld a, d                     ; r ^= r >> 9
    srl a
    xor e
    ld e, a
    xor d                       ; r ^= r << 8
    ld d, a
    push hl
    ld h, HIGH(GaussTab)
    ld l, e
    ld a, [hl]
    pop hl
    ld [hl+], a
    ld a, l
    cp LOW(wX + DPIX)
    jr nz, .lus
    ld a, h
    cp HIGH(wX + DPIX)
    jr nz, .lus
    ret

; Een denoise-stap (stapnummer in wStep).
Step:
    ld d, HIGH(VTab)            ; laag 1: v = VTab[x]
    ld hl, wX
    ld bc, wV
.v:
    ld e, [hl]
    ld a, [de]
    ld [bc], a
    inc bc
    inc hl
    ld a, l
    cp LOW(wX + DPIX)
    jr nz, .v
    ld a, h
    cp HIGH(wX + DPIX)
    jr nz, .v

    ld a, [wCat]                ; bias laag 1: index c*16 + stap, 2H bytes (opgevuld), 1 << BIAS_SHIFT per bank
    swap a
    ld b, a
    ld a, [wStep]
    or b
    ld b, a
    REPT BIAS_SHIFT
    srl a
    ENDR
    add BIAS1_BANK
    ld [rROMB0], a
    ld a, b
    and (1 << BIAS_SHIFT) - 1
    REPT BIAS_PAGES_LOG2
    add a
    ENDR
    add $40
    ld h, a
    ld l, 0
    ld bc, 2 * NH
    call CopyBias
    ld hl, Layer1
    call Layer
    ld de, wH1
    ld a, K1
    ld b, NB16
    call Act

    ld hl, B2
    ld bc, 2 * NH
    call CopyBias
    ld hl, Layer2
    call Layer
    ld de, wH2
    ld a, K2
    ld b, NB16
    call Act

    ld hl, B3
    ld bc, 2 * DPIX
    call CopyBias
    ld hl, Layer3
    call Layer
    call Out

    ld a, [wStep]               ; DDIM: x = clamp((A*x + B*x0) >> 7)
    ld e, a
    ld d, 0
    ld hl, StepA
    add hl, de
    ld a, [hl]
    ldh [hA], a
    ld hl, StepB
    add hl, de
    ld a, [hl]
    ldh [hB], a
    ld a, LOW(wX)
    ldh [hXp], a
    ld a, HIGH(wX)
    ldh [hXp + 1], a
    ld a, LOW(wX0)
    ldh [hX0p], a
    ld a, HIGH(wX0)
    ldh [hX0p + 1], a
.ddim:
    ldh a, [hX0p]
    ld l, a
    ldh a, [hX0p + 1]
    ld h, a
    ld b, [hl]
    inc hl
    ld a, l
    ldh [hX0p], a
    ld a, h
    ldh [hX0p + 1], a
    ldh a, [hB]
    ld c, a
    ld a, b
    call MulSU
    push hl
    ldh a, [hXp]
    ld l, a
    ldh a, [hXp + 1]
    ld h, a
    ld b, [hl]
    ldh a, [hA]
    ld c, a
    ld a, b
    call MulSU
    pop de
    add hl, de                  ; y
    sla l                       ; y >> 7 = hoge byte van y << 1; het teken staat in carry
    rl h
    ld a, h
    jr c, .neg
    bit 7, a
    jr z, .ok
    ld a, 127
    jr .ok
.neg:
    bit 7, a
    jr nz, .ok
    ld a, $80
.ok:
    ld b, a
    ldh a, [hXp]
    ld l, a
    ldh a, [hXp + 1]
    ld h, a
    ld [hl], b
    inc hl
    ld a, l
    ldh [hXp], a
    ld a, h
    ldh [hXp + 1], a
    ld a, l
    cp LOW(wX + DPIX)
    jr nz, .ddim
    ld a, h
    cp HIGH(wX + DPIX)
    jr nz, .ddim
    ret

; hl = x * c, met x (a) als int8 en c zonder teken
MulSU:
    ld e, a
    add a
    sbc a
    ld d, a
    ld hl, 0
.lus:
    srl c
    jr nc, .geen
    add hl, de
.geen:
    sla e
    rl d
    ld a, c
    or a
    jr nz, .lus
    ret

; hl = bron, bc = lengte -> wAcc
CopyBias:
    ld de, wAcc
    jp Memcpy

; Parameters per laag, in de volgorde van hInLo..hStrHi
Layer1: dw wV, DPIX
        db W1_BANK, CPB_H, NB16
        dw NH
Layer2: dw wH1, NH
        db W2_BANK, CPB_H, NB16
        dw NH
Layer3: dw wH2, NH
        db W3_BANK, CPB_D, NB16D
        dw DPIX

; Een laag: wAcc += sum_i T[in_i][W[:, i]].  hl = parameters (LayerN).
Layer:
    ld c, LOW(hInLo)
    ld b, 9
.param:
    ld a, [hl+]
    ldh [c], a
    inc c
    dec b
    jr nz, .param
    xor a
    ldh [hColLo], a
    ld a, $40
    ldh [hColHi], a
    ldh a, [hCPB]
    ldh [hLeft], a
.kolom:
    ldh a, [hInLo]
    ld l, a
    ldh a, [hInHi]
    ld h, a
    ld a, [hl]
    or a
    jp z, .volgende             ; invoer 0: niets op te tellen
    add 24                      ; rij r = v + 24
    ld b, a
    and 7
    add a
    add $D0
    ld d, a                     ; pagina met de lage bytes van rij r
    ld a, b
    srl a
    srl a
    srl a
    inc a
    ldh [rSVBK], a
    ldh a, [hBank]
    ld [rROMB0], a
    ldh a, [hColLo]
    ld c, a
    ldh a, [hColHi]
    ld b, a
    ld hl, wAcc
    ldh a, [hBlocks]
    ldh [hM], a
.blok:                          ; 16 uitgangen; c loopt binnen de pagina, b gaat mee na een volle pagina
    REPT 16
    ld a, [bc]                  ; gewicht W[j][i]
    ld e, a
    ld a, [de]                  ; T[v][w], lage byte
    add a, [hl]
    ld [hl+], a
    inc d
    ld a, [de]                  ; hoge byte
    adc a, [hl]
    ld [hl+], a
    dec d
    inc c
    ENDR
    ld a, c
    or a
    jr nz, .zelfde
    inc b
.zelfde:
    ldh a, [hM]
    dec a
    ldh [hM], a
    jp nz, .blok
.volgende:
    ldh a, [hInLo]              ; volgende invoer
    inc a
    ldh [hInLo], a
    jr nz, .kol
    ldh a, [hInHi]
    inc a
    ldh [hInHi], a
.kol:
    ldh a, [hStrLo]             ; volgende kolom, of de volgende bank
    ld b, a
    ldh a, [hColLo]
    add b
    ldh [hColLo], a
    ldh a, [hStrHi]
    ld b, a
    ldh a, [hColHi]
    adc b
    ldh [hColHi], a
    ldh a, [hLeft]
    dec a
    ldh [hLeft], a
    jr nz, .tel
    ldh a, [hBank]
    inc a
    ldh [hBank], a
    xor a
    ldh [hColLo], a
    ld a, $40
    ldh [hColHi], a
    ldh a, [hCPB]
    ldh [hLeft], a
.tel:
    ldh a, [hCntLo]
    sub 1
    ldh [hCntLo], a
    ldh a, [hCntHi]
    sbc 0
    ldh [hCntHi], a
    ld b, a
    ldh a, [hCntLo]
    or b
    jp nz, .kolom
    ld a, 7                     ; bank 7 weer zichtbaar (toestand, kop, voet, canvas)
    ldh [rSVBK], a
    ret

; ReLU naar 4 bits: [de] = clamp(acc >> a, 0, 15) voor b blokken van 16 (de op een veelvoud van 16)
Act:
    ldh [hK], a
    ld a, b
    ldh [hM], a
    ld hl, wAcc
.lus:
    ld a, [hl+]
    ld b, a
    ld a, [hl+]
    ld c, a
    bit 7, c
    jr nz, .nul
    push hl
    ldh a, [hK]
    ld l, a
.schuif:
    srl c
    rr b
    dec l
    jr nz, .schuif
    pop hl
    ld a, c
    or a
    jr nz, .max
    ld a, b
    cp 16
    jr c, .zet
.max:
    ld a, 15
    jr .zet
.nul:
    xor a
.zet:
    ld [de], a
    inc de
    ld a, e
    and $0F
    jr nz, .lus
    ldh a, [hM]
    dec a
    ldh [hM], a
    jr nz, .lus
    ret

; Uitgang: x0 = TanhTab[clamp(acc >> K3, -128, 127)]
Out:
    ld hl, wAcc
    ld de, wX0
.lus:
    ld a, [hl+]
    ld b, a
    ld a, [hl+]
    ld c, a
    push hl
    ld l, K3
.schuif:
    sra c
    rr b
    dec l
    jr nz, .schuif
    ld a, c
    or a
    jr z, .hoog0
    inc a
    jr z, .hoogFF
    bit 7, c
    ld a, $80
    jr nz, .opzoeken
    ld a, 127
    jr .opzoeken
.hoog0:
    ld a, b
    bit 7, a
    jr z, .opzoeken
    ld a, 127
    jr .opzoeken
.hoogFF:
    ld a, b
    bit 7, a
    jr nz, .opzoeken
    ld a, $80
.opzoeken:
    ld h, HIGH(TanhTab)
    ld l, a
    ld a, [hl]
    pop hl
    ld [de], a
    inc de
    ld a, e
    cp LOW(wX0 + DPIX)
    jp nz, .lus
    ld a, d
    cp HIGH(wX0 + DPIX)
    jp nz, .lus
    ret

; Producttabel T[v][w] = v * w in WRAM-banken 1-6
BuildProducts:
    xor a
    ldh [hI], a
.rij:
    ldh a, [hI]
    ld b, a
    srl a
    srl a
    srl a
    inc a
    ldh [rSVBK], a
    ld a, b
    and 7
    add a
    add $D0
    ld h, a
    ld a, b
    sub 24
    ld b, a                     ; v
    add a
    sbc a
    ld c, a                     ; tekenbyte van v
    ld de, 0
    ld l, 0
.pos:                           ; w = 0..127: p = 0, v, 2v, ...
    ld [hl], e
    inc h
    ld [hl], d
    dec h
    ld a, e
    add b
    ld e, a
    ld a, d
    adc c
    ld d, a
    inc l
    bit 7, l
    jr z, .pos
    ld de, 0
    ld l, 255
.neg:                           ; w = -1..-128: p = -v, -2v, ...
    ld a, e
    sub b
    ld e, a
    ld a, d
    sbc c
    ld d, a
    ld [hl], e
    inc h
    ld [hl], d
    dec h
    dec l
    bit 7, l
    jr nz, .neg
    ldh a, [hI]
    inc a
    ldh [hI], a
    cp 48
    jr nz, .rij
    ret


INCLUDE "build/data.asm"
