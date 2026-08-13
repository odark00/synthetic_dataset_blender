"""Pure-python ECC200 Data Matrix encoder (ASCII encodation, square symbols).

No dependencies - importable from Blender's bundled Python and from plain
python3 alike. Verified module-for-module against BWIPP/ghostscript output,
and the Reed-Solomon stage reproduces the ISO/IEC 16022 worked example
("123456" -> ECC 114, 25, 5, 88, 102).

    size, grid = encode("PKG000123")     # grid[row][col], 1 = dark module

Payloads must fit a single data region (<= 44 ASCII codewords, 26x26).
"""

# (symbol size incl. finder, data region size, data codewords, ecc codewords)
DMC_SIZES = [
    (10, 8, 3, 5), (12, 10, 5, 7), (14, 12, 8, 10), (16, 14, 12, 12),
    (18, 16, 18, 14), (20, 18, 22, 18), (22, 20, 30, 20), (24, 22, 36, 24),
    (26, 24, 44, 28),
]

_LOG = [0] * 256
_ALOG = [0] * 256


def _init_gf():
    p = 1
    for i in range(255):
        _ALOG[i] = p
        _LOG[p] = i
        p <<= 1
        if p >= 256:
            p ^= 0x12D
    _ALOG[255] = _ALOG[0]


_init_gf()


def _rs_ecc(data, nc):
    """Reed-Solomon check words, GF(256) poly 0x12D (iec16022 formulation)."""
    # generator polynomial (nc+1 coefficients; the monic top one is unused)
    rs = [0] * (nc + 1)
    rs[0] = 1
    for i in range(1, nc + 1):
        rs[i] = 1
        for j in range(i - 1, 0, -1):
            if rs[j]:
                rs[j] = _ALOG[(_LOG[rs[j]] + i) % 255]
            rs[j] ^= rs[j - 1]
        rs[0] = _ALOG[(_LOG[rs[0]] + i) % 255]

    # LFSR division; the register is nc wide, feedback tap at res[nc-1]
    res = [0] * nc
    for value in data:
        m = res[nc - 1] ^ value
        for j in range(nc - 1, 0, -1):
            if m and rs[j]:
                res[j] = res[j - 1] ^ _ALOG[(_LOG[m] + _LOG[rs[j]]) % 255]
            else:
                res[j] = res[j - 1]
        res[0] = _ALOG[(_LOG[m] + _LOG[rs[0]]) % 255] if (m and rs[0]) else 0
    return [res[nc - 1 - i] for i in range(nc)]


def _encode_ascii(text):
    cw = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isdigit() and i + 1 < n and text[i + 1].isdigit():
            cw.append(int(text[i:i + 2]) + 130)
            i += 2
            continue
        o = ord(c)
        if o < 128:
            cw.append(o + 1)
            i += 1
        else:
            cw.append(235)
            cw.append(o - 128 + 1)
            i += 1
    return cw


def _pad(cw, nd):
    if len(cw) < nd:
        cw.append(129)                       # end of message
    while len(cw) < nd:
        pos = len(cw) + 1                    # 1-based position
        r = ((149 * pos) % 253) + 1
        v = 129 + r
        cw.append(v if v <= 254 else v - 254)
    return cw


def _placement(nrow, ncol):
    """ECC200 module placement: returns nrow*ncol of bit-index or None."""
    arr = [None] * (nrow * ncol)

    def mod(row, col, ch, bit):
        if row < 0:
            row += nrow
            col += 4 - ((nrow + 4) % 8)
        if col < 0:
            col += ncol
            row += 4 - ((ncol + 4) % 8)
        arr[row * ncol + col] = (ch - 1) * 8 + (bit - 1)

    def utah(row, col, ch):
        mod(row - 2, col - 2, ch, 1)
        mod(row - 2, col - 1, ch, 2)
        mod(row - 1, col - 2, ch, 3)
        mod(row - 1, col - 1, ch, 4)
        mod(row - 1, col, ch, 5)
        mod(row, col - 2, ch, 6)
        mod(row, col - 1, ch, 7)
        mod(row, col, ch, 8)

    def corner1(ch):
        mod(nrow - 1, 0, ch, 1); mod(nrow - 1, 1, ch, 2); mod(nrow - 1, 2, ch, 3)
        mod(0, ncol - 2, ch, 4); mod(0, ncol - 1, ch, 5)
        mod(1, ncol - 1, ch, 6); mod(2, ncol - 1, ch, 7); mod(3, ncol - 1, ch, 8)

    def corner2(ch):
        mod(nrow - 3, 0, ch, 1); mod(nrow - 2, 0, ch, 2); mod(nrow - 1, 0, ch, 3)
        mod(0, ncol - 4, ch, 4); mod(0, ncol - 3, ch, 5); mod(0, ncol - 2, ch, 6)
        mod(0, ncol - 1, ch, 7); mod(1, ncol - 1, ch, 8)

    def corner3(ch):
        mod(nrow - 3, 0, ch, 1); mod(nrow - 2, 0, ch, 2); mod(nrow - 1, 0, ch, 3)
        mod(0, ncol - 2, ch, 4); mod(0, ncol - 1, ch, 5)
        mod(1, ncol - 1, ch, 6); mod(2, ncol - 1, ch, 7); mod(3, ncol - 1, ch, 8)

    def corner4(ch):
        mod(nrow - 1, 0, ch, 1); mod(nrow - 1, ncol - 1, ch, 2)
        mod(0, ncol - 3, ch, 3); mod(0, ncol - 2, ch, 4); mod(0, ncol - 1, ch, 5)
        mod(1, ncol - 3, ch, 6); mod(1, ncol - 2, ch, 7); mod(1, ncol - 1, ch, 8)

    ch = 1
    row, col = 4, 0
    while True:
        if row == nrow and col == 0:
            corner1(ch); ch += 1
        if row == nrow - 2 and col == 0 and (ncol % 4):
            corner2(ch); ch += 1
        if row == nrow - 2 and col == 0 and (ncol % 8) == 4:
            corner3(ch); ch += 1
        if row == nrow + 4 and col == 2 and not (ncol % 8):
            corner4(ch); ch += 1
        while True:
            if row < nrow and col >= 0 and arr[row * ncol + col] is None:
                utah(row, col, ch); ch += 1
            row -= 2
            col += 2
            if not (row >= 0 and col < ncol):
                break
        row += 1
        col += 3
        while True:
            if row >= 0 and col < ncol and arr[row * ncol + col] is None:
                utah(row, col, ch); ch += 1
            row += 2
            col -= 2
            if not (row < nrow and col >= 0):
                break
        row += 3
        col += 1
        if not (row < nrow or col < ncol):
            break

    if arr[nrow * ncol - 1] is None:          # fixed corner pattern
        arr[nrow * ncol - 1] = -2
        arr[nrow * ncol - ncol - 2] = -2
    return arr


def encode(text, min_size=0):
    """Return (size, rows) where rows is a list of lists of 0/1, 1 = dark."""
    cw = _encode_ascii(text)
    spec = None
    for size, region, nd, nc in DMC_SIZES:
        if len(cw) <= nd and size >= min_size:
            spec = (size, region, nd, nc)
            break
    if spec is None:
        raise ValueError(f"payload too long for a single-region symbol: {text!r}")
    size, region, nd, nc = spec

    cw = _pad(cw, nd)
    cw = cw + _rs_ecc(cw, nc)

    arr = _placement(region, region)
    grid = [[0] * size for _ in range(size)]

    # finder pattern: solid left column + solid bottom row, alternating
    # clock track along the top row and right column
    for i in range(size):
        grid[size - 1][i] = 1                       # bottom solid
        grid[i][0] = 1                              # left solid
        if i % 2 == 0:
            grid[0][i] = 1                          # top clock
        if i % 2 == 1:
            grid[i][size - 1] = 1                   # right clock

    for r in range(region):
        for c in range(region):
            v = arr[r * region + c]
            if v is None:
                bit = 0
            elif v == -2:
                bit = 1
            else:
                bit = (cw[v // 8] >> (7 - (v % 8))) & 1
            grid[1 + r][1 + c] = bit
    return size, grid


if __name__ == "__main__":
    import sys
    size, grid = encode(sys.argv[1] if len(sys.argv) > 1 else "ABC123")
    print(f"{size}x{size}")
    for row in grid:
        print("".join("##" if v else "  " for v in row))
