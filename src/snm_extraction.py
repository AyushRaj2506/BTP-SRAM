"""
snm_extraction.py  --  butterfly-curve SNM (largest inscribed square), hold and read.

Pure-numpy core: takes the two voltage-transfer curves and returns the SNM.
No LTspice / file-path dependencies, so it can be unit-tested on its own.

Curve conventions (match the two-cell butterfly testbench, SRAM_6T_butterfly_SNM_*.asc):
  curve A : cell 1, Q forced 0->VDD,   x = V(Q)  , y = V(Qb)      ->  y = f(x)
  curve B : cell 2, Qb2 forced 0->VDD, x = V(Q2) , y = V(Qb2)     ->  x = g(y)
(In the .asc these are V(bfx),V(bfy) for .step mode=0 (A) and mode=1 (B).)

SNM = min(lobe1, lobe2), each lobe = side of the largest axis-aligned square that fits
between the curves.  Works for asymmetric (faulty) cells too, because f and g are used
separately.  A lobe that does not exist (cell not bistable) gives 0.

NOTE: the 45-degree "rotate and take max separation" shortcut is NOT used on purpose:
with rail-to-rail VTCs (|slope| > 1) the rotated curve is not single valued and the
shortcut over-estimates SNM badly.
"""
import numpy as np


def _as_function(x, y):
    """Return (xs, ys) sorted by x with duplicates removed, for np.interp."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    o = np.argsort(x, kind="stable")
    x, y = x[o], y[o]
    keep = np.concatenate(([True], np.diff(x) > 0))
    return x[keep], y[keep]


def lobe_sides(ax, ay, bx, by, vdd, n=4001):
    """
    ax,ay : curve A samples (x=Q,  y=Qb)   -> y=f(x)
    bx,by : curve B samples (x=Q2, y=Qb2)  -> x=g(y)
    Returns (lobe1, lobe2) side lengths in volts.
      lobe1 = upper-left  lobe (Q low / Qb high state)
      lobe2 = lower-right lobe (Q high / Qb low state)
    """
    fx, fy = _as_function(ax, ay)            # y = f(x)
    gy, gx = _as_function(by, bx)            # x = g(y)
    f = lambda x: np.interp(x, fx, fy)
    g = lambda y: np.interp(y, gy, gx)

    t = np.linspace(0.0, vdd, n)

    def best(start_fn, h_fn):
        # start_fn(t) -> (x0,y0) ; h_fn(x0,y0,s) decreasing in s, square fits while >=0
        x0, y0 = start_fn(t)
        h0 = h_fn(x0, y0, np.zeros_like(t))
        lo = np.zeros_like(t); hi = np.full_like(t, vdd)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            ok = h_fn(x0, y0, mid) >= 0
            lo = np.where(ok, mid, lo); hi = np.where(ok, hi, mid)
        lo = np.where(h0 > 0, lo, 0.0)
        return float(lo.max())

    # lobe 1 (upper-left): left edge on B (x0 = g(y0)), top/right corner under A
    l1 = best(lambda t: (g(t), t),
              lambda x0, y0, s: f(x0 + s) - (y0 + s))
    # lobe 2 (lower-right): bottom edge on A (y0 = f(x0)), right edge left of B
    l2 = best(lambda t: (t, f(t)),
              lambda x0, y0, s: g(y0 + s) - (x0 + s))
    return l1, l2


def snm_from_curves(ax, ay, bx, by, vdd):
    """SNM (volts) = min of the two lobes."""
    l1, l2 = lobe_sides(ax, ay, bx, by, vdd)
    return min(l1, l2)


def snm_from_raw(raw_path, vdd=1.0, ax_name="bfx", ay_name="bfy"):
    """Read a stepped LTspice .raw (via PyLTSpice) from the butterfly testbench, return dict."""
    try:
        from PyLTSpice import RawRead
    except ImportError:
        from spicelib.raw.raw_read import RawRead
    r = RawRead(str(raw_path))
    ax = r.get_trace(f"V({ax_name})").get_wave(0); ay = r.get_trace(f"V({ay_name})").get_wave(0)
    bx = r.get_trace(f"V({ax_name})").get_wave(1); by = r.get_trace(f"V({ay_name})").get_wave(1)
    l1, l2 = lobe_sides(ax, ay, bx, by, vdd)
    return {"snm_v": min(l1, l2), "lobe1_v": l1, "lobe2_v": l2}
