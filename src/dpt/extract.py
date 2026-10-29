"""Switching-waveform post-processing for a double-pulse test (DPT).

All quantities are extracted from *terminal* waveforms of the low-side device
under test (DUT) and the high-side freewheeling device, i.e. what a real DPT
measures with probes on the package leads. Definitions (see DESIGN.md, section
"Integration windows") follow IEC 60747-8/-9 style gate- and threshold-referenced
windows; every threshold is a parameter so the window sensitivity can be shown.

Sign conventions
----------------
* ``vds``  : DUT drain-to-power-source voltage (V)
* ``id``   : current into the DUT drain terminal (A), positive when conducting
* ``vgs``  : DUT gate to Kelvin-source voltage (V)
* ``vds_hs``/``i_hs``: freewheeling device drain-source voltage and drain current;
  ``i_hs`` is negative while the body diode conducts the load current and
  becomes positive during reverse recovery.
* ``vgs_hs``: freewheeling device gate to Kelvin-source voltage (V)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class ExtractionError(RuntimeError):
    """Raised when a waveform feature (edge, crossing) cannot be located."""


@dataclass(frozen=True)
class Thresholds:
    """Fractions used to build the integration and slope windows."""

    vgs_on_start: float = 0.10  # Eon starts when vgs passes 10 % of its swing (rising)
    vgs_off_start: float = 0.90  # Eoff starts when vgs falls through 90 % of its swing
    vds_on_end: float = 0.02  # Eon ends when vds has fallen to 2 % of Vbus
    id_off_end: float = 0.02  # Eoff ends when id has fallen to 2 % of I_load
    slope_low: float = 0.20  # dv/dt, di/dt evaluated between 20 % ...
    slope_high: float = 0.80  # ... and 80 % of the final value
    rr_end: float = 0.02  # Err window ends when reverse current < 2 % of Irrm


@dataclass
class Waveforms:
    """Terminal waveforms of one double-pulse run (uniform or non-uniform time)."""

    t: FloatArray
    vds: FloatArray
    id: FloatArray
    vgs: FloatArray
    vds_hs: FloatArray
    i_hs: FloatArray
    vgs_hs: FloatArray

    def window(self, t0: float, t1: float) -> Waveforms:
        """Return the samples with ``t0 <= t <= t1``."""
        m = (self.t >= t0) & (self.t <= t1)
        return Waveforms(*(np.asarray(getattr(self, f))[m] for f in _FIELDS))


_FIELDS = ("t", "vds", "id", "vgs", "vds_hs", "i_hs", "vgs_hs")


# --------------------------------------------------------------------------- helpers
def crossing_time(
    t: FloatArray,
    y: FloatArray,
    level: float,
    direction: str,
    t_start: float | None = None,
    t_stop: float | None = None,
) -> float:
    """First time in ``[t_start, t_stop]`` where ``y`` crosses ``level``.

    ``direction`` is ``"rising"`` or ``"falling"``. The crossing instant is
    linearly interpolated between the two bracketing samples.
    """
    if direction not in ("rising", "falling"):
        raise ValueError("direction must be 'rising' or 'falling'")
    lo = 0 if t_start is None else int(np.searchsorted(t, t_start, side="left"))
    hi = len(t) if t_stop is None else int(np.searchsorted(t, t_stop, side="right"))
    ys = y[lo:hi] - level
    if direction == "rising":
        idx = np.nonzero((ys[:-1] < 0) & (ys[1:] >= 0))[0]
    else:
        idx = np.nonzero((ys[:-1] > 0) & (ys[1:] <= 0))[0]
    if idx.size == 0:
        raise ExtractionError(f"no {direction} crossing of {level:.4g} in [{t_start}, {t_stop}]")
    k = lo + int(idx[0])
    y0, y1 = y[k] - level, y[k + 1] - level
    frac = 0.0 if y1 == y0 else -y0 / (y1 - y0)
    return float(t[k] + frac * (t[k + 1] - t[k]))


def integrate(t: FloatArray, y: FloatArray, t0: float, t1: float) -> float:
    """Trapezoidal integral of ``y`` over ``[t0, t1]`` with interpolated end points."""
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")
    inner = (t > t0) & (t < t1)
    ts = np.concatenate(([t0], t[inner], [t1]))
    ys = np.concatenate(([np.interp(t0, t, y)], y[inner], [np.interp(t1, t, y)]))
    return float(np.trapezoid(ys, ts))


def ringing(t: FloatArray, y: FloatArray, t0: float, t1: float) -> tuple[float, float]:
    """Dominant ringing frequency (Hz) and damping ratio of ``y`` on ``[t0, t1]``.

    Fits ``c + A*exp(-s*t)*sin(2*pi*f*t + phi)`` by least squares on a uniform
    resampling of the window, seeded with the FFT peak. Fitting the offset ``c``
    avoids the bias that mean removal introduces for a decaying oscillation.
    Returns ``(nan, nan)`` when the fit fails or the window holds less than two
    periods.
    """
    from scipy.optimize import curve_fit  # local import keeps module import light

    n = 4096
    tu = np.linspace(t0, t1, n)
    yu = np.interp(tu, t, y)
    x = tu - t0
    dt = x[1] - x[0]
    spec = np.abs(np.fft.rfft((yu - yu.mean()) * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, dt)
    spec[0] = 0.0
    f_guess = float(freqs[int(np.argmax(spec))])
    if f_guess * (t1 - t0) < 2.0:
        return float("nan"), float("nan")

    def model(xx: FloatArray, c: float, a: float, s: float, f: float, ph: float) -> FloatArray:
        return c + a * np.exp(-s * xx) * np.sin(2 * np.pi * f * xx + ph)

    amp = 0.5 * float(np.ptp(yu[: n // 4]))
    p0 = [float(yu[-n // 8 :].mean()), amp, 1.0 / (t1 - t0), f_guess, 0.0]
    best: tuple[float, float] | None = None
    best_cost = np.inf
    for ph0 in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
        p0[4] = ph0
        try:
            popt, _ = curve_fit(model, x, yu, p0=p0, maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        cost = float(np.sum((model(x, *popt) - yu) ** 2))
        if cost < best_cost and popt[3] > 0:
            best_cost = cost
            s_fit, f_fit = float(abs(popt[2])), float(popt[3])
            wd = 2 * np.pi * f_fit
            best = (f_fit, s_fit / np.hypot(s_fit, wd))
    if best is None:
        return float("nan"), float("nan")
    return best


def max_slope(t: FloatArray, y: FloatArray, t0: float, t1: float, window: float = 1e-9) -> float:
    """Largest |dy/dt| on ``[t0, t1]``, each slope averaged over ``window`` seconds.

    Averaging over a short window (1 ns, about the rise time a 350 MHz probe resolves)
    keeps single-step numerical spikes out of the result.
    """
    n = max(int((t1 - t0) / (window / 8)), 16)
    tu = np.linspace(t0, t1, n)
    yu = np.interp(tu, t, y)
    k = max(round(float(window / (tu[1] - tu[0]))), 1)
    if k >= n:
        return float("nan")
    return float(np.max(np.abs(yu[k:] - yu[:-k])) / (tu[k] - tu[0]))


def _slope(
    t: FloatArray,
    y: FloatArray,
    y_lo: float,
    y_hi: float,
    direction: str,
    t_start: float,
    t_stop: float,
) -> float:
    """Average slope between the ``y_lo`` and ``y_hi`` crossings (units of y per s)."""
    if direction == "rising":
        ta = crossing_time(t, y, y_lo, "rising", t_start, t_stop)
        tb = crossing_time(t, y, y_hi, "rising", ta, t_stop)
        return (y_hi - y_lo) / (tb - ta)
    ta = crossing_time(t, y, y_hi, "falling", t_start, t_stop)
    tb = crossing_time(t, y, y_lo, "falling", ta, t_stop)
    return (y_lo - y_hi) / (tb - ta)


# --------------------------------------------------------------------------- results
@dataclass
class TurnOffResult:
    e_off: float  # J
    t_start: float
    t_end: float
    dv_dt: float  # V/s, 20-80 %
    dv_dt_max: float  # V/s, peak slope (1 ns window)
    di_dt: float  # A/s (negative), 80-20 %
    vds_peak: float  # V
    overshoot: float  # V above Vbus
    f_ring: float  # Hz
    zeta_ring: float
    vgs_hs_min: float  # V, most negative freewheeling gate excursion
    vgs_max: float  # V, DUT gate extremes in the window (rating / oscillation check)
    vgs_min: float  # V
    td_off: float  # s, 90 % vgs -> 90 % id
    tf: float  # s, 90 % -> 10 % id
    td_off_v: float  # s, 90 % vgs -> 10 % vds (voltage-based, JEDEC-style)
    tf_v: float  # s, 10 % -> 90 % vds


@dataclass
class TurnOnResult:
    e_on: float  # J
    t_start: float
    t_end: float
    dv_dt: float  # V/s (negative), 80-20 %
    dv_dt_max: float  # V/s, peak |slope| (1 ns window), positive
    di_dt: float  # A/s, 20-80 %
    id_peak: float  # A
    vgs_hs_max: float  # V, highest freewheeling gate excursion (false turn-on risk)
    vds_hs_peak: float  # V, freewheeling-device voltage peak (diode-recovery overshoot)
    vgs_max: float  # V, DUT gate extremes in the window (rating / oscillation check)
    vgs_min: float  # V
    e_rr: float  # J, freewheeling-diode reverse-recovery energy
    q_rr: float  # C
    i_rrm: float  # A
    td_on: float  # s, 10 % vgs -> 10 % id
    tr: float  # s, 10 % -> 90 % id
    td_on_v: float  # s, 10 % vgs -> 90 % vds (voltage-based, JEDEC-style)
    tr_v: float  # s, 90 % -> 10 % vds


def extract_turn_off(
    w: Waveforms,
    t_cmd: float,
    vbus: float,
    i_load: float,
    vgs_on: float,
    vgs_off: float,
    th: Thresholds | None = None,
    t_span: float = 400e-9,
) -> TurnOffResult:
    """Extract turn-off metrics around the gate command edge ``t_cmd``.

    Window: from ``vgs`` falling through ``vgs_off + 0.9*(vgs_on - vgs_off)`` to
    ``id`` falling through ``2 % * i_load`` (first crossing after the vds rise).
    """
    th = th or Thresholds()
    t, t_stop = w.t, t_cmd + t_span
    swing = vgs_on - vgs_off
    t0 = crossing_time(t, w.vgs, vgs_off + th.vgs_off_start * swing, "falling", t_cmd, t_stop)
    t_i90 = crossing_time(t, w.id, 0.9 * i_load, "falling", t0, t_stop)
    t_i10 = crossing_time(t, w.id, 0.1 * i_load, "falling", t_i90, t_stop)
    t1 = crossing_time(t, w.id, th.id_off_end * i_load, "falling", t_i90, t_stop)
    t_v10 = crossing_time(t, w.vds, 0.1 * vbus, "rising", t0, t_stop)
    t_v90 = crossing_time(t, w.vds, 0.9 * vbus, "rising", t_v10, t_stop)
    p = w.vds * w.id
    e_off = integrate(t, p, t0, t1)

    dv_dt = _slope(t, w.vds, th.slope_low * vbus, th.slope_high * vbus, "rising", t0, t_stop)
    di_dt = _slope(t, w.id, th.slope_low * i_load, th.slope_high * i_load, "falling", t0, t_stop)

    seg = w.window(t0, t_stop)
    k_pk = int(np.argmax(seg.vds))
    vds_peak = float(seg.vds[k_pk])
    t_pk = float(seg.t[k_pk])
    f_ring, zeta = ringing(t, w.vds, t_pk, min(t_pk + 0.6 * t_span, t_stop))
    return TurnOffResult(
        e_off=e_off,
        t_start=t0,
        t_end=t1,
        dv_dt=dv_dt,
        dv_dt_max=max_slope(t, w.vds, t0, t1),
        di_dt=di_dt,
        vds_peak=vds_peak,
        overshoot=vds_peak - vbus,
        f_ring=f_ring,
        zeta_ring=zeta,
        vgs_hs_min=float(np.min(seg.vgs_hs)),
        vgs_max=float(np.max(seg.vgs)),
        vgs_min=float(np.min(seg.vgs)),
        td_off=t_i90 - t0,
        tf=t_i10 - t_i90,
        td_off_v=t_v10 - t0,
        tf_v=t_v90 - t_v10,
    )


def extract_turn_on(
    w: Waveforms,
    t_cmd: float,
    vbus: float,
    i_load: float,
    vgs_on: float,
    vgs_off: float,
    th: Thresholds | None = None,
    t_span: float = 400e-9,
) -> TurnOnResult:
    """Extract turn-on and freewheeling-diode recovery metrics after ``t_cmd``.

    Eon window: ``vgs`` rising through 10 % of its swing to ``vds`` falling to
    2 % of ``vbus``. Err window: freewheeling current crossing zero towards
    reverse to the reverse current decaying below 2 % of its peak.
    """
    th = th or Thresholds()
    t, t_stop = w.t, t_cmd + t_span
    swing = vgs_on - vgs_off
    t0 = crossing_time(t, w.vgs, vgs_off + th.vgs_on_start * swing, "rising", t_cmd, t_stop)
    t_i10 = crossing_time(t, w.id, 0.1 * i_load, "rising", t0, t_stop)
    t_i90 = crossing_time(t, w.id, 0.9 * i_load, "rising", t_i10, t_stop)
    t1 = crossing_time(t, w.vds, th.vds_on_end * vbus, "falling", t0, t_stop)
    t_v90 = crossing_time(t, w.vds, 0.9 * vbus, "falling", t0, t_stop)
    t_v10 = crossing_time(t, w.vds, 0.1 * vbus, "falling", t_v90, t_stop)
    e_on = integrate(t, w.vds * w.id, t0, t1)

    di_dt = _slope(t, w.id, th.slope_low * i_load, th.slope_high * i_load, "rising", t0, t_stop)
    dv_dt = _slope(t, w.vds, th.slope_low * vbus, th.slope_high * vbus, "falling", t0, t_stop)

    seg = w.window(t0, t_stop)
    # Freewheeling device: reverse recovery shows as i_hs going positive.
    t_rr0 = crossing_time(t, w.i_hs, 0.0, "rising", t0, t_stop)
    seg_rr = w.window(t_rr0, t_stop)
    k_rrm = int(np.argmax(seg_rr.i_hs))
    i_rrm = float(seg_rr.i_hs[k_rrm])
    t_rrm = float(seg_rr.t[k_rrm])
    try:
        t_rr1 = crossing_time(t, w.i_hs, th.rr_end * i_rrm, "falling", t_rrm, t_stop)
    except ExtractionError:
        t_rr1 = t_stop
    e_rr = integrate(t, w.vds_hs * w.i_hs, t_rr0, t_rr1)
    q_rr = integrate(t, w.i_hs, t_rr0, t_rr1)
    return TurnOnResult(
        e_on=e_on,
        t_start=t0,
        t_end=t1,
        dv_dt=dv_dt,
        dv_dt_max=max_slope(t, w.vds, t0, t1),
        di_dt=di_dt,
        id_peak=float(np.max(seg.id)),
        vgs_hs_max=float(np.max(seg.vgs_hs)),
        vds_hs_peak=float(np.max(seg.vds_hs)),
        vgs_max=float(np.max(seg.vgs)),
        vgs_min=float(np.min(seg.vgs)),
        e_rr=e_rr,
        q_rr=q_rr,
        i_rrm=i_rrm,
        td_on=t_i10 - t0,
        tr=t_i90 - t_i10,
        td_on_v=t_v90 - t0,
        tr_v=t_v10 - t_v90,
    )


def as_dict(result: TurnOffResult | TurnOnResult) -> dict[str, float]:
    return {k: float(v) for k, v in asdict(result).items()}
