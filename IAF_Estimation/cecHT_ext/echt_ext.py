"""Local ECHT extensions layered on the pristine ``cecHT`` submodule.

``ECHTExt`` adds two things to upstream :class:`phase.ECHT` without touching the
submodule:

1. ``filter_type="custom"`` -- use caller-supplied ``b`` / ``a`` transfer-function
   coefficients for the band-pass instead of designing a Butterworth/Bessel/etc.
   The calibration gain is fitted against whatever :meth:`_design_bandpass`
   returns (see below), so ``custom`` and subclass overrides calibrate correctly.

2. A sliding-DFT forward transform (:meth:`transform_sdft`) that replaces the
   one-shot ``fft`` with a per-sample recursive DFT (``jurihock/sdft``), so the
   two paths differ *only* in how the forward spectrum is produced.

Upstream ``_calibration`` is a ``@staticmethod`` that calls
``ECHT._design_bandpass`` by name, so a plain subclass override of
``_design_bandpass`` would be ignored during calibration. ``ECHTExt`` therefore
also overrides ``_calibration`` as an instance method that routes through
``self._design_bandpass``. This is the only upstream method body duplicated here.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))
import _bootstrap  # noqa: E402,F401  -> puts the cecHT submodule on sys.path

import numpy as np  # noqa: E402
from scipy.signal import freqz  # noqa: E402
from scipy.fft import ifft, fftshift, ifftshift, next_fast_len  # noqa: E402

from phase import ECHT  # noqa: E402


class ECHTExt(ECHT):
    """:class:`phase.ECHT` + custom band-pass coefficients + sliding-DFT transform."""

    def __init__(self, *args, b=None, a=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.filter_type == "custom" and (b is None or a is None):
            raise ValueError("filter_type='custom' requires both b and a to be provided.")
        self.b = None if b is None else np.asarray(b, dtype=float)
        self.a = None if a is None else np.asarray(a, dtype=float)

    # ------------------------------------------------------------------
    # Band-pass design (instance method: honours self / custom coefficients)
    # ------------------------------------------------------------------
    def _design_bandpass(self, l_freq, h_freq, sfreq, filt_order, n_fft,
                         filter_type="butter", b=None, a=None):
        if filter_type != "custom":
            # Delegate to the pristine upstream implementation (SOS-based,
            # lru_cache'd). It does not take b/a.
            return ECHT._design_bandpass(l_freq, h_freq, sfreq, filt_order,
                                         n_fft, filter_type)

        b = self.b if b is None else np.asarray(b, dtype=float)
        a = self.a if a is None else np.asarray(a, dtype=float)
        if b is None or a is None:
            raise ValueError("filter_type='custom' requires both b and a to be provided.")

        # Hilbert analytic multiplier (same construction as upstream).
        h = np.zeros(n_fft, dtype=float)
        h[0] = 1
        if n_fft % 2 == 0:
            h[1:n_fft // 2] = 2
            h[n_fft // 2] = 1
        else:
            h[1:(n_fft + 1) // 2] = 2

        filt_freq = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1 / sfreq))
        _, H_center = freqz(b, a, worN=filt_freq, fs=sfreq)
        return h, H_center

    # ------------------------------------------------------------------
    # Calibration (instance method so it uses self._design_bandpass)
    # ------------------------------------------------------------------
    def _calibration(self, f0, sfreq, N, l_freq, h_freq, filt_order=1,
                     filter_type="butter", L=None, fft_mode="fast", b=None, a=None):
        """Theoretical endpoint error for a pure cosine input.

        Body copied verbatim from upstream :meth:`phase.ECHT._calibration`; the
        only change is ``ECHT._design_bandpass(...)`` -> ``self._design_bandpass(...)``
        so that ``custom`` coefficients and subclass overrides are respected.
        """
        if L is None:
            L = next_fast_len(N) if fft_mode == "fast" else N

        def _dirichlet_N(alpha):
            alpha = np.asarray(alpha, dtype=float)
            D = np.empty(alpha.shape, dtype=np.complex128)
            small = np.abs(alpha) < 1e-12
            D[small] = N
            a_ns = alpha[~small]
            D[~small] = (
                np.exp(1j * a_ns * (N - 1) / 2)
                * np.sin(0.5 * N * a_ns)
                / np.sin(0.5 * a_ns)
            )
            return D

        k = np.arange(L)
        omega_k = 2 * np.pi * k / L
        omega0 = 2 * np.pi * f0 / sfreq
        n = N - 1

        h, H_center = self._design_bandpass(
            l_freq=l_freq, h_freq=h_freq, sfreq=sfreq, filt_order=filt_order,
            n_fft=L, filter_type=filter_type, b=b, a=a,
        )

        H_eff = ifftshift(H_center)
        G = h * H_eff

        D_plus = _dirichlet_N(omega0 - omega_k)
        D_minus = _dirichlet_N(-omega0 - omega_k)

        X_plus = 0.5 * D_plus
        X_minus = 0.5 * D_minus

        phase = np.exp(1j * omega_k * n)
        P = (G * X_plus * phase).sum() / L
        M = (G * X_minus * phase).sum() / L

        Gplus = P * np.exp(-1j * omega0 * n)
        Gminus = M * np.exp(-1j * omega0 * n)

        denom = np.abs(Gplus) ** 2 + np.abs(Gminus) ** 2
        if denom == 0:
            C_opt = 1 + 0j
            J_opt = 0
        else:
            C_opt = np.conj(Gplus) / denom
            J_opt = np.abs(Gminus) ** 2 / denom

        return {"Gplus": Gplus, "Gminus": Gminus, "C_opt": C_opt, "J_opt": J_opt}

    # ------------------------------------------------------------------
    # fit(): pass custom coefficients through to design + calibration
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        X = np.asarray(X)
        n_samples = X.shape[0]

        if self.n_fft is None:
            self.n_fft = (
                next_fast_len(n_samples) if self.fft_mode == "fast" else n_samples
            )

        self.h_, H_center = self._design_bandpass(
            l_freq=self.l_freq, h_freq=self.h_freq, sfreq=self.sfreq,
            filt_order=self.filt_order, n_fft=self.n_fft,
            filter_type=self.filter_type, b=self.b, a=self.a,
        )
        self.coef_ = H_center[:, None]

        self.calib_gain_ = None
        self.calib_err_ = None
        if self.calibrate:
            if self.f0 is None:
                raise ValueError("f0 (signal frequency) must be provided when calibrate=True.")
            err = self._calibration(
                f0=self.f0, sfreq=self.sfreq, N=n_samples,
                l_freq=self.l_freq, h_freq=self.h_freq, filt_order=self.filt_order,
                L=self.n_fft, fft_mode=self.fft_mode, filter_type=self.filter_type,
                b=self.b, a=self.a,
            )
            C_opt = err["C_opt"]
            if not (np.isfinite(C_opt.real) and np.isfinite(C_opt.imag)):
                raise RuntimeError("Non-finite calibration gain C_opt computed for ECHT.")
            self.calib_gain_ = C_opt
            self.calib_err_ = err

        return self

    # ------------------------------------------------------------------
    # Sliding-DFT forward transform
    # ------------------------------------------------------------------
    def _forward_sdft(self, X):
        """Forward spectrum equivalent to ``fft(X, self.n_fft, axis=0)`` computed
        with a per-sample sliding DFT (jurihock/sdft, boxcar window, latency=1).

        Only bins ``0..n_fft//2`` (DC through Nyquist) are populated; everything
        above Nyquist is zeroed by the Hilbert multiplier anyway.
        """
        try:
            from sdft import SDFT
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transform_sdft requires the 'sdft' package (pip install sdft)."
            ) from exc

        n_fft = self.n_fft
        if n_fft % 2 != 0:
            raise ValueError(
                "transform_sdft requires an even n_fft: jurihock/sdft's sliding "
                "window is always length 2*dftsize (even) internally, so an odd "
                "n_fft cannot be represented exactly. Use fft_mode='exact' with "
                "an even-length signal, or pass an explicit even n_fft."
            )
        n_channels = X.shape[1]
        m = n_fft // 2  # bins 0..m-1 are DC..just-below-Nyquist

        Xf = np.zeros((n_fft, n_channels), dtype=complex)

        for ch in range(n_channels):
            x_padded = np.zeros(n_fft, dtype=float)
            n_copy = min(n_fft, X.shape[0])
            x_padded[:n_copy] = X[:n_copy, ch]

            sdft = SDFT(m, window="boxcar", latency=1)
            dfts = sdft.sdft(x_padded)   # (n_fft, m): one row per streamed sample
            bins = dfts[-1]              # spectrum after the full window

            # sdft normalizes by n_fft; undo it to match np.fft.fft's convention.
            Xf[:m, ch] = bins * n_fft

            # Nyquist bin (index m) is outside the sliding DFT's range; it is just
            # an alternating sum, so compute it directly.
            signs = np.where(np.arange(n_fft) % 2 == 0, 1.0, -1.0)
            Xf[m, ch] = np.sum(x_padded * signs)

        return Xf

    def transform_sdft(self, X, **kwargs):
        """Like :meth:`phase.ECHT.transform` but with the sliding-DFT forward step.

        Any difference between ``transform`` and ``transform_sdft`` output is
        attributable to the forward-transform algorithm alone (filtering and
        calibration are identical).
        """
        X = np.asarray(X)
        if not np.isrealobj(X):
            X = np.real(X)

        if self.h_ is None or self.coef_ is None:
            self.fit(X)

        if X.ndim == 1:
            X = X[:, np.newaxis]

        n_samples = X.shape[0]

        Xf = self._forward_sdft(X)
        Xf = Xf * self.h_[:, None]
        Xf = fftshift(Xf, axes=0)
        Xf = Xf * self.coef_
        Xf = ifftshift(Xf, axes=0)
        Xf = ifft(Xf, axis=0)
        Xf = self._apply_calibration(Xf, **kwargs)
        return Xf[:n_samples, :]

    def fit_transform_sdft(self, X, y=None):
        return self.fit(X).transform_sdft(X)
