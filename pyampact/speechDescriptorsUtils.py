"""
speechDescriptorsUtils
==============


.. autosummary::
    :toctree: generated/

    compute_correlation_dimension,
    compute_dfa,
    compute_emd_features,
    compute_mfcc,
    compute_ppe,
    compute_recurrence_period_and_rpde,
    compute_spectral_spread,
    compute_tqwt_features,
    compute_wt_features,
    feature_spectral_flux,
    compute_vot,
    compute_bark_band_energies,
    compute_dpi,
    compute_avqi        
        
"""

import parselmouth
import librosa
import json
import pywt
import pandas as pd
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy.signal import find_peaks

__all__ = [
    "compute_correlation_dimension",
    "compute_dfa",
    "compute_emd_features",
    "compute_mfcc",
    "compute_ppe",
    "compute_recurrence_period_and_rpde",
    "compute_spectral_spread",
    "compute_tqwt_features",
    "compute_wt_features",
    "feature_spectral_flux",
    "compute_vot",
    "compute_bark_band_energies",
    "compute_dpi",
    "compute_avqi"
]


def feature_spectral_flux(X, f_s):
    """
    Calculates spectral flux

    Parameters
    ----------
    X : ndarray
        spectrogram

    f_s : float
        sample rate of audio data    

    Returns
    -------
    vsf : float
        spectral_flux

    """

    isSpectrum = X.ndim == 1
    if isSpectrum:
        X = np.expand_dims(X, axis=1)

    # difference spectrum (set first diff to zero)
    X = np.c_[X[:, 0], X]

    afDeltaX = np.diff(X, 1, axis=1)

    # flux
    vsf = np.sqrt((afDeltaX**2).sum(axis=0)) / X.shape[0]

    return np.squeeze(vsf) if isSpectrum else vsf


def compute_ppe(pitch_values_hz):
    """
    Compute Pitch Period Entropy (PPE) from a 1D array of pitch values in Hz.

    Parameters
    ----------
    pitch_values_hz : ndarray
        array of pitch values


    Returns
    -------
    entropy : float
        Pitch Period Entropy (PPE)

    """
    
    voiced = pitch_values_hz[pitch_values_hz > 0]
    if len(voiced) < 2:
        return np.nan
    
    # Convert to semitones relative to median
    f0_median = np.median(voiced)
    semitones = 12 * np.log2(voiced / f0_median)
    
    # Kernel density estimate over semitone deviations
    hist, edges = np.histogram(semitones, bins=50, density=True)
    probs = hist / np.sum(hist)
    probs = probs[probs > 0]
    
    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def compute_vot(seg, pitch, sr):
    """
    Compute Voice Onset Time (VOT)

    Parameters
    ----------
    seg : Praat Sound Object
        segment/slice of Snd object between onset and offset
    
    pitch : Pitch (CC) object
        pitch object from Praat
    
    sr : float
        sample rate of audio


    Returns
    -------
    entropy : float
        voice onset time (in ms)

    """

    
    y = seg.values[0]
    frame_length = int(0.02 * sr)  # 20 ms window
    hop = int(0.005 * sr)          # 5 ms hop
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop)
    
    # Release burst = first strong jump in energy
    release_idx = np.argmax(energy > np.mean(energy) + 2*np.std(energy))
    t_release = times[release_idx] if release_idx > 0 else None
        
    t_voicing = None
    if t_release:
        for t in times[release_idx:]:
            f0 = pitch.get_value_at_time(t)
            if f0 and f0 > 0:
                t_voicing = t
                break
    
    if t_release and t_voicing:
        return (t_voicing - t_release) * 1000.0  # ms
    else:
        return np.nan


# Bark band energies
def bark_scale_freqs(n_bands=24, fmax=16000):
    """
    Return Bark band edges up to fmax.

    Parameters
    ----------
    n_bands : int
        number of bark bands
    
    fmax : float
        maximum frequency of analysis


    Returns
    -------
    freqs : array
        array of bark band frequencies

    """
    
    bark_edges = np.linspace(0, 24, n_bands+1)
    freqs = 1960 * ( (bark_edges + 0.53) / (26.28 - bark_edges) )
    freqs = np.clip(freqs, 0, fmax)
    return freqs

def compute_bark_band_energies(S, sr, n_bands=24, mean_energies=True):
    """
    Compute bark band energies via bark_scale_freqs edges function

    Parameters
    ----------
    S : ndraay
        magnitude spectrum
    
    sr : float
        sample rate of audio file

    n_bands : int
        number of bark bands; default = 24
    
    mean_energies : bool
        default (True) returns mean of band energies, False returns array of band energies


    Returns
    -------
    band_energies : float
        mean of band energies as float, or array of band energies

    """
    # S = magnitude spectrogram (freq_bins x frames)
    freqs = np.linspace(0, sr/2, S.shape[0])
    edges = bark_scale_freqs(n_bands, sr/2)

    band_energies = []
    for b in range(n_bands):
        mask = (freqs >= edges[b]) & (freqs < edges[b+1])
        if not np.any(mask):
            band_energies.append(np.zeros(S.shape[1]))
            continue
        band_energy = S[mask, :].sum(axis=0)
        band_energies.append(band_energy)
    
    # Returns scalar of BBEs
    # Remove two lines to instead return array
    band_energies = np.stack(band_energies, axis=0)  # (bands × frames)

    if mean_energies == True:
        return float(np.mean(band_energies))
    else:
        return np.array(band_energies)  # shape (n_bands, frames)

    
    
def compute_correlation_dimension(y, m=3, tau=2, r_frac=0.1):
    """
    Compute Correlation Dimension (CD)
    
    Parameters
    ----------
    y : array
        time series values
    
    m : int
        embedding dimension for phase-space reconstruction; default = 3
    
    tau : int
        time delay (in samples); default = 2

    r_frac : float
        fraction of the distance standard deviation used to set the radius threshold; default = 0.1
            
        
        
    Returns
    ----------
    cd : float
        estimated correlation dimenstion, or np.nan if insufficient data
    """

    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size < 200:
        return np.nan

    y -= np.mean(y)

    N = len(y) - (m - 1) * tau
    if N < 50:
        return np.nan

    X = np.array([y[i:i + m*tau:tau] for i in range(N)])

    dists = np.linalg.norm(X[:, None] - X[None, :], axis=2)
    r = r_frac * np.std(dists)
    cd = np.mean(dists < r)

    return cd



# Duration of Pause Intervals (DPI) - NOT WORKING
def compute_dpi(snd, pitch, intensity, intensity_thresh, time_step=0.01,
                min_pause_dur=0.15):    
    """
    Compute Duration of Pause Intervals (DPI).

    Parameters
    ----------
    snd : Praat Sound Object
        Parselmouth/Praat Sound Object
    
    pitch : Praat Pitch Object
        Parselmouth/Praat Pitch Object
    
    intensity : Praat Intensity Object
        time delay (in samples)

    intensity_thresh : int
        dB SPL threshold for silence
    
    time_step : float
        analysis step in seconds; default = 0.01
    
    min_pause_dur : float
        minimum pause duration to count (sec); default = 0.15
                    
        
    Returns
    ----------
    pause_durations : array
        list of pause durations (s)
    
    mean_dpi: float
        average pause duration

    """
    # Segment bounds (absolute time)
    t0 = snd.xmin
    t1 = snd.xmax

    times = np.arange(t0, t1, time_step)

    silent = []

    for t in times:
        f0 = pitch.get_value_at_time(t)
        inten = intensity.get_value(t)

        is_silent = (
            (f0 is None or f0 <= 0) or
            (inten is None or inten < intensity_thresh)
        )
        silent.append(is_silent)

    pause_durations = []
    in_pause = False
    pause_start = None

    for t, s in zip(times, silent):
        if s and not in_pause:
            in_pause = True
            pause_start = t
        elif not s and in_pause:
            dur = t - pause_start
            if dur >= min_pause_dur:
                pause_durations.append(dur)
            in_pause = False

    # Handle trailing pause
    if in_pause and pause_start is not None:
        dur = t1 - pause_start
        if dur >= min_pause_dur:
            pause_durations.append(dur)

    mean_dpi = np.mean(pause_durations) if pause_durations else np.nan
    return pause_durations, mean_dpi

def compute_dfa(y, min_window=4, max_window=None, n_windows=20):
    """
    Compute DFA (Detrended Fluctuation Analysis) scaling component (alpha) for 1D signal y

    Parameters
    ----------
    y : array
        speech segment waveform
    
    min_window : int
        smallest window size (samples); default = 4
    
    min_window : max_window
        largest window size (samples); default = len(y)//4
    
    n_windows : int
        number of window sizes to test; default = 20

        
    Returns
    ----------
    alpha : float
        DFA scaling exponent

    """

    y = np.array(y, dtype=float)
    y -= np.mean(y)
    N = len(y)
    if N < min_window*4:
        return np.nan

    # Integrated signal
    y_int = np.cumsum(y)

    # Window sizes (log spaced)
    if max_window is None:
        max_window = N // 4
    window_sizes = np.unique(np.logspace(np.log10(min_window),
                                         np.log10(max_window),
                                         n_windows, dtype=int))
    flucts = []
    for w in window_sizes:
        if w < 2: 
            continue
        n_segments = N // w
        if n_segments < 2:
            continue
        shape = (n_segments, w)
        segments = np.reshape(y_int[:n_segments*w], shape)

        # detrend each segment
        t = np.arange(w)
        detrended = []
        for seg in segments:
            p = np.polyfit(t, seg, 1)
            trend = np.polyval(p, t)
            detrended.append(np.sqrt(np.mean((seg - trend)**2)))
        flucts.append(np.mean(detrended))

    if len(flucts) < 2:
        return np.nan
    flucts = np.array(flucts)
    slope, _ = np.polyfit(np.log(window_sizes[:len(flucts)]),
                          np.log(flucts), 1)
    return float(slope)


# Empirical Mode Decomposition (EMD)
def compute_emd_features(y, max_imfs=6, max_sift=10):
    """
    Empirical Mode Decomposition (EMD) features.
    Returns scalar complexity measures, not raw IMFs.

    Parameters
    ----------
    y : 1D numpy array
        input signal (e.g. gneVals or waveform)
    max_imfs : int
        maximum IMFs to extract; default = 6
    max_sift : int
        max sifting iterations per IMF; default = 10

    Returns
    -------
    emd_n_imfs : int
        number of imfs
    emd_energy_ratio : float
        high-frequency IMF energy / total energy
    emd_entropy : float
        Shannon entropy of IMF energy distribution
    """
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]

    if y.size < 100:
        return np.nan, np.nan, np.nan

    y = y - np.mean(y)
    residue = y.copy()
    imfs = []

    for _ in range(max_imfs):
        h = residue.copy()

        for _ in range(max_sift):
            peaks, _ = find_peaks(h)
            troughs, _ = find_peaks(-h)

            if len(peaks) < 2 or len(troughs) < 2:
                break

            # Envelope interpolation
            t = np.arange(len(h))
            upper = np.interp(t, peaks, h[peaks])
            lower = np.interp(t, troughs, h[troughs])
            mean_env = 0.5 * (upper + lower)

            h = h - mean_env

        imfs.append(h)
        residue = residue - h

        if np.std(residue) < 1e-6:
            break

    if len(imfs) == 0:
        return np.nan, np.nan, np.nan

    # Energy per IMF
    energies = np.array([np.sum(imf**2) for imf in imfs])
    total_energy = np.sum(energies)

    if total_energy <= 0:
        return np.nan, np.nan, np.nan

    # Metrics
    emd_n_imfs = len(imfs)

    # High-frequency = first half of IMFs
    hf_energy = np.sum(energies[: max(1, emd_n_imfs // 2)])
    emd_energy_ratio = hf_energy / total_energy

    probs = energies / total_energy
    emd_entropy = -np.sum(probs * np.log2(probs + 1e-12))

    return int(emd_n_imfs), float(emd_energy_ratio), float(emd_entropy)


# Mel Frequency Cepstral Coefficients (MFCCs) — STRING RETURN
def compute_mfcc(seg, sr, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Compute Mel Frequency Cepstral Coefficients (MFCCs)

    Parameters
    ----------
    seg : Praat Sound Object
        segment of Sound cut around onset and offset
    n_mfcc : int
        number of coefficients to return; default = 13
    n_fft : int
        FFT window size; default = 2048
    hop_length : int
        number of samples of analysis advances between consecutive frames; default = 512

    Returns
    -------
    mfcc : JSON Stringified list
        list of MFCCs
    
    """
    y = seg.values[0].astype(float)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        htk=False
    )
    
    return json.dumps(mfcc.astype(np.float32).tolist())



def compute_recurrence_period_and_rpde(pp, onset, offset, n_bins=50):
    """
    Compute recurrence period (RP) and recurrence period density entropy (RPDE)
    
    RP is the mean inter-pulse interval within a segment.
    RPDE is the normalized entropy of the distribution of recurrence periods.

    Low = highly periodic/stable signal
    High = irregular, noisy, chaotic signal

    Parameters
    ----------
    pp : Praat Pitch Object
        Periodic (cc) of the segment
    onset : float        
    offset : float
        segment boundaries in seconds
    n_bins : int
        bin count for histogram calculation; default = 50

    Returns
    -------
    rp_ms : float
        recurrence period in ms
    
    rpde : float
        normalized entropy of recurrence-period distribution
    
    """

    n = parselmouth.praat.call(pp, "Get number of points")

    times = []
    for k in range(1, n + 1):
        t = parselmouth.praat.call(pp, "Get time from index", k)
        if onset <= t <= offset:
            times.append(t)

    times = np.asarray(times)

    if times.size < 3:
        return np.nan, np.nan

    # Inter-pulse periods (seconds)
    periods = np.diff(times)
    periods = periods[(periods > 0) & np.isfinite(periods)]

    if periods.size < 3:
        return np.nan, np.nan

    # RP (mean period)
    rp_ms = float(np.mean(periods) * 1000.0)

    # RPDE (normalized entropy)
    hist, _ = np.histogram(periods, bins=n_bins, density=True)
    probs = hist / np.sum(hist)
    probs = probs[probs > 0]

    rpde = float(-np.sum(probs * np.log(probs)) / np.log(len(probs)))

    return rp_ms, rpde


def compute_spectral_spread(S, freqs):
    """
    Compute spectral spread

    Parameters
    ----------
    S : ndarray
        magnitude spectrum
    
    freqs : array
        frequency axis in Hz (freq_bins,)

    Returns
    -------
    rp_ms : float
        recurrence period in ms
    
    rpde : float
        normalized entropy of recurrence-period distribution
    
    """
    # S: magnitude spectrogram (freq_bins × frames)
    # freqs: frequency axis in Hz (freq_bins,)

    S_sum = np.sum(S, axis=0)
    valid = S_sum > 0

    if not np.any(valid):
        return np.nan

    # Spectral centroid per frame
    centroid = np.sum(freqs[:, None] * S, axis=0) / S_sum

    # Spectral spread per frame
    spread = np.sqrt(
        np.sum(((freqs[:, None] - centroid) ** 2) * S, axis=0) / S_sum
    )

    return float(np.mean(spread[valid]))



# Tuneable Q-Wavelet Transform (TQWT)

def _tqwt_theta(w):
    # Eq. (17) in Selesnick 2011: theta(ω) = 0.5(1+cos ω)*sqrt(2 - cos ω), |ω|<=π
    w = np.asarray(w, dtype=float)
    return 0.5 * (1.0 + np.cos(w)) * np.sqrt(np.maximum(2.0 - np.cos(w), 0.0))

def _tqwt_analysis_filters(N, alpha, beta):
    # Eqs. (11)-(16) in Selesnick 2011; even-symmetric magnitude response on DFT grid
    k = np.arange(N)
    w = 2.0 * np.pi * k / N
    w = (w + np.pi) % (2.0 * np.pi) - np.pi  # [-π, π)
    aw = np.abs(w)

    H0 = np.zeros(N, dtype=float)
    H1 = np.zeros(N, dtype=float)

    wP = (1.0 - beta) * np.pi
    wS = alpha * np.pi
    denom = (alpha + beta - 1.0)

    # Pass/stop
    P = aw <= wP
    S = aw >= wS
    H0[P] = 1.0
    H1[S] = 1.0

    # Transition: (1-β)π < |ω| < απ
    T = (~P) & (~S)
    wt = aw[T]
    u0 = (wt + (beta - 1.0) * np.pi) / denom
    u1 = (alpha * np.pi - wt) / denom
    H0[T] = _tqwt_theta(u0)
    H1[T] = _tqwt_theta(u1)

    return H0.astype(np.complex128), H1.astype(np.complex128)

def _tqwt_lps(x, alpha):
    # Low-pass scaling (α<1): keep center αN bins (FFT-domain truncation)
    x = np.asarray(x, dtype=float)
    N = x.size
    M = int(np.ceil(alpha * N))
    if M < 2:
        return np.zeros(2, dtype=float)

    Xs = fftshift(fft(x))
    start = (N - M) // 2
    Ys = Xs[start:start + M]
    y = np.real(ifft(ifftshift(Ys))) * np.sqrt(alpha)
    return y

def _tqwt_hps(x, beta):
    # High-pass scaling (β<1): keep outer βN bins (FFT-domain) and pack into length βN
    x = np.asarray(x, dtype=float)
    N = x.size
    M = int(np.ceil(beta * N))
    if M < 2:
        return np.zeros(2, dtype=float)

    Xs = fftshift(fft(x))
    left_len = M // 2
    right_len = M - left_len
    left = Xs[:left_len]
    right = Xs[-right_len:] if right_len > 0 else np.array([], dtype=Xs.dtype)
    Ys = np.concatenate([left, right])
    y = np.real(ifft(ifftshift(Ys))) * np.sqrt(beta)
    return y

def tqwt_forward(x, Q=1.0, r=3.0, J=8):
    """
    Forward TQWT (DFT-domain, finite-length).
    Returns: coeffs (list of J arrays), residual (array), (alpha, beta)

    Uses Eq. (26): beta = 2/(Q+1), alpha = 1 - beta/r.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 16:
        return [np.array([], dtype=float) for _ in range(int(J))], np.array([], dtype=float), (np.nan, np.nan)

    Q = float(Q)
    r = float(r)
    J = int(J)

    beta = 2.0 / (Q + 1.0)
    alpha = 1.0 - (beta / r)

    v = x
    coeffs = []
    for _ in range(J):
        N = v.size
        if N < 16:
            coeffs.append(np.array([], dtype=float))
            continue

        H0, H1 = _tqwt_analysis_filters(N, alpha, beta)
        V = fft(v)
        v0 = np.real(ifft(V * H0))
        v1 = np.real(ifft(V * H1))

        w = _tqwt_hps(v1, beta)
        v = _tqwt_lps(v0, alpha)

        coeffs.append(w)

    return coeffs, v, (alpha, beta)

def compute_tqwt_features(y, sr, Q=1.0, r=3.0, J=8, n_keep=5):
    """
    Compute a fixed-width feature vector from the Tunable Q-factor Wavelet Transform (TQWT).
    Features are derived from subband energies of the TQWT decomposition of a 1-D signal.

    Parameters
    ----------
    y : 1D numpy array
        input signal
    sr : float
        sample rate of the signal in Hz. (Currently unused; included for API consistency.)
    Q : float
        Q-factor controlling oscillatory behavior of the wavelets; default = 1.0
    r : float
        redundancy factor of the TQWT; default = 3.0
    J : int
        number of TQWT decomposition levels; default = 8
    n_keep : int
        number of lowest-index subband log-energies to return; default = 5

    Returns
    -------
    features : list of float
        Feature vector containing, in order:

        - tqwt_e1 .. tqwt_e{n_keep} : float
            Log10 subband energies of the first `n_keep` TQWT bands.
        - tqwt_entropy : float
            Normalized Shannon entropy of the subband energy distribution.
        - tqwt_centroid : float
            Energy-weighted centroid of subband indices.
        - tqwt_low_high_ratio : float
            Ratio of summed low-band energy to high-band energy.

        If insufficient valid data are available, all features are returned as NaN.
    """
    
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 16:
        feats = [np.nan] * n_keep + [np.nan, np.nan, np.nan]
        return feats

    coeffs, resid, _ = tqwt_forward(y, Q=Q, r=r, J=J)

    eps = 1e-12
    energies = np.array([float(np.mean(c*c)) if c.size else 0.0 for c in coeffs], dtype=float)
    energies = np.maximum(energies, 0.0)

    # log-energies (first n_keep bands)
    loge = np.log10(energies + eps)
    e_keep = [float(loge[i]) if i < loge.size else np.nan for i in range(n_keep)]

    # entropy over subband energy distribution
    tot = float(np.sum(energies))
    if tot <= eps:
        tqwt_entropy = np.nan
        tqwt_centroid = np.nan
        tqwt_low_high_ratio = np.nan
    else:
        p = energies / tot
        p = p[p > 0]
        tqwt_entropy = float(-np.sum(p * np.log(p)) / np.log(max(len(p), 2)))

        idx = np.arange(1, energies.size + 1, dtype=float)
        tqwt_centroid = float(np.sum(idx * energies) / (tot + eps))  # band-index centroid

        low = float(np.sum(energies[: max(1, energies.size // 2)]))
        high = float(np.sum(energies[max(1, energies.size // 2):]))
        tqwt_low_high_ratio = float((low + eps) / (high + eps))

    return e_keep + [tqwt_entropy, tqwt_centroid, tqwt_low_high_ratio]




def compute_wt_features(y, wavelet="db4", level=5):
    """
    Compute wavelet-based features from a 1-D signal using discrete wavelet transform (DWT).

    Features are derived from the normalized energy distribution of wavelet subbands.

    Parameters
    ----------
    y : 1D numpy array
        input signal
    wavelet : str
        Wavelet family used for decomposition; default = "db4"
    level : int
        Decomposition level; default = 5

    Returns
    -------
    wt_entropy : float
        Shannon entropy (base 2) of the normalized wavelet subband energies.
    wt_low_high_ratio : float
        Ratio of high-frequency subband energy to low-frequency subband energy.
    """

    coeffs = pywt.wavedec(y, wavelet, level=level)
    
    energies = np.array([np.sum(c**2) for c in coeffs])
    energies /= np.sum(energies)

    wt_entropy = -np.sum(energies * np.log2(energies + 1e-12))
    wt_low_high_ratio = np.sum(energies[level//2:]) / np.sum(energies[:level//2])

    return wt_entropy, wt_low_high_ratio




def compute_avqi(shimmer_local, hnr_mean_db,cpps_vowel, slope_vowel,cpps_speech, slope_speech):
    """Compute the Acoustic Voice Quality Index (AVQI)."""
    avqi = (
        3.41
        + (0.33 * shimmer_local)
        - (0.09 * hnr_mean_db)
        - (0.88 * cpps_vowel)
        + (0.12 * slope_vowel)
        - (1.18 * cpps_speech)
        + (0.46 * slope_speech)
    )
    return avqi

# This will give the proper 0-10 AVQI score?
def avqi_v0206(cpps_vowel,
               hnr_speech,
               shimmer_local_percent,
               shimmer_local_db,
               ltas_slope_speech,
               ltas_tilt_speech):
    """
    Compute the REAL Acoustic Voice Quality Index (AVQI) version 02.06
    as defined by Maryn, Weenink, Roy, et al.

    Parameters
    ----------
    cpps_vowel : float
        Smoothed CPP (dB) from sustained vowel /a/ segment.
    hnr_speech : float
        Harmonics-to-noise ratio (dB) from connected speech.
    shimmer_local_percent : float
        Shimmer local (percent) from sustained vowel.
    shimmer_local_db : float
        Shimmer local (dB) from sustained vowel.
    ltas_slope_speech : float
        LTAS spectral slope (dB/decade) from connected speech.
    ltas_tilt_speech : float
        LTAS spectral tilt (dB) from connected speech.

    Returns
    -------
    float
        AVQI (real clinical AVQI v02.06)
        Range roughly: 0–10 (normal voices ~1–3, dysphonic ≥4).
    """

    # Raw regression (unscaled)
    raw = (
        3.295
        - 0.111 * cpps_vowel
        - 0.073 * hnr_speech
        - 0.213 * shimmer_local_percent
        + 2.789 * shimmer_local_db
        - 0.032 * ltas_slope_speech
        + 0.077 * ltas_tilt_speech
    )

    # Final scaling and offset (critical!)
    avqi = raw * 2.208 + 1.797
    return avqi

    

    


# # Johanna's code...
# def vocal_presense(y, sr, nfft=1024, hop_len=256):
#     # Features
#     rmse = librosa.feature.rms(y=y, frame_length=nfft, hop_length=hop_len, center=False)[0]
#     zcr  = librosa.feature.zero_crossing_rate(y=y, frame_length=nfft, hop_length=hop_len, center=False)[0]

#     # Clip-adaptive thresholds (percentile-based) + sanity clamps
#     # Energy (RMS)
#     ste_on  = max(np.percentile(rmse, 65), 0.02)   # don't go below 0.02 on normalized audio
#     ste_off = max(np.percentile(rmse, 45), 0.015)

#     # ZCR (lower is more voiced) – invert logic with hysteresis
#     zcr_on  = np.clip(np.percentile(zcr, 35), 0.08, 0.18)
#     zcr_off = np.clip(np.percentile(zcr, 65), 0.18, 0.30)

#     # Hysteresis state machine
#     is_vocal = np.zeros_like(rmse, dtype=bool)
#     state = False
#     for i, (e, z) in enumerate(zip(rmse, zcr)):
#         if not state:
#             state = (e >= ste_on) and (z <= zcr_on)
#         else:
#             state = not ((e <= ste_off) or (z >= zcr_off))
#         is_vocal[i] = state

#     # Debounce flicker (median filter ~30–50 ms)
#     k = int(round(0.04 * sr / hop_len)) | 1  # odd kernel size, ~40 ms
#     is_vocal = sps.medfilt(is_vocal.astype(float), kernel_size=max(k, 3)) > 0.5
#     return is_vocal