"""
performance
==============


.. autosummary::
    :toctree: generated/

    estimate_perceptual_parameters
    calculate_vibrato
    perceived_pitch
"""

import numpy as np
import librosa
import warnings

__all__ = [
    "estimate_perceptual_parameters",
    "calculate_vibrato",
    "perceived_pitch"
]

np.seterr(all='ignore')
warnings.filterwarnings(
    "ignore", message="Mean of empty slice.", category=RuntimeWarning)


def estimate_perceptual_parameters(f0_vals, pwr_vals, M, SR, hop, gt_flag):
    """
    Estimates a range of performance parameters from the inputted fundamental
    frequency (f0_vales), power (pwr_vals), and spectrum estimates (M)

    Parameters
    ----------
    f0_vals : np.ndarray
        A vector of fundamental frequency estimates.

    pwr_vals : np.ndarray
        A vector of power values corresponding to the fundamental frequency estimates.

    M : np.ndarray or dict
        Either:
          - dict of precomputed spectral feature vectors (FAST PATH), or
          - spectrogram slice [freq x frames] (SLOW / LEGACY PATH)

    SR : float
        The sample rate used to generate the fundamental frequency estimates.

    hop : float
        The hop size used in the analysis, expressed in seconds.

    gt_flag : bool
        A flag indicating whether to use ground truth information for the estimation.

    Returns
    -------
    res : dict
        Dictionary of per-note vectors and summary statistics.
    """

    # Perceived pitch
    res_ppitch = perceived_pitch(f0_vals, SR)

    # Work with clean (non-NaN) f0 values for voiced-frame descriptors
    f0_clean = f0_vals[~np.isnan(f0_vals)]

    # Jitter (mean absolute frame-to-frame f0 change, voiced frames only)
    if len(f0_clean) > 1:
        tmp_jitter = np.abs(np.diff(f0_clean))
        res_jitter = float(np.nanmean(tmp_jitter))
    else:
        res_jitter = np.nan

    # Vibrato rate and depth (voiced frames only)
    if len(f0_clean) > 2:
        mean_f0_clean = np.mean(f0_clean)
        detrended_f0 = f0_clean - mean_f0_clean
        frame_rate = SR / hop
        res_vibrato_depth, res_vibrato_rate = calculate_vibrato(detrended_f0, frame_rate)
    else:
        res_vibrato_depth, res_vibrato_rate = np.nan, np.nan

    # Shimmer – work with positive power values
    pwr_safe = np.where(pwr_vals > 0, pwr_vals, np.nan)
    res_pwr_vals = 10 * np.log10(np.where(pwr_vals > 0, pwr_vals, 1e-20))
    if np.sum(~np.isnan(pwr_safe)) > 1:
        pwr0 = np.nanmean(pwr_safe)  # use mean as reference to avoid division-by-first-frame issue
        tmp_shimmer = np.abs(10 * np.log10(pwr_safe[1:] / pwr0))
        res_shimmer = float(np.nanmean(tmp_shimmer))
    else:
        res_shimmer = np.nan

    res_f0_vals = f0_vals

    if isinstance(M, dict):
        # FAST PATH: precomputed spectral feature slices passed in from dataCompilation
        _sc  = np.asarray(M["spec_centroid"]).flatten()
        _sb  = np.asarray(M["spec_bandwidth"]).flatten()
        _sco = np.asarray(M["spec_contrast"])          # (bands, frames) or (frames,)
        _sf  = np.asarray(M["spec_flatness"]).flatten()
        _sr  = np.asarray(M["spec_rolloff"]).flatten()

        res_spec_centroid       = _sc.reshape(1, -1)
        res_mean_spec_centroid  = float(np.nanmean(_sc))

        res_spec_bandwidth      = _sb.reshape(1, -1)
        res_mean_spec_bandwidth = float(np.nanmean(_sb))

        res_spec_contrast       = _sco
        res_mean_spec_contrast  = float(np.nanmean(_sco))

        res_spec_flatness       = _sf.reshape(1, -1)
        res_mean_spec_flatness  = float(np.nanmean(_sf))

        res_spec_rolloff        = _sr.reshape(1, -1)
        res_mean_spec_rolloff   = float(np.nanmean(_sr))

    else:
        # LEGACY PATH: raw spectrogram slice [freq x frames]
        if gt_flag:
            M = np.abs(M) ** 2

        # Spectral bandwidth
        res_spec_bandwidth      = librosa.feature.spectral_bandwidth(S=M)
        res_mean_spec_bandwidth = np.mean(res_spec_bandwidth)

        # Spectral centroid
        res_spec_centroid       = librosa.feature.spectral_centroid(S=M)
        res_mean_spec_centroid  = np.mean(res_spec_centroid)

        # Spectral contrast
        res_spec_contrast       = librosa.feature.spectral_contrast(S=M)
        res_mean_spec_contrast  = np.mean(res_spec_contrast)

        # Spectral flatness
        res_spec_flatness       = librosa.feature.spectral_flatness(S=M)
        res_mean_spec_flatness  = np.mean(res_spec_flatness)

        # Spectral rolloff
        res_spec_rolloff        = librosa.feature.spectral_rolloff(S=M)
        res_mean_spec_rolloff   = np.mean(res_spec_rolloff)

    res = {
        "ppitch":               res_ppitch,
        "jitter":               res_jitter,
        "vibrato_depth":        res_vibrato_depth,
        "vibrato_rate":         res_vibrato_rate,
        "shimmer":              res_shimmer,
        "pwr_vals":             res_pwr_vals,
        "f0_vals":              res_f0_vals,
        "spec_centroid":        res_spec_centroid,
        "mean_spec_centroid":   res_mean_spec_centroid,
        "spec_bandwidth":       res_spec_bandwidth,
        "mean_spec_bandwidth":  res_mean_spec_bandwidth,
        "spec_contrast":        res_spec_contrast,
        "mean_spec_contrast":   res_mean_spec_contrast,
        "spec_flatness":        res_spec_flatness,
        "mean_spec_flatness":   res_mean_spec_flatness,
        "spec_rolloff":         res_spec_rolloff,
        "mean_spec_rolloff":    res_mean_spec_rolloff,
    }

    return res


def calculate_vibrato(note_vals, sr):
    """
    Calculate the vibrato depth and rate from a note's frequency signal.

    Parameters
    ----------
    note_vals : np.ndarray
        The time-domain signal values of the note, typically a 1D array representing
        the amplitude of the sound wave over time.

    sr : int
        The sampling rate of the signal in Hertz (samples per second).

    Returns
    -------
    vibrato_depth : float
        The depth of the vibrato, calculated as twice the amplitude of the
        dominant frequency component.
    vibrato_rate : float
        The rate of the vibrato, in Hertz (Hz), derived from the position of
        the dominant frequency in the Fast Fourier Transform (FFT).
    """
    L = len(note_vals)           # Length of signal
    Y = np.fft.fft(note_vals) / L   # Run FFT on normalized note vals
    w = np.arange(0, L) * sr / L    # Set FFT frequency grid

    vibrato_depth_tmp, noteVibratoPos = max(abs(Y)), np.argmax(abs(Y))
    vibrato_depth = vibrato_depth_tmp * 2   # Multiply by 2 (above and below zero)
    vibrato_rate  = w[noteVibratoPos]       # Index into FFT freq grid → Hz

    return vibrato_depth, vibrato_rate


def perceived_pitch(f0s, sr, gamma=100000):
    """
    Calculate the perceived pitch of a note based on
    Gockel, H., B.J.C. Moore, and R.P. Carlyon. 2001.
    Influence of rate of change of frequency on the overall
    pitch of frequency-modulated Tones. Journal of the
    Acoustical Society of America. 109(2):701-12.

    Parameters
    ----------
    f0s : np.ndarray
        Vector of fundamental frequency estimates, typically a 1D array
        representing frequency values over time.

    sr : int
        The sampling rate of the f0 estimates in Hertz (Hz).

    gamma : float, optional
        A parameter that sets the relative weighting of quickly changing
        versus slowly changing portions of notes. A high gamma value (e.g.,
        1000000) gives more weight to slowly changing portions. Default is 100000.

    Returns
    -------
    pp1 : float
        The perceived pitch using the entire vector of f0 estimates.
    pp2 : float
        The perceived pitch using the central 80% of f0 estimates.
    """

    # Remove all NaNs in the f0 vector
    f0s = f0s[~np.isnan(f0s)]

    # Need at least 2 voiced frames for meaningful pitch estimates
    if len(f0s) < 2:
        if len(f0s) == 1:
            return float(f0s[0]), float(f0s[0])
        return np.nan, np.nan

    # Calculate the rate of change (derivative)
    deriv = np.diff(f0s) * sr
    deriv = np.append(deriv, deriv[-1])   # Extend to match original length

    # Weights based on inverse of rate of change
    weights = 1.0 / np.clip(np.abs(deriv), 1e-6, None)
    weights /= np.sum(weights)            # Normalize weights

    # pp1: weighted average of all f0s
    pp1 = float(np.dot(f0s, weights))

    # pp2: weighted average of central 80% of f0 estimates
    ord_ = np.argsort(f0s)
    ind  = ord_[int(np.ceil(len(f0s) * 0.1)) : int(np.floor(len(f0s) * 0.9))]

    if len(ind) == 0:
        pp2 = pp1
    else:
        w_sum = np.sum(weights[ind])
        pp2 = float(np.dot(f0s[ind], weights[ind]) / w_sum) if w_sum > 0 else pp1

    return pp1, pp2
