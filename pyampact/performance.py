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

    M : np.ndarray
        The spectral estimates from which the perceptual parameters will be derived.

    SR : float
        The sample rate used to generate the fundamental frequency estimates.

    hop : float
        The hop size used in the analysis, expressed in seconds.

    gt_flag : bool
        A flag indicating whether to use ground truth information for the estimation.

    Returns
    -------
    pp1 : float
        The perceived pitch calculated using the entire vector of f0 estimates.
    pp2 : float
        The perceived pitch calculated using the central 80% of f0 estimates.

    """

    # Perceived pitch
    res_ppitch = perceived_pitch(f0_vals, SR)
    # Jitter
    tmp_jitter = np.abs(np.diff(f0_vals))
    res_jitter = np.mean(tmp_jitter)

    # Vibrato rate and depth
    mean_f0_vals = np.mean(f0_vals)
    detrended_f0_vals = f0_vals - mean_f0_vals
    res_vibrato_depth, res_vibrato_rate = calculate_vibrato(
        detrended_f0_vals, SR / hop)

    # Shimmer
    tmp_shimmer = 10 * np.log10(pwr_vals[1:] / pwr_vals[0])
    res_shimmer = np.mean(np.abs(tmp_shimmer))
    res_pwr_vals = 10 * np.log10(pwr_vals)
    res_f0_vals = f0_vals

    if gt_flag:
        M = np.abs(M) ** 2

    # spectral bandwidth
    res_spec_bandwidth = librosa.feature.spectral_bandwidth(S=M)
    res_mean_spec_bandwidth = np.mean(res_spec_bandwidth)

    # Spectral Centroid
    # S, phase = librosa.magphase(librosa.stft(y=y))
    res_spec_centroid = librosa.feature.spectral_centroid(S=M)
    res_mean_spec_centroid = np.mean(res_spec_centroid)

    # spectral contrast
    res_spec_contrast = librosa.feature.spectral_contrast(S=M)
    res_mean_spec_contrast = np.mean(res_spec_contrast)

    # spectral flatness
    res_spec_flatness = librosa.feature.spectral_flatness(S=M)
    res_mean_spec_flatness = np.mean(res_spec_flatness)

    # spectral rolloff
    res_spec_rolloff = librosa.feature.spectral_rolloff(S=M)
    res_mean_spec_rolloff = np.mean(res_spec_rolloff)

    # Spectral Flatness
    XLog = np.log(M + 1e-20)
    res_spec_flat = np.exp(np.mean(XLog, axis=0)) / np.mean(M, axis=0)
    res_spec_flat[np.sum(M, axis=0) == 0] = 0
    res_mean_spec_flat = np.mean(res_spec_flat)

    res = {
        "ppitch": res_ppitch,
        "jitter": res_jitter,
        "vibrato_depth": res_vibrato_depth,
        "vibrato_rate": res_vibrato_rate,
        "shimmer": res_shimmer,
        "pwr_vals": res_pwr_vals,
        "f0_vals": res_f0_vals,
        "spec_centroid": res_spec_centroid,
        "mean_spec_centroid": res_mean_spec_centroid,
        "spec_bandwidth": res_spec_bandwidth,
        "mean_spec_bandwidth": res_mean_spec_bandwidth,
        "spec_contrast": res_spec_contrast,
        "mean_spec_contrast": res_mean_spec_contrast,
        "spec_flatness": res_spec_flatness,
        "mean_spec_flatness": res_mean_spec_flatness,
        "spec_rolloff": res_spec_rolloff,
        "mean_spec_rolloff": res_mean_spec_rolloff,
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
    L = len(note_vals)  # Length of signal
    Y = np.fft.fft(note_vals) / L  # Run FFT on normalized note vals
    w = np.arange(0, L) * sr / L  # Set FFT frequency grid

    vibrato_depth_tmp, noteVibratoPos = max(abs(Y)), np.argmax(
        abs(Y))  # Find the max value and its position
    # Multiply the max by 2 to find depth (above and below zero)
    vibrato_depth = vibrato_depth_tmp * 2
    # Index into FFT frequency grid to find position in Hz
    vibrato_rate = w[noteVibratoPos]

    return vibrato_depth, vibrato_rate


def perceived_pitch(f0s, sr, gamma=100000):
    """
    Calculate the perceived pitch of a note based on 
    Gockel, H., B.J.C. Moore,and R.P. Carlyon. 2001. 
    Influence of rate of change of frequency on the overall 
    pitch of frequency-modulated Tones. Journal of the 
    Acoustical Society of America. 109(2):701?12.

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

    # Calculate the rate of change (derivative)
    deriv = np.diff(f0s) * sr
    deriv = np.append(deriv, deriv[-1])  # Extend to match original length

    # Weights based on inverse of rate of change
    # Using np.clip to avoid division by zero for small values
    weights = 1.0 / np.clip(np.abs(deriv), 1e-6, None)
    weights /= np.sum(weights)  # Normalize weights

    # Calculate pp1: Weighted average of f0s based on smooth weights
    pp1 = np.dot(f0s, weights)

    # Calculate central 80% of the f0 vector
    ord = np.argsort(f0s)
    ind = ord[int(np.ceil(len(f0s) * 0.1)):int(np.floor(len(f0s) * 0.9))]

    # pp2: Weighted average of central 80% of f0 estimates
    pp2 = np.dot(f0s[ind], weights[ind]) / np.sum(weights[ind])

    return pp1, pp2
