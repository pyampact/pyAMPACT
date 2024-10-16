"""
alignmentUtils
==============

.. autosummary::
    :toctree: generated/

    dpcore
    dp
    gh    
    g
    orio_simmx
    simmx
    maptimes    
    f0_est_weighted_sum
    f0_est_weighted_sum_spec
    durations_from_midi_ticks
    load_audiofile

"""

import wave
import numpy as np
import pandas as pd
import librosa
import mido
import warnings


__all__ = [
    "dpcore",
    "dp",
    "gh",
    "g",
    "orio_simmx",
    "simmx",
    "maptimes",
    "f0_est_weighted_sum",
    "f0_est_weighted_sum_spec",
    "durations_from_midi_ticks",
    "load_audiofile"
]


def dpcore(M, pen):
    """Core dynamic programming calculation of best path.
       M[r,c] is the array of local costs.
       Create D[r,c] as the array of costs-of-best-paths to r,c,
       and phi[r,c] as the indicator of the point preceding [r,c] to
       allow traceback; 0 = (r-1,c-1), 1 = (r,c-1), 2 = (r-1, c)

    Parameters
    ----------
    M : np.ndarray
        A 2D array of local costs, where M[r, c] represents the cost at position (r, c).

    pen : float
        A penalty value applied for non-diagonal movements in the path.

    Returns
    -------
    D : np.ndarray
        A 2D array of cumulative best costs to each point (r,c), starting from (0,0).

    phi : np.ndarray
        A 2D array of integers used for traceback, where:
        - 0 indicates the previous point was (r-1, c-1).
        - 1 indicates the previous point was (r, c-1).
        - 2 indicates the previous point was (r-1, c).
    """

    # Pure python equivalent
    D = np.zeros(M.shape, dtype=float)
    phi = np.zeros(M.shape, dtype=int)
    # bottom edge can only come from preceding column
    D[0, 1:] = M[0, 0]+np.cumsum(M[0, 1:]+pen)
    phi[0, 1:] = 1
    # left edge can only come from preceding row
    D[1:, 0] = M[0, 0]+np.cumsum(M[1:, 0]+pen)
    phi[1:, 0] = 2
    # initialize bottom left
    D[0, 0] = M[0, 0]
    phi[0, 0] = 0
    # Calculate the rest recursively
    for c in range(1, np.shape(M)[1]):
        for r in range(1, np.shape(M)[0]):
            best_preceding_costs = [
                D[r-1, c-1], pen+D[r, c-1], pen+D[r-1, c]]
            tb = np.argmin(best_preceding_costs)
            D[r, c] = best_preceding_costs[tb] + M[r, c]
            phi[r, c] = tb

    return D, phi


def dp(local_costs, penalty=0.1, gutter=0.0, G=0.5):
    """
    Use dynamic programming to find a min-cost path through a matrix
    of local costs.

    Parameters
    ----------
    local_costs : np.ndarray
        A 2D matrix of local costs, where each cell represents the cost associated 
        with that specific position.

    penalty : float, optional
        An additional cost incurred for moving in the horizontal or vertical direction 
        (i.e., (0,1) and (1,0) steps). Default is 0.1.

    gutter : float, optional
        A proportion of edge length that allows for deviations away from the 
        bottom-left corner (-1,-1) in the optimal path. Default is 0.0, meaning 
        the path must reach the top-right corner.

    G : float, optional
        A proportion of the edge length considered for identifying gulleys in the 
        cost matrix. Default is 0.5.

    Returns
    -------
    p : np.ndarray
        An array of row indices corresponding to the best path.

    q : np.ndarray
        An array of column indices corresponding to the best path.

    total_costs : np.ndarray
        A 2D array of minimum costs to reach each cell in the local costs matrix.

    phi : np.ndarray
        A traceback matrix indicating the preceding best-path step for each cell, where:
        - 0 indicates a diagonal predecessor.
        - 1 indicates the previous column (same row).
        - 2 indicates the previous row (same column).
    """
    rows, cols = np.shape(local_costs)
    total_costs = np.zeros((rows + 1, cols + 1), float)
    total_costs[0, :] = np.inf
    total_costs[:, 0] = np.inf
    total_costs[0, 0] = 0
    total_costs[1:(rows + 1), 1:(cols + 1)] = local_costs

    phi = np.zeros((rows, cols), int)

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            choices = [total_costs[i - 1, j - 1],
                       total_costs[i - 1, j] + penalty,
                       total_costs[i, j - 1] + penalty]
            tb = np.argmin(choices)
            total_costs[i, j] += choices[tb]
            phi[i - 1, j - 1] = tb + 1

    total_costs = total_costs[1:, 1:]
    phi = phi[1:, 1:]

    if gutter == 0:
        i, j = rows - 1, cols - 1
    else:
        best_top_pt = np.argmin(total_costs[-1, :int(G * cols)])
        best_right_pt = np.argmin(total_costs[:int(G * rows), -1])

        if total_costs[-1, best_top_pt] < total_costs[best_right_pt, -1]:
            i, j = rows - 1, best_top_pt
        else:
            i, j = best_right_pt, cols - 1

    p, q = [i], [j]
    while i > 0 and j > 0:
        tb = phi[i - 1, j - 1]
        if tb == 1:
            i -= 1
            j -= 1
        elif tb == 2:
            i -= 1
        elif tb == 3:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)

    return np.array(p[1:]), np.array(q[1:]), total_costs, phi


def gh(v1, i1, v2, i2, domain, frac=0.5):
    """
    Get an element that is `frac` fraction of the way between `v1[i1]` and `v2[i2]`,
    but check bounds on both vectors. `frac` of 0 returns `v1[i1]`, `frac` of 1 returns `v2[i2]`,
    `frac` of 0.5 (the default) returns halfway between them.

    Parameters
    ----------
    v1 : np.ndarray
        A 1D numpy array representing the first vector.

    i1 : int
        The index in `v1` from which to retrieve the value.

    v2 : np.ndarray
        A 1D numpy array representing the second vector.

    i2 : int
        The index in `v2` from which to retrieve the value.

    domain : tuple
        A tuple representing the valid bounds for both vectors. This should
        define the minimum and maximum allowable indices.

    frac : float, optional
        A fraction indicating how far between the two specified elements 
        to interpolate. Default is 0.5.

    Returns
    -------
    float
        The element that is `frac` fraction of the way between `v1[i1]` 
        and `v2[i2]`, clipped to the specified domain bounds.
    """

    x1 = g(v1, i1, domain)
    x2 = g(v2, i2, domain)
    return int(frac * x1 + (1 - frac) * x2)


def g(vec, idx, domain):
    """
    Get an element from `vec`, checking bounds. `Domain` is the set of points
    that `vec` is a subset of.

    Parameters
    ----------
    vec : np.ndarray
        A 1D numpy array representing the input vector.

    idx : int
        The index of the desired element in `vec`.

    domain : np.ndarray
        A 1D numpy array representing the set of valid points, 
        of which `vec` is a subset.

    Returns
    -------
    float
        The element from `vec` at index `idx` if it is within bounds;
        otherwise, the first element of `domain` if `idx` is less than 0,
        or the last element of `domain` if `idx` exceeds the bounds of `vec`.
    """
    if idx < 1:
        return domain[0]
    elif idx > len(vec):
        return domain[-1]
    else:
        return vec[idx - 1]


def orio_simmx(M, D):
    """
    Calculate an Orio&Schwartz-style (Peak Structure Distance) similarity matrix.

    Parameters
    ----------
    M : np.ndarray
        A binary mask where each column corresponds to a row in the output similarity matrix S.
        The mask indicates the presence or absence of MIDI notes or relevant features.

    D : np.ndarray
        The regular spectrogram where the columns of the similarity matrix S correspond to the columns of D.
        This spectrogram represents the audio signal over time and frequency.

    Returns
    -------
    np.ndarray
        The similarity matrix S, calculated based on the Peak Structure Distance between the binary mask M and the spectrogram D.
    """
    # Convert to NumPy arrays if input is DataFrame
    M = M.values if isinstance(M, pd.DataFrame) else M
    D = D.values if isinstance(D, pd.DataFrame) else D

    # Calculate the similarities
    S = np.zeros((M.shape[1], D.shape[1]))

    D = D**2
    M = M**2

    nDc = np.sqrt(np.sum(D, axis=0))
    nDc = nDc + (nDc == 0)

    # Evaluate one row at a time
    with warnings.catch_warnings():  # Suppress imaginary number warnings, for now
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for r in range(M.shape[1]):
            S[r, :] = np.sqrt(M[:, r] @ D) / nDc

    return S


def simmx(A, B):
    """
    Calculate a similarity matrix between feature matrices A and B.

    Parameters
    ----------
    A : np.ndarray
        The first feature matrix, where each row represents a sample and each column represents a feature.

    B : np.ndarray, optional
        The second feature matrix. If not provided, B will be set to A, allowing for self-similarity calculation.

    Returns
    -------
    np.ndarray
        The similarity matrix between A and B, where the element at (i, j) represents the similarity
        between the i-th sample of A and the j-th sample of B.
    """
    A = A.values if isinstance(A, pd.DataFrame) else A
    B = B.values if isinstance(B, pd.DataFrame) else B

    EA = np.sqrt(np.sum(A**2, axis=0))
    EB = np.sqrt(np.sum(B**2, axis=0))

    M = (A.T @ B) / (EA[:, None] @ EB[None, :])

    return M


def maptimes(t, intime, outtime):
    """
    Map the times in t according to the mapping that each point in intime
    corresponds to that value in outtime.

    Parameters
    ----------
    t : array-like
        The input times to be mapped. This can be a list or a numpy array of time values.

    intime : array-like
        The reference input time points. Each point corresponds to a value in `outtime`.

    outtime : array-like
        The corresponding output time points that map to the values in `intime`.

    Returns
    -------
    np.ndarray
        An array of the mapped times, where each time in `t` is replaced by its corresponding
        value in `outtime` based on the mapping provided by `intime`. If a time in `t` does
        not have a corresponding `intime`, it will be returned unchanged.
    """

    t = np.array(t)
    intime = np.array(intime)
    outtime = np.array(outtime)

    # Ensure that intime and outtime are sorted
    sort_idx = np.argsort(intime)
    intime = intime[sort_idx]
    outtime = outtime[sort_idx]

    u = np.interp(t, intime, outtime, left=outtime[0], right=outtime[-1])
    u = np.round(u, decimals=3)

    return u


# These are the new functions to replace calculate_f0_est
def f0_est_weighted_sum(x, f, f0i, fMax=5000, fThresh=None):
    """
    Calculate F0, power, and spectrum for an inputted spectral representation.

    Parameters
    ----------
    x : np.ndarray, shape (F, T)
        Matrix of complex spectrogram values, where F is the number of frequency bins
        and T is the number of time frames.

    f : np.ndarray, shape (F, T)
        Matrix of frequencies corresponding to each of the spectrogram values in `x`.

    f0i : np.ndarray, shape (1, T)
        Initial estimates of F0 for each time frame. This should be a 1D array containing
        the F0 estimates for each time point.

    fMax : float, optional
        Maximum frequency to consider in the weighted sum. Defaults to 5000 Hz.

    fThresh : float, optional
        Maximum distance in Hz from each harmonic to consider. If not specified, no threshold
        will be applied.

    Returns
    -------    
    f0 : np.ndarray
        Vector of estimated F0s from the beginning to the end of the input time series.
    p : np.ndarray
        Vector of corresponding "powers" derived from the weighted sum of the estimated F0.
    strips : np.ndarray
        Estimated spectrum for each partial frequency based on the weighted contributions.
    """
    # if ~exist('fMax', 'var') || isempty(fMax), fMax = 5000; end
    # if ~exist('fThresh', 'var') || isempty(fThresh), fThresh = 2*median(diff(f(:,1))); end
    # if fMax is None:
    #     fMax = 5000

    if fThresh is None:
        fThresh = 2 * np.nanmedian(np.diff(f[:, 0]))

    f[np.isnan(f)] = 0
    x2 = np.abs(x) ** 2
    np.isnan(x2)
    wNum = np.zeros_like(x2)
    wDen = np.zeros_like(x2)
    maxI = np.max(fMax / f0i[f0i > 0])
    strips = []

    for i in range(1, int(maxI) + 1):
        mask = np.abs(f - (f0i * i)) < fThresh
        strip = x2 * mask
        strips.append(strip)

        wNum += 1 / i * strip  # .toarray()
        wDen += strip  # .toarray()

    wNum *= (f < fMax)
    wDen *= (f < fMax)

    f0 = np.sum(wNum * f, axis=0) / np.sum(wDen, axis=0)
    pow = np.sum(wDen, axis=0)

    return f0, pow, strips


def f0_est_weighted_sum_spec(filename, noteStart_s, noteEnd_s, midiNote, y, sr, useIf=True):
    """
    Calculate F0, power, and spectrum for a single note.

    Parameters
    ----------
    filename : str
        Name of the WAV file to analyze.

    noteStart_s : float
        Start position (in seconds) of the note to analyze.

    noteEnd_s : float
        End position (in seconds) of the note to analyze.

    midiNote : int
        MIDI note number of the note to analyze.

    y : np.ndarray
        Audio time series data from the WAV file.

    sr : int
        Sample rate of the audio signal.

    useIf : bool, optional
        If true, use instantaneous frequency; otherwise, use spectrogram frequencies.
        Defaults to True.

    Returns
    -------    
    f0 : np.ndarray
        Vector of estimated F0s from `noteStart_s` to `noteEnd_s`.
    p : np.ndarray
        Vector of corresponding "powers" derived from the weighted sum of the estimated F0.
    M : np.ndarray
        Estimated spectrum for the analyzed note.
    """

    # set window and hop
    win_s = 0.064
    win = round(win_s * sr)
    hop = round(win / 8)

    # load if gram
    freqs, times, D = librosa.reassigned_spectrogram(
        y=y, sr=sr, hop_length=hop)

   # indices for indexing into ifgram (D)
    noteStart_hop = int(np.floor(noteStart_s * sr / hop))
    noteEnd_hop = int(np.floor(noteEnd_s * sr / hop))
    inds = range(noteStart_hop, noteEnd_hop)

    x = np.abs(D[:, inds])**(1/6)

    f = np.arange(win/2 + 1) * sr / win

    if useIf:
        xf = freqs[:, inds]
    else:
        xf = np.tile(f, (x.shape[1], 1)).T

    f0i = librosa.midi_to_hz(midiNote)

    fMax = 5000
    fThresh = 2 * np.nanmedian(np.diff(xf[:, 0]))

    f0, _, _ = f0_est_weighted_sum(x, xf, f0i, fMax, fThresh)

    _, p, partials = f0_est_weighted_sum(x ** 6, xf, f0, sr)

    M = partials[0]
    for i in range(1, len(partials)):
        M += partials[i]

    t = np.arange(len(inds)) * win_s

    return f0, p, t, M, xf


def durations_from_midi_ticks(filename):
    """
    Extract note durations from a MIDI file using MIDI ticks. This function processes a MIDI file, calculates note onset and offset times 
    based on MIDI ticks and tempo, and returns the duration matrix (nmat). It handles 
    tempo changes and computes times by converting MIDI ticks to seconds.

    Assumes a default pulses-per-quarter-note (PPQN) value of 96.

    Parameters
    ----------
    filename : str
        Path to the MIDI file to be processed.

    Returns
    -------
    np.ndarray
        A numpy array where each row contains the start and end times of notes 
        (in seconds) based on MIDI ticks and tempo changes.

    """

    # symbolic.py converts the files beforehand
    mid = mido.MidiFile(filename)

    nmat = []

    # Set PPQN to 96
    ppqn = 96

    # Convert ticks per beat to seconds per tick
    seconds_per_tick = 60 / (500000 / ppqn)

    # Default tempo in microseconds per quarter note (500000 µs = 120 BPM)
    current_tempo = 500000

    for track in mid.tracks:
        cum_time = 0

        for msg in track:
            cum_time += msg.time

            if msg.type == 'set_tempo':
                current_tempo = msg.tempo
                # Update seconds per tick based on new tempo
                seconds_per_tick = current_tempo / (1_000_000 * ppqn)

            if msg.type == 'note_on' and msg.velocity > 0:
                note = msg.note
                velocity = msg.velocity
                start_time = cum_time * seconds_per_tick
                nmat.append([start_time, 0])

            if msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                for event in reversed(nmat):
                    if event[1] == 0 and event[0] <= cum_time * seconds_per_tick:
                        end_time = cum_time * seconds_per_tick
                        event[1] = end_time
                        break

    # Convert nmat to np.array and return only the 3rd and 4th columns
    return np.array(nmat)


def load_audiofile(audio_file):
    """
    Loads audio file

    Parameters
    ----------
    audio_file : str
        The path to the audio file to be loaded.

    Returns
    -------
    audio_data : The loaded audio data as a numpy array.
    sr : The original sample rate of the audio file.

    """

    with wave.open(audio_file, 'rb') as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()

        # Read the raw audio data
        raw_audio = wf.readframes(n_frames)

        # Convert raw audio to numpy array
        audio_data = np.frombuffer(raw_audio, dtype=np.int16)

        # If stereo, reshape to (n_frames, n_channels)
        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels)

        # Flatten to a single channel if needed
        audio_data = audio_data.flatten()

        return audio_data, sr
