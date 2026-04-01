"""
alignmentUtils
==============

.. autosummary::
    :toctree: generated/

    dp
    gh    
    g
    orio_simmx    
    maptimes    
    f0_est_weighted_sum
    f0_est_weighted_sum_spec
    trim_silences
    merge_grace_notes

"""

import wave
import numpy as np
import pandas as pd
import librosa
import warnings


__all__ = [
    "dp",
    "gh",
    "g",
    "orio_simmx",    
    "maptimes",
    "f0_est_weighted_sum",
    "f0_est_weighted_sum_spec",
    "trim_silences",
    "merge_grace_notes"
]


def dp(M):
    """
    Port of Dan Ellis dp.m
    [p,q,D] = dp(M)
    """
    M = np.asarray(M, dtype=float)
    r, c = M.shape

    D = np.zeros((r + 1, c + 1), dtype=float)
    D[0, :] = np.nan
    D[:, 0] = np.nan
    D[0, 0] = 0.0
    D[1:r+1, 1:c+1] = M

    phi = np.zeros((r, c), dtype=np.int32)

    for i in range(r):
        for j in range(c):
            # MATLAB: min([D(i,j), D(i,j+1), D(i+1,j)])
            choices = np.array([D[i, j], D[i, j+1], D[i+1, j]])
            tb = int(np.nanargmin(choices)) + 1  # 1..3
            D[i+1, j+1] = D[i+1, j+1] + choices[tb - 1]
            phi[i, j] = tb

    # Traceback from bottom-right (r-1,c-1) to (0,0)
    i = r - 1
    j = c - 1
    p = [i]
    q = [j]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            tb = phi[i, j]
            if tb == 1:
                i -= 1
                j -= 1
            elif tb == 2:
                i -= 1
            elif tb == 3:
                j -= 1
            else:
                raise RuntimeError("bad traceback")
        p.insert(0, i)
        q.insert(0, j)

    D = D[1:r+1, 1:c+1]
    return np.asarray(p, dtype=int), np.asarray(q, dtype=int), D

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

def maptimes(t, intime, outtime):
    """
    Map onset/offset times using a monotone linear interpolation from `intime` to `outtime`.
    Handles duplicate `intime` entries (DTW path repeats) by averaging their `outtime`.
    """

    t = np.asarray(t, dtype=float)
    intime = np.asarray(intime, dtype=float)
    outtime = np.asarray(outtime, dtype=float)

    original_shape = t.shape
    x = t.reshape(-1)

    # Sort by intime
    order = np.argsort(intime)
    intime = intime[order]
    outtime = outtime[order]

    # Collapse duplicate intime values by averaging outtime
    uniq, inv = np.unique(intime, return_inverse=True)
    out_sum = np.bincount(inv, weights=outtime)
    out_cnt = np.bincount(inv)
    out_u = out_sum / np.maximum(out_cnt, 1)

    # Enforce nondecreasing outtime (DTW should be monotone; this prevents tiny inversions)
    out_u = np.maximum.accumulate(out_u)

    # Linear interpolation with endpoint clamping
    y = np.interp(x, uniq, out_u, left=out_u[0], right=out_u[-1])

    return y.reshape(original_shape)


# These are the new functions to replace calculate_f0_est
def f0_est_weighted_sum(x, f, f0i, fMax=20000, fThresh=None):
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

        wNum += 1 / i * strip
        wDen += strip

    wNum *= (f < fMax)
    wDen *= (f < fMax)

    f0 = np.sum(wNum * f, axis=0) / np.sum(wDen, axis=0)
    pow = np.sum(wDen, axis=0)

    return f0, pow, strips


def f0_est_weighted_sum_spec(noteStart_s, noteEnd_s, midiNote, freqs, D, sr, useIf=True):
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
    

   # indices for indexing into ifgram (D)
    noteStart_hop = int(np.floor(noteStart_s * sr / hop))
    noteEnd_hop = int(np.floor(noteEnd_s * sr / hop))

    # # Cap noteEnd_hop to the maximum allowed index
    # noteEnd_hop = min(noteEnd_hop, D.shape[1])
    
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


# Internal utility function, no documentation
def trim_silences(nmat_dict, y, sr, rms_thresh_db=-40.0, pad=0.25):       
    """
    Remove or clamp note events in a note matrix that fall outside the active
    audio region, as determined by an RMS energy threshold.

    Parameters
    ----------
    nmat_dict : pd.DataFrame
        Note matrix dictionary keyed by part name.

    y : np.ndarray
        Audio time series at sample rate ``sr``.

    sr : int
        Sample rate of ``y`` in Hz.

    rms_thresh_db : float, optional
        RMS energy threshold in dBFS below which frames are considered silent.
        Default is ``-40.0``.

    pad : float, optional
        Reserved for future use. Currently unused. Default is ``0.25``.

    Returns
    -------
    nmat_dict : pd.DataFrame
        The input dictionary with each part's DataFrame trimmed in-place to the
        active audio window.
    """
    # Compute RMS and active time window
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    active = rms > librosa.db_to_amplitude(rms_thresh_db)    

    if not np.any(active):
        raise ValueError("No active audio found above threshold")

    first_active_idx = np.argmax(active)
    last_active_idx = len(active) - 1 - np.argmax(active[::-1])
    start_trim = times[first_active_idx]
    end_trim = times[last_active_idx]

    for part, df in nmat_dict.items():
        new_rows = []
        for _, row in df.iterrows():
            onset = row['ONSET_SEC']
            offset = row['OFFSET_SEC']

            # Discard completely silent notes
            if offset < start_trim or onset > end_trim:
                continue

            # Clamp only edges
            new_onset = onset if onset >= start_trim else start_trim
            new_offset = offset if offset <= end_trim else end_trim

            if new_offset > new_onset:
                new_row = row.copy()
                new_row['ONSET_SEC'] = new_onset
                new_row['OFFSET_SEC'] = new_offset
                if 'DURATION' in row:
                    new_row['DURATION'] = new_offset - new_onset
                new_rows.append(new_row)

        nmat_dict[part] = pd.DataFrame(new_rows, columns=df.columns)

    return nmat_dict

# Internal utility function, no documentation
# Merges grace notes and odd polyrhythmic values into one voice and orders them
def merge_grace_notes(nmat, offset=0.025):
    """
    Merge grace-note sub-parts back into their parent voice and resolve any
    resulting onset overlaps.

    Parameters
    ----------
    nmat : pd.DataFrame
        Note matrix dictionary keyed by part name.

    offset : float, optional
        Time in seconds added to the ``ONSET_SEC`` and ``OFFSET_SEC`` of every
        grace-note sub-part before merging. Default is ``0.025``.

    Returns
    -------
    nmat : dict of str → pd.DataFrame
        The updated note matrix with all grace-note sub-parts folded into their
        respective base parts and removed as separate keys.
    """
    base_parts = {}
    for part in list(nmat.keys()):
        base = part.split(":")[0]
        base_parts.setdefault(base, []).append(part)

    for base, parts in base_parts.items():
        if len(parts) == 1:
            continue  # no merging needed

        dfs = []
        for part in parts:
            df = nmat[part].copy()
            if part != base:
                # Apply offset to ONSET_SEC (time in seconds), not ONSET (beats)
                df['ONSET_SEC'] += offset
                df['OFFSET_SEC'] += offset
            dfs.append(df)

        # Sort by ONSET_SEC (time in seconds) instead of ONSET (beats)
        merged = pd.concat(dfs).sort_values('ONSET_SEC').copy()

        # Remove duplicate indices that might cause reindexing issues
        merged = merged[~merged.index.duplicated(keep='first')]

        # Recalculate DURATION in beats based on the new ordering
        # But preserve the original ONSET_SEC and OFFSET_SEC timing
        # Only update DURATION to match the time-based values
        merged['DURATION'] = merged['OFFSET_SEC'] - merged['ONSET_SEC']

        # Ensure minimum duration to prevent zero-duration notes
        merged['DURATION'] = merged['DURATION'].clip(lower=0.01)  # Minimum 10ms
        merged['OFFSET_SEC'] = merged['ONSET_SEC'] + merged['DURATION']

        # Ensure no overlapping notes by adjusting onset times if necessary
        merged = merged.sort_values('ONSET_SEC')
        for i in range(1, len(merged)):
            prev_offset = merged.iloc[i-1]['OFFSET_SEC']
            curr_onset = merged.iloc[i]['ONSET_SEC']
            if curr_onset < prev_offset:
                # Adjust current onset to avoid overlap
                merged.iloc[i, merged.columns.get_loc('ONSET_SEC')] = prev_offset + 0.001
                merged.iloc[i, merged.columns.get_loc('OFFSET_SEC')] = merged.iloc[i]['ONSET_SEC'] + merged.iloc[i]['DURATION']

        # Replace in dict
        nmat[base] = merged
        for part in parts:
            if part != base:
                del nmat[part]

    return nmat    


