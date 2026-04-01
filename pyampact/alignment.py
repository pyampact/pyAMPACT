"""
alignment
==============


.. autosummary::
    :toctree: generated/

    run_alignment
    run_DTW_alignment
    align_midi_wav
"""

from pyampact.alignmentUtils import *
from pyampact.symbolic import *
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
import scipy.signal

import sys
import os
sys.path.append(os.pardir)

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")



__all__ = [
    "run_alignment",
    "run_DTW_alignment",
    "align_midi_wav",
]


def run_alignment(audio_file, score_file, width=3, target_sr=4000, nharm=3, win_ms=100, hop=32):
    """
    Parameters
    ----------
    audio_file : string
        Path to audio file
    score_file : string
        Path to score/symbolic file
    width : float
        Width parameter for the DTW alignment.
    target_sr : int
        Target sample rate for resampling the audio (if needed).
    nharm : int
        Number of harmonics to include in the analysis.
    win_ms : float
        Window size in milliseconds for the analysis.
    hop : int
        Number of samples between successive frames.

    Returns
    -------
    spec : ndarray
        Spectrogram of the audio file.
    piece : Music21 Object
        All labels in Music21 format
    newNmat : DataFrame
        Updated DataFrame containing the note matrix (nmat) data after alignment.
    y : ndarray
        Audio data of audio_file
    original_sr : int
        Sample rate returned by audio_file

    Notes
    -----
    This function leverages DTW to align MIDI note information with the time series audio signal. 
    It computes onset and offset times and updates the alignment using a similarity matrix. 
    Optionally, it can display the audio spectrogram for visual analysis.
    """
    
    piece = load_score(score_file)

    pd.set_option('display.max_rows', None)

    nmat = nmats(piece)

    y, original_sr = librosa.load(
        audio_file,
        sr=None,
        mono=True,
        dtype=np.float32
    )
    # Normalize audio file
    y = y / np.sqrt(np.mean(y ** 2)) * 0.6
        
    nmat = merge_grace_notes(nmat) 

    # Tempo rescaling
    audio_duration = len(y) / original_sr
    score_end = max(
        df['OFFSET_SEC'].max()
        for df in nmat.values()
        if not df.empty and 'OFFSET_SEC' in df.columns
    )
    if score_end > 0 and audio_duration > 0:
        scale = audio_duration / score_end
        for df in nmat.values():
            df['ONSET_SEC']  = (df['ONSET_SEC']  * scale).round(3)
            df['OFFSET_SEC'] = (df['OFFSET_SEC'] * scale).round(3)
            df['DURATION']   = (df['OFFSET_SEC'] - df['ONSET_SEC']).round(6)

    # Run DTW alignment
    spec, newNmat = run_DTW_alignment(
        y, original_sr, piece, 0.025, width, target_sr, nharm, win_ms, hop, nmat)
    
    
    trimmedNmat = trim_silences(newNmat, y, original_sr, rms_thresh_db=-40.0)        
    return spec, piece, trimmedNmat, y, original_sr


def run_DTW_alignment(y, original_sr, piece, tres, width, target_sr, nharm, win_ms, hop, nmat):
    """
    Perform a dynamic time warping (DTW) alignment between an audio file and its corresponding MIDI file.

    This function returns the aligned onset and offset times with corresponding MIDI note numbers, 
    as well as the spectrogram of the audio and other DTW-related data.

    Parameters
    ----------
    y : ndarray
        Audio time series of the file.
    original_sr : int
        Original sample rate of the audio file.
    piece : Score
        A `Score` instance containing the symbolic (MIDI) data.
    tres : float
        Time resolution for MIDI-to-spectrum information conversion.
    width : float
        Width parameter for the DTW alignment.
    target_sr : int
        Target sample rate for resampling the audio (if needed).
    nharm : int
        Number of harmonics to include in the analysis.
    win_ms : float
        Window size in milliseconds for analysis.
    hop : int
        Number of samples between successive frames for analysis.
    nmat : DataFrame
        DataFrame containing note matrix (nmat) data before alignment.

    Returns
    -------
    align : dict
        MIDI-audio alignment structure from DTW containing:
        - 'on': Onset times of the notes.
        - 'off': Offset times of the notes.
        - 'midiNote': MIDI note numbers corresponding to the aligned notes.
    spec : ndarray
        Spectrogram of the audio file.
    dtw : dict
        A dictionary of DTW returns, including:
        - 'M': The map such that M[:,m] corresponds to the alignment.
        - 'MA/RA': Path from dynamic programming (DP) for MIDI-audio alignment.
        - 'S': Similarity matrix used in the alignment process.
        - 'D': Spectrogram of the audio.
        - 'notemask': The MIDI-note-derived mask used in the alignment.
        - 'pianoroll': MIDI-note-derived piano roll.
    nmat : DataFrame
        Updated DataFrame containing the note matrix (nmat) data after alignment.
    """
    
    p, q, S, D, M, times = align_midi_wav(
        nmat, piece, WF=y, sr=original_sr, TH=tres, width=width, tsr=target_sr, nhar=nharm, hop=hop, wms=win_ms)
    
    # Avoid log(0) by replacing with smallest nonzero value
    D[D == 0] = np.min(D[D > 0])

    spec = D
    
    p = np.asarray(p, dtype=int)
    q = np.asarray(q, dtype=int)

    p_sec = p * tres
    q_sec = times[q]


    """
    p = (np.array(p)-1)*tres
    q = (np.array(q)-1)*tres

    """
        
    # Fix csv nmat formatting
    if (getattr(piece, 'fileExtension')) == 'csv':
        nmat['Part-1'] = nmat['Part-1'].reset_index()

        # Recreate the index for each row
        nmat['Part-1'].index = [
            f"pyAMPACT-{i+1}" for i in range(len(nmat['Part-1']))]

        # Move key columns to the front, keep the rest
        front = ['MEASURE', 'ONSET', 'DURATION', 'PART', 'MIDI', 'ONSET_SEC', 'OFFSET_SEC']
        rest = [c for c in nmat['Part-1'].columns if c not in front]
        nmat['Part-1'] = nmat['Part-1'][front + rest]

    if (getattr(piece, 'fileExtension')) != 'csv':
        for key, df in nmat.items():            
            onsOffs = np.column_stack([df['ONSET_SEC'].values, df['OFFSET_SEC'].values])            
            onsOffs = np.column_stack([df['ONSET_SEC'].values, df['OFFSET_SEC'].values])
            onsOffs = np.round(onsOffs, 3)
            mapped = maptimes(onsOffs, p_sec, q_sec)


            df['ONSET_SEC'] = np.round(mapped[:, 0], 3)
            df['OFFSET_SEC'] = np.round(mapped[:, 1], 3)
            df['DURATION'] = np.round(df['OFFSET_SEC'] - df['ONSET_SEC'], 6)

    return spec, nmat


def align_midi_wav(nmat, piece, WF, sr, TH, width, tsr, nhar, wms, hop):
    """
    Align a midi file to a wav file using the "peak structure
    distance" of Orio et al. that use the MIDI notes to build
    a mask that is compared against harmonics in the audio

    Parameters
    ----------
    piece : Music21 Object
        Object from load_score with all data and labels
    WF : ndarray
        Audio time series of the WAV file.
    sr : int
        Sampling rate of the audio file.
    TH : float
        Time step resolution, typically in seconds (default is 0.025).
    ST : int
        Similarity type; 0 (default) uses the triangle inequality.
    width : float
        Width of the mask for the analysis.
    tsr : int
        Target sample rate for resampling the audio (if needed).
    nhar : int
        Number of harmonics to include in the mask.
    wms : float
        Window size in milliseconds.
    hop : int
        Hop size for the analysis window.

    Returns
    -------
    m : ndarray
        The map such that M[:,m] corresponds to the alignment.
    path : tuple of ndarrays
        [p, q], the path from dynamic programming (DP) that aligns the MIDI and audio.
    S : ndarray
        The similarity matrix used for alignment.
    D : ndarray
        The spectrogram of the audio.
    M : ndarray
        The MIDI-note-derived mask, including harmonic information if available.
    """

    # STFT config: hop is defined by TH, not by external hop_length
    fft_len = int(2 ** np.round(np.log(wms / 1000 * tsr) / np.log(2)))
    hop_samp = int(round(TH * tsr))
    ovlp = int(fft_len - hop_samp)
    ovlp = max(0, min(ovlp, fft_len - 1))

    # resample audio to tsr without changing duration semantics
    y = signal.resample(WF, int(round(len(WF) * tsr / sr)))

    freqs, times, D = signal.stft(
        y, fs=tsr, window="hamming",
        nperseg=fft_len, noverlap=ovlp, nfft=fft_len
    )

    # magnitude + normalize
    D = np.abs(D)
    mx = D.max()
    if mx > 0:
        D /= mx


    # NOTE THAT I TOOK THESE OUT!!! WHAT HAPPENS?!
    # force time axis to match TH exactly (used later for p,q scaling)
    # frame_sec = hop_samp / tsr
    # times = np.arange(D.shape[1]) * frame_sec
    # TH = frame_sec  # enforce consistency everywhere downstream

    # build mask in absolute seconds from the existing music21 score
    M = build_mask_from_nmat_seconds(
        nmat, sample_rate=tsr, num_harmonics=nhar,
        width=width, tres=TH, n_freqs=D.shape[0]
    )

    S = orio_simmx(M, D)

    # Drop your own DTW here...
    p, q, total_costs = dp(1.0-S)    
    
    return p, q, S, D, M, times
