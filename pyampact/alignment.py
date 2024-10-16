"""
alignment
==============


.. autosummary::
    :toctree: generated/

    run_alignment
    run_DTW_alignment
    align_midi_wav
    alignment_visualiser
    ifgram
    get_ons_offs
"""

from pyampact.alignmentUtils import orio_simmx, simmx, dp, maptimes
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd

import sys
import os
sys.path.append(os.pardir)


__all__ = [
    "run_alignment",
    "run_DTW_alignment",
    "align_midi_wav",
    "alignment_visualiser",
    "ifgram",
    "get_ons_offs"
]


def run_alignment(y, original_sr, piece, nmat, width=3, target_sr=4000, nharm=3, win_ms=100, hop=32, showSpec=False):
    """
    Parameters
    ----------
    y : ndarray
        Audio time series.
    original_sr : int
        Original sample rate of the audio file.
    piece : Score
        A `Score` instance containing the symbolic MIDI data.
    means : ndarray
        Mean values for each state in the alignment process.
    covars : ndarray
        Covariance values for each state in the alignment process.
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
    showSpec : bool
        If True, displays the spectrogram of the audio.

    Returns
    -------
    align : dict
        MIDI-audio alignment structure from DTW containing:
        - 'on': Onset times of the notes.
        - 'off': Offset times of the notes.
        - 'midiNote': MIDI note numbers corresponding to the aligned notes.
    dtw : dict
        A dictionary of DTW returns, including:
        - 'M': The map such that M[:,m] corresponds to the alignment.
        - 'MA': Path from dynamic programming (DP) for MIDI-audio alignment.
        - 'RA': Path from DP for real audio alignment.
        - 'S': Similarity matrix used in the alignment process.
        - 'D': Spectrogram of the audio.
        - 'notemask': The MIDI-note-derived mask used in the alignment.
    spec : ndarray
        Spectrogram of the audio file.
    newNmat : DataFrame
        Updated DataFrame containing the note matrix (nmat) data after alignment.

    Notes
    -----
    This function leverages DTW to align MIDI note information with the time series audio signal. 
    It computes onset and offset times and updates the alignment using a similarity matrix. 
    Optionally, it can display the audio spectrogram for visual analysis.
    """

    # Normalize audio file
    y = y / np.sqrt(np.mean(y ** 2)) * 0.6

    # Run DTW alignment
    spec, dtw, newNmat = run_DTW_alignment(
        y, original_sr, piece, 0.025, width, target_sr, nharm, win_ms, hop, nmat, showSpec)

    nmat = newNmat
    return dtw, spec, nmat


def run_DTW_alignment(y, original_sr, piece, tres, width, target_sr, nharm, win_ms, hop, nmat, showSpec):
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
    showSpec : bool
        If True, displays the spectrogram of the audio file.

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

    p, q, S, D, M = align_midi_wav(
        piece, WF=y, sr=original_sr, TH=tres, ST=0, width=width, tsr=target_sr, nhar=nharm, hop=hop, wms=win_ms, showSpec=showSpec)

    dtw = {
        'MA': p,
        'RA': q,
        'S': S,
        'D': D,
        'notemask': M,
    }

    # Avoid log(0) by replacing with smallest nonzero value
    D[D == 0] = np.min(D[D > 0])

    if showSpec == True:
        # Plot spectrogram
        plt.subplot(2, 1, 1)
        plt.imshow(20 * np.log10(D), aspect='auto',
                   origin='lower', cmap='gray_r')
        plt.colorbar()
        plt.clim(np.max(20 * np.log10(D)) + np.array([-50, 0]))

        # Zoom in to see the detail
        maxcol = min(1000, min(M.shape[1], D.shape[1]))
        plt.xlim([0, maxcol])
        plt.ylim([0, D.shape[0]])

        plt.show()

    spec = dtw['D']

    dtw['MA'] = np.array(dtw['MA']-1)*tres
    dtw['RA'] = np.array(dtw['RA']-1)*tres

    dtw['RA'] = dtw['RA'] * 1/2

    # loop through voices
    onset_sec = []
    offset_sec = []

    for key, df in nmat.items():
        onset_sec = df['ONSET_SEC'].values
        offset_sec = df['OFFSET_SEC'].values

        onsOffs = np.array([[on, off]
                           for on, off in zip(onset_sec, offset_sec)])

        maskLength = M.shape[1] * tres
        factor = maskLength / onsOffs.max()
        onsOffs = onsOffs * factor
        onsOffs = np.round(onsOffs, decimals=3)

        x = maptimes(onsOffs, dtw['MA'], dtw['RA'])

        df.loc[:, 'ONSET_SEC'] = x[:, 0]
        df.loc[:, 'OFFSET_SEC'] = x[:, 1]
        df.at[df.index[0], 'ONSET_SEC'] = 0  # Set first value to 0 always

    return spec, dtw, nmat


def align_midi_wav(piece, WF, sr, TH, ST, width, tsr, nhar, wms, hop, showSpec):
    """
    Align a midi file to a wav file using the "peak structure
    distance" of Orio et al. that use the MIDI notes to build
    a mask that is compared against harmonics in the audio

    Parameters
    ----------
    piece : Score
        A `Score` instance containing the symbolic MIDI data.
    WF : ndarray
        Audio time series of the WAV file.
    sr : int
        Sampling rate of the audio file.
    TH : float
        Time step resolution, typically in seconds (default is 0.050).
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
    showSpec : bool
        If True, displays the spectrogram.

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

    # Calculate spectrogram
    fft_len = int(2**np.round(np.log(wms/1000*tsr)/np.log(2)))
    ovlp = round(fft_len - TH*tsr)
    # y = librosa.resample(WF, orig_sr=sr, target_sr=tsr)
    y = signal.resample(WF, int(len(WF) * tsr / sr))

    freqs, times, D = signal.stft(y, fs=tsr, window='hamming',
                                  nperseg=fft_len, noverlap=ovlp, nfft=fft_len)

    D = np.abs(D)

    # Normalize D
    D_max = np.max(D)
    if D_max != 0:
        D = D / D_max

    times = librosa.times_like(D, sr=tsr, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=tsr, n_fft=fft_len)

    if showSpec == True:
        alignment_visualiser(D, times, freqs, showSpec=showSpec)

    M = piece.mask(sample_rate=tsr, num_harmonics=nhar,
                   width=width, winms=wms, obs=24)

    # Calculate the peak-structure-distance similarity matrix
    if ST == 1:
        S = orio_simmx(M, D)
    else:
        S = simmx(M, D)

    # Ensure no NaNs (only going to happen with simmx)
    S[np.isnan(S)] = 0

    p, q, D, phi = dp(1-S)

    # Add harms to nmat
    harm = piece.harm(snap_to=M, output='series')
    if not harm.isna().all():
        M = pd.concat((M, harm))

    return p, q, S, D, M


def alignment_visualiser(audio_spec, times=None, freqs=None, fig=1, showSpec=True):
    """
    Visualizes the dynamic time warping (DTW) alignment.    

    Parameters
    ----------
    audio_spec : ndarray
        Spectrogram of the audio file to be visualized.
    times : ndarray, optional
        Array of segment times corresponding to the audio spectrogram. If not provided, defaults to None.
    freqs : ndarray, optional
        Array of sample frequencies corresponding to the audio spectrogram. If not provided, defaults to None.
    fig : int, optional
        Figure number for the plot. Default is 1.
    showSpec : bool, optional
        If True, displays the spectrogram overlayed with the alignment information. Default is True.

    Returns
    -------
    matplotlib.figure.Figure
        The visualized spectrogram plot with DTW alignment overlays.
    """

    if (len(times) > 0 and len(freqs) > 0) and showSpec == True:
        # Convert complex to real and take the magnitude
        plt.pcolormesh(times, freqs, 10 *
                       np.log10(np.abs(audio_spec)), shading='auto')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title('Alignment Spectrogram')
        plt.colorbar(label='Power/Frequency (dB/Hz)')
        # plt.show()
    else:
        # print("To show spectrogram, make sure to provide freqs/times matrices and set showSpec=True")
        return


def ifgram(audiofile, tsr, win_ms, showSpec=False):
    """
    Compute the instantaneous frequency (IF) spectrogram of an audio file using
    the reassigned spectrogram and Short-Time Fourier Transform (STFT).

    Parameters
    ----------
    audiofile : str
        Path to the audio file to be analyzed.
    tsr : int
        Target sample rate of the audio signal.
    win_ms : float
        Window size in milliseconds for spectral analysis.
    showSpec : bool, optional
        If True, displays the spectrogram of the reassigned spectrogram. Default is False.

    Returns
    -------
    freqs : ndarray
        Reassigned frequency bins of the spectrogram.
    times : ndarray
        Time frames corresponding to the spectrogram.
    mags : ndarray
        Magnitudes of the reassigned spectrogram.
    f0_values : ndarray
        Fundamental frequency estimates for each time frame.
    mags_mat : ndarray
        Magnitude matrix from the Short-Time Fourier Transform (STFT).
    """

    # win_samps = int(tsr / win_ms) # low-res
    win_samps = 2048  # Placeholder for now, default
    y, sr = librosa.load(audiofile)

    freqs, times, mags = librosa.reassigned_spectrogram(y=y, sr=tsr,
                                                        n_fft=win_samps, reassign_frequencies=False)

    # Find the index of the maximum magnitude frequency bin for each time frame
    max_mag_index = np.argmax(mags, axis=0)

    # Extract the corresponding frequencies as f0 values
    f0_values = freqs[max_mag_index]

    # Calculate the Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)

    # Extract the magnitude and phase information
    mags_mat = np.abs(D)
    mags_db = librosa.amplitude_to_db(mags, ref=np.max)

    if showSpec == True:
        fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
        img = librosa.display.specshow(
            mags_db, x_axis="s", y_axis="linear", sr=tsr, hop_length=win_samps//4, ax=ax[0])
        ax[0].set(title="Spectrogram", xlabel=None)
        ax[0].label_outer()
        ax[1].scatter(times, freqs, c=mags_db, cmap="magma", alpha=0.1, s=5)
        ax[1].set_title("Reassigned spectrogram")
        fig.colorbar(img, ax=ax, format="%+2.f dB")

        # plt.show()

    return freqs, times, mags, f0_values, mags_mat


def get_ons_offs(onsoffs):
    """
    Extract onset and offset times from a 3*N alignment matrix generated by AMPACT's 
    HMM-based alignment algorithm.

    Parameters
    ----------
    onsoffs : ndarray
        A 3*N alignment matrix where:
        - The first row contains N states.
        - The second row contains the corresponding ending times for each state.
        - The third row contains the state indices.

    Returns
    -------
    res : dict
        A dictionary containing:
        - 'ons': List of onset times.
        - 'offs': List of offset times.

    """

    # Find indices where the first row is equal to 3
    stopping = np.where(onsoffs[0] == 3)[0]

    # Calculate starting indices by subtracting 1 from stopping indices
    starting = stopping - 1

    res = {'ons': [], 'offs': []}
    for i in range(len(starting)):
        res['ons'].append(onsoffs[1, starting[i]])
        res['offs'].append(onsoffs[1, stopping[i]])

    return res
