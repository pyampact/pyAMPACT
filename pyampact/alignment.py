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

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

import sys
import os
sys.path.append(os.pardir)

from scipy.signal import spectrogram

from pyampact.alignmentUtils import orio_simmx, simmx, dp, maptimes

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
    Calls the DTW alignment function.


    :param y: Audio time series
    :param original_sr: original sample rate of audio
    :param piece: Score instance of symbolic data
    :param means: Mean values for each state
    :param covars: Covariance values for each state
    :param width: Width parameter (you need to specify this value)
    :param target_sr: Target sample rate (you need to specify this value)
    :param nharm: Number of harmonics (you need to specify this value)
    :param win_ms: Window size in milliseconds (you need to specify this value)
    :param hop: Number of samples between successive frames
    :param showSpec: Boolean to show the spectrogram

    :returns:
        - align: Dynamic time warping MIDI-audio alignment structure
            - on: onset times
            - off: offset times
            - midiNote: MIDI note numbers
        - dtw: Dict of dynamic time warping returns
            - M: map s.t. M(:,m)
            - MA/RA [p,q]: path from DP
            - S: similarity matrix
            - D: spectrogram
            - notemask: midi-note-derived mask
            - pianoroll: midi-note-derived pianoroll
        - spec: Spectrogram of the audio file
        - newNmat: updated DataFrame of nmat data
    """

    # Normalize audio file
    y = y / np.sqrt(np.mean(y ** 2)) * 0.6

    # Run DTW alignment
    align, spec, dtw, newNmat = run_DTW_alignment(
        y, original_sr, piece, 0.050, width, target_sr, nharm, win_ms, hop, nmat, showSpec)

    nmat = newNmat
    return align, dtw, spec, nmat
    
def run_DTW_alignment(y, original_sr, piece, tres, width, target_sr, nharm, win_ms, hop, nmat, showSpec):
    """
    Perform a dynamic time warping alignment between specified audio and MIDI files.

    Returns a matrix with the aligned onset and offset times (with corresponding MIDI
    note numbers) and a spectrogram of the audio.

    :param y: Audio time series of audio
    :param original_sr: original sample rate of audio
    :param piece: Score instance of symbolic data
    :param tres: Time resolution for MIDI to spectrum information conversion.
    :param width: Width parameter (you need to specify this value)
    :param target_sr: Target sample rate (you need to specify this value)
    :param nharm: Number of harmonics (you need to specify this value)
    :param win_ms: Window size in milliseconds (you need to specify this value)
    :param showSpec: Boolean to show the spectrogram

    :returns:
        - align: Dynamic time warping MIDI-audio alignment structure
            - align.on: onset times
            - align.off: offset times
            - align.midiNote: MIDI note numbers
        - spec: Spectrogram of the audio file
        - dtw: Dict of dynamic time warping returns
            - M: map s.t. M(:,m)
            - MA/RA [p,q]: path from DP
            - S: similarity matrix
            - D: spectrogram
            - notemask: midi-note-derived mask
            - pianoroll: midi-note-derived pianoroll
        - nmat: updated DataFrame of nmat data
    """


    m, p, q, S, D, M, N = align_midi_wav(
        piece, WF=y, sr=original_sr, TH=tres, ST=0, width=width, tsr=target_sr, nhar=nharm, hop=hop, wms=win_ms, showSpec=showSpec)

    dtw = {
        'M': m,
        'MA': p,
        'RA': q,
        'S': S,
        'D': D,
        'notemask': M,
        'pianoroll': N
    }

    spec = dtw['D']


    # Iterate through each key-value pair (dataframe) in the nmat dictionary
    for key, df in nmat.items():
        # Filter out rows where MIDI column is not equal to -1.0
        filtered_df = df[df['MIDI'] != -1.0]
        # Store the filtered dataframe in the filtered_nmat dictionary with the same key
        nmat[key] = filtered_df

    # this should be the same dimensions as the nmat parts
    align = {
        'nmat': nmat.copy(),
        'on': np.empty(0),         # Create an empty 1D array
        'off': np.empty(0),        # Create an empty 1D array
        'midiNote': np.empty(0)    # Create an empty 1D array
    }

    #tres = 0.025
    #dtw['MA'] = np.array(dtw['MA']) * tres
    #dtw['RA'] = np.array(dtw['RA']) * tres


    # loop through voices
    onset_sec = []
    offset_sec = []
    midi_notes = []


    for key, df in nmat.items():
        onset_sec = df['ONSET'].values
        offset_sec = df['ONSET'].values+df['DURATION'].values
        # double check if this is necessary
        # onset_sec = onset_sec/(np.max(offset_sec)/(len(y)/original_sr))
        # offset_sec = offset_sec/(np.max(offset_sec)/(len(y)/original_sr))
        midi_notes = df['MIDI'].values

        # combined_slice = np.column_stack((np.concatenate(onset_sec), np.concatenate(offset_sec)))
        combined_slice = [[on, off] for on, off in zip(onset_sec, offset_sec)]
        combined_slice = np.array(combined_slice)

        #maptimes(align.nmat(:,6:7),(dtw.MA-1)*tres,(dtw.RA-1)*tres)
        #x = pyampact.maptimes(combined_slice, [x * tres for x in dtw['MA']], [x * tres for x in dtw['RA']])
 
        x = maptimes(combined_slice, dtw['MA'], dtw['RA'])
 

        # Assign 'on', 'off', and 'midiNote' values from nmat
        align['on'] = np.append(align['on'], x[:,0])
        align['off'] = np.append(align['off'], x[:,1])
        align['midiNote'] = np.append(align['midiNote'], midi_notes)
        spec = D # from align_midi_wav

        df.loc[:,'ONSET_SEC'] = x[:,0]
        df.loc[:,'OFFSET_SEC'] = x[:,1]
        df.at[df.index[0], 'ONSET_SEC'] = 0 # Set first value to 0 always

    return align, spec, dtw, nmat


def align_midi_wav(piece, WF, sr, TH, ST, width, tsr, nhar, wms, hop, showSpec):    
    """
    Align a midi file to a wav file using the "peak structure
    distance" of Orio et al. that use the MIDI notes to build 
    a mask that is compared against harmonics in the audio
        
    :param MF: Score instance of symbolic data
    :param WF: Audio time series of file
    :param TH: is the time step resolution (default 0.050)
    :param ST: is the similarity type: 0 (default) is triangle inequality
    :param showSpec: Boolean to show the spectrogram

    :returns:
        - m: Is the map s.t. M(:,m)          
        - [p,q]: Are the path from DP
        - S: The similarity matrix
        - D: Is the spectrogram
        - M: Is the midi-note-derived mask
        - N: Is Orio-style "peak structure distance"
    """
        
    pianoRoll = piece.pianoRoll()

    # Construct N
    sampled_grid = []
    for row in pianoRoll:
        sampled_grid.append(row)

    N = np.array(sampled_grid)


    # Calculate spectrogram
    fft_len = int(2**np.round(np.log(wms/1000*tsr)/np.log(2)))
    y = librosa.resample(WF, orig_sr=sr, target_sr=tsr)


    # Compute spectrogram
    D = np.abs(librosa.stft(y, n_fft=fft_len, hop_length=hop))
    times = librosa.times_like(D, sr=sr, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=fft_len)


    if showSpec == True:
        alignment_visualiser(D, times, freqs, showSpec=showSpec)

    M = piece.mask(sample_rate=tsr, num_harmonics=nhar, width=width, winms=wms)


    # Calculate the peak-structure-distance similarity matrix
    if ST == 0:
        S = orio_simmx(M, D)
    else:
        S = simmx(M, D)

    # Ensure no NaNs (only going to happen with simmx)
    S[np.isnan(S)] = 0


    # Do the DP search
    #p, q, D_dp = dpmod(1 - S)
    p, q, D_dp = dp(1-S)

    m = np.zeros(D.shape[0], dtype=int)
    for i in range(D.shape[0]):
        if np.any(q == i):
            m[i] = p[np.min(np.where(q == i))]
        else:
            m[i] = 1

    return m, p, q, S, D, M, N

  

def alignment_visualiser(audio_spec, times=None, freqs=None, fig=1, showSpec=True):    
    """    
    Plots a gross DTW alignment overlaid with the fine alignment
    resulting from the HMM aligner on the output of YIN.  Trace(1,:)
    is the list of states in the HMM, and trace(2,:) is the number of YIN
    frames for which that state is occupied.  Highlight is a list of 
    notes for which the steady state will be highlighted.
    
    :param audio_spec: Spectrogram of audio file
    :param freqs: Array of sample frequencies
    :param times: Array of segment times    
    
    :return: Visualized spectrogram    
    """    

    
    
    if (len(times) > 0 and len(freqs) > 0) and showSpec == True:
        plt.pcolormesh(times, freqs, 10 * np.log10(np.abs(audio_spec)), shading='auto')  # Convert complex to real and take the magnitude
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title('Alignment Spectrogram')    
        plt.colorbar(label='Power/Frequency (dB/Hz)')
        plt.show()
    else:
        # print("To show spectrogram, make sure to provide freqs/times matrices and set showSpec=True")
        return
    
    


def ifgram(audiofile, tsr, win_ms, showSpec=False):    
    # win_samps = int(tsr / win_ms) # low-res
    win_samps = 2048 # Placeholder for now, default
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
        img = librosa.display.specshow(mags_db, x_axis="s", y_axis="linear",sr=tsr, hop_length=win_samps//4, ax=ax[0])
        ax[0].set(title="Spectrogram", xlabel=None)
        ax[0].label_outer()
        ax[1].scatter(times, freqs, c=mags_db, cmap="magma", alpha=0.1, s=5)
        ax[1].set_title("Reassigned spectrogram")
        fig.colorbar(img, ax=ax, format="%+2.f dB")
                      
        plt.show()
    
    return freqs, times, mags, f0_values, mags_mat

def get_ons_offs(onsoffs):
    """
    Extracts a list of onset and offset from an inputted 
             3*N matrix of states and corresponding ending times 
             from AMPACT's HMM-based alignment algorithm
    
    :param onsoffs: A 3*N alignment matrix, the first row is a list of N states
        the second row is the time which the state ends, and the
        third row is the state index
    :returns: 
        - res.ons: List of onset times
        - res.offs: List of offset times
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
