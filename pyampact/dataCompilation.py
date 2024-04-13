"""
dataCompilations
==============


.. autosummary::
    :toctree: generated/
    
    data_compilation
"""

import numpy as np
import librosa
from pyampact.performance import estimate_perceptual_parameters
from pyampact.alignmentUtils import f0_est_weighted_sum_spec
from pyampact.symbolic import Score

__all__ = [
    "data_compilation"
]

def data_compilation(res, nmat, audio_file):
    """
    This function takes the results of the alignment and the note matrix and compiles the data into a JSON object
    that can be used to insert the audio analysis into the score.
    :param res: The results of the alignment
    :param nmat: The note matrix
    :param audio_file: The audio file
    :return: A JSON object that contains the data
    """
    y, original_sr = librosa.load(audio_file)

    all_note_vals = []
    all_note_ids = []

    for key, df in nmat.items():

        midiList = np.array(nmat[key]['MIDI'])
        loc = 1
        f0 = []
        pwr = []
        t = []
        M = []
        xf = []

        ons = res['on']
        offs = res['off']

        note_vals = []
        note_ids = []

        # # ons = np.nonzero(estimatedOns)e[0]
        # # offs = np.nonzero(estimatedOffs)[0]
        for loc in range(len(ons)):
            #Estimate f0 for a matrix (or vector) of amplitudes and frequencies
            [f0, pwr, t, M, xf] = f0_est_weighted_sum_spec(audio_file, ons[loc], offs[loc], midiList[loc], y, original_sr);
            # Estimate note-wise perceptual values
            note_vals.append(estimate_perceptual_parameters(f0, pwr, M, original_sr, 256, 1))
            note_ids.append(nmat[key].index[loc])
        all_note_vals.append(note_vals)
        all_note_ids.append(note_ids)

    loc = 0

    for key, df in nmat.items():
        for i in range (len(df)):
            # Create a dictionary for the current time interval - added np.mean
            df.loc[i,'f0Vals'] = str(all_note_vals[loc][i]['f0_vals'])
            df.loc[i,'meanf0'] = np.mean(all_note_vals[loc][i]['f0_vals'])
            df.loc[i,'ppitch1'] = all_note_vals[loc][i]['ppitch'][0]
            df.loc[i,'ppitch2'] = all_note_vals[loc][i]['ppitch'][1]
            df.loc[i,'jitter'] = all_note_vals[loc][i]['jitter']
            df.loc[i,'vibratoDepth'] = all_note_vals[loc][i]['vibrato_depth']
            df.loc[i,'vibratoRate'] = all_note_vals[loc][i]['vibrato_rate']
            df.loc[i,'pwrVals'] = str(all_note_vals[loc][i]['pwr_vals'])
            df.loc[i,'shimmer'] = all_note_vals[loc][i]['shimmer']
            df.loc[i,'meanPwr'] = np.mean(all_note_vals[loc][i]['pwr_vals'])
            df.loc[i,'specCentVals'] = str(all_note_vals[loc][i]['spec_centroid'])
            df.loc[i,'meanSpecCent'] = np.mean(all_note_vals[loc][i]['spec_centroid'])
            df.loc[i,'specBandwidthVals'] = str(all_note_vals[loc][i]['spec_bandwidth'])
            df.loc[i,'meanSpecBandwidth'] = np.mean(all_note_vals[loc][i]['spec_bandwidth'])
            df.loc[i,'specContrastVals'] = str(all_note_vals[loc][i]['spec_contrast'])
            df.loc[i,'meanSpecContrast'] = np.mean(all_note_vals[loc][i]['spec_contrast'])
            df.loc[i,'specFlatnessVals'] = str(all_note_vals[loc][i]['spec_flatness'])
            df.loc[i,'meanSpecFlatness'] = np.mean(all_note_vals[loc][i]['spec_flatness'])
            df.loc[i,'specRolloffVals'] = str(all_note_vals[loc][i]['spec_rolloff'])
            df.loc[i,'meanSpecRolloff'] = np.mean(all_note_vals[loc][i]['spec_rolloff'])

        loc += 1

    nmat, jsonData = Score.toJSON(nmat)
    return nmat, jsonData
    
