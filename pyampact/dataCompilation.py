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


def data_compilation(nmat, audio_file, piece, output_path='./output.mei'):
    """
    This function takes the results of the alignment and the note matrix and compiles the data into a JSON object
    that can be used to insert the audio analysis into the score.

    Parameters
    ----------
    nmat : np.ndarray
        The note matrix containing information about notes, including their timing and duration.

    audio_file : str
        The path to the audio file associated with the performance data.

    piece : Score
        An instantiation of the original Score object containing the data input for the musical piece.

    output_path : str, optional
        The file path for the output MEI file. Defaults to './output.mei'.

    Returns
    -------            
    nmat : The note matrix with performance data appended.
    json_data : A JSON object containing the compiled data.
    xml_data : XML data representing the MEI output.

    """

    y, original_sr = librosa.load(audio_file)

    all_note_vals = []
    all_note_ids = []

    for key, df in nmat.items():
        df = df.drop(columns=['MEASURE', 'ONSET', 'DURATION', 'PART', 'MIDI'])

        midiList = np.array(nmat[key]['MIDI'])
        loc = 1
        f0 = []
        pwr = []
        t = []
        M = []
        xf = []

        note_vals = []
        note_ids = []

        for loc in range(len(df)):
            # Estimate f0 for a matrix (or vector) of amplitudes and frequencies
            [f0, pwr, t, M, xf] = f0_est_weighted_sum_spec(
                audio_file, df['ONSET_SEC'].iloc[loc], df['OFFSET_SEC'].iloc[loc], midiList[loc], y, original_sr)
            # Estimate note-wise perceptual values
            note_vals.append(estimate_perceptual_parameters(
                f0, pwr, M, original_sr, 256, 1))
            note_ids.append(nmat[key].index[loc])
        all_note_vals.append(note_vals)
        all_note_ids.append(note_ids)

    loc = 0

    for key, df in nmat.items():
        # Create new columns for each attribute
        df['f0Vals'] = [str(all_note_vals[loc][i]['f0_vals'])
                        for i in range(len(df))]
        df['meanf0'] = [np.mean(all_note_vals[loc][i]['f0_vals'])
                        for i in range(len(df))]
        df['ppitch1'] = [all_note_vals[loc][i]['ppitch'][0]
                         for i in range(len(df))]
        df['ppitch2'] = [all_note_vals[loc][i]['ppitch'][1]
                         for i in range(len(df))]
        df['jitter'] = [all_note_vals[loc][i]['jitter']
                        for i in range(len(df))]
        df['vibratoDepth'] = [all_note_vals[loc][i]['vibrato_depth']
                              for i in range(len(df))]
        df['vibratoRate'] = [all_note_vals[loc][i]['vibrato_rate']
                             for i in range(len(df))]
        df['pwrVals'] = [str(all_note_vals[loc][i]['pwr_vals'])
                         for i in range(len(df))]
        df['shimmer'] = [all_note_vals[loc][i]['shimmer']
                         for i in range(len(df))]
        df['meanPwr'] = [np.mean(all_note_vals[loc][i]['pwr_vals'])
                         for i in range(len(df))]
        df['specCentVals'] = [str(all_note_vals[loc][i]['spec_centroid'])
                              for i in range(len(df))]
        df['meanSpecCent'] = [
            np.mean(all_note_vals[loc][i]['spec_centroid']) for i in range(len(df))]
        df['specBandwidthVals'] = [
            str(all_note_vals[loc][i]['spec_bandwidth']) for i in range(len(df))]
        df['meanSpecBandwidth'] = [
            np.mean(all_note_vals[loc][i]['spec_bandwidth']) for i in range(len(df))]
        df['specContrastVals'] = [
            str(all_note_vals[loc][i]['spec_contrast']) for i in range(len(df))]
        df['meanSpecContrast'] = [
            np.mean(all_note_vals[loc][i]['spec_contrast']) for i in range(len(df))]
        df['specFlatnessVals'] = [
            str(all_note_vals[loc][i]['spec_flatness']) for i in range(len(df))]
        df['meanSpecFlatness'] = [
            np.mean(all_note_vals[loc][i]['spec_flatness']) for i in range(len(df))]
        df['specRolloffVals'] = [
            str(all_note_vals[loc][i]['spec_rolloff']) for i in range(len(df))]
        df['meanSpecRolloff'] = [
            np.mean(all_note_vals[loc][i]['spec_rolloff']) for i in range(len(df))]

        loc += 1

    meiOutput = piece.insertAudioAnalysis(
        output_path=output_path, data=nmat, mimetype='audio/aiff', target=audio_file)

    return nmat, meiOutput
