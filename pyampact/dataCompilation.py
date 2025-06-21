"""
dataCompilation
===============




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


def data_compilation(y, original_sr, nmat, piece, audio_file_path, output_path='output.mei'):
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

    # y, original_sr = librosa.load(audio_file)
    # print(original_sr)

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
            if df['OFFSET_SEC'].iloc[loc] - df['ONSET_SEC'].iloc[loc] > 0:                
                [f0, pwr, t, M, xf] = f0_est_weighted_sum_spec(
                    df['ONSET_SEC'].iloc[loc], df['OFFSET_SEC'].iloc[loc], midiList[loc], y, original_sr)
                # Estimate note-wise perceptual values
                # if flag = M1, call pass in original, if M2 pass in reconstructed
                note_vals.append(estimate_perceptual_parameters(
                    f0, pwr, M, original_sr, 256, 1))
                note_ids.append(nmat[key].index[loc])
            else:                
                print([df['ONSET_SEC'].iloc[loc], df['OFFSET_SEC'].iloc[loc]])
        all_note_vals.append(note_vals)
        all_note_ids.append(note_ids)

    loc = 0

    for key, df in nmat.items():
        df['f0Vals'] = [all_note_vals[loc][i]['f0_vals'] for i in range(len(df))]
        df['meanf0'] = [np.mean(vals) for vals in df['f0Vals']]

        df['ppitch1'] = [all_note_vals[loc][i]['ppitch'][0] for i in range(len(df))]
        df['ppitch2'] = [all_note_vals[loc][i]['ppitch'][1] for i in range(len(df))]
        df['jitter'] = [all_note_vals[loc][i]['jitter'] for i in range(len(df))]

        df['vibratoDepth'] = [all_note_vals[loc][i]['vibrato_depth'] for i in range(len(df))]
        df['vibratoRate'] = [all_note_vals[loc][i]['vibrato_rate'] for i in range(len(df))]

        df['pwrVals'] = [all_note_vals[loc][i]['pwr_vals'] for i in range(len(df))]
        df['meanPwr'] = [np.mean(vals) for vals in df['pwrVals']]
        df['shimmer'] = [all_note_vals[loc][i]['shimmer'] for i in range(len(df))]

        df['specCentVals'] = [all_note_vals[loc][i]['spec_centroid'] for i in range(len(df))]
        df['meanSpecCent'] = [np.mean(vals) for vals in df['specCentVals']]

        df['specBandwidthVals'] = [all_note_vals[loc][i]['spec_bandwidth'] for i in range(len(df))]
        df['meanSpecBandwidth'] = [np.mean(vals) for vals in df['specBandwidthVals']]

        df['specContrastVals'] = [all_note_vals[loc][i]['spec_contrast'] for i in range(len(df))]
        df['meanSpecContrast'] = [np.mean(vals) for vals in df['specContrastVals']]

        df['specFlatnessVals'] = [all_note_vals[loc][i]['spec_flatness'] for i in range(len(df))]
        df['meanSpecFlatness'] = [np.mean(vals) for vals in df['specFlatnessVals']]

        df['specRolloffVals'] = [all_note_vals[loc][i]['spec_rolloff'] for i in range(len(df))]
        df['meanSpecRolloff'] = [np.mean(vals) for vals in df['specRolloffVals']]

        loc += 1
        
    
    def convert_nmat_for_export(nmat):
        """
        Converts columns with list values into stringified versions for export (e.g., MEI, CSV, JSON).

        Parameters
        ----------
        nmat : dict of DataFrames
            The processed note matrix with internal Python lists.

        Returns
        -------
        export_nmat : dict of DataFrames
            A new dictionary where list-valued columns are stringified.
        """
        list_columns = [
            'f0Vals', 'pwrVals', 'specCentVals', 'specBandwidthVals',
            'specContrastVals', 'specFlatnessVals', 'specRolloffVals'
        ]
        
        export_nmat = {}
        for part, df in nmat.items():
            df_copy = df.copy()
            for col in list_columns:
                if col in df_copy.columns:
                    df_copy[col] = df_copy[col].apply(lambda x: str(x))
            export_nmat[part] = df_copy

        return export_nmat


    nmat_export = convert_nmat_for_export(nmat)

    if getattr(piece, 'fileExtension') != 'csv':        
        output_path = './output.mei'
        fileOutput = piece.insertAudioAnalysis(
            output_path=output_path, data=nmat_export, mimetype='audio/aiff', target=audio_file_path)
    else:
        # Save each part of nmat to a CSV file
        for part, df in nmat_export.items():
            # Construct the filename using the filepath, label, and part
            filename = f"./output.csv"

            # Save the DataFrame to CSV
            fileOutput = df.to_csv(filename)
            print(f"Saved {filename}")

    return nmat, fileOutput
