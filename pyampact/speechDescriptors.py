"""
speechDescriptors
==============


.. autosummary::
    :toctree: generated/

    get_descriptors
        
"""

import parselmouth
import pandas as pd

__all__ = [
    "get_descriptors"
]

def createDataframe(text_file):
    df = pd.read_csv(text_file, header=None, names=["onset_sec", "word"])
    return df

def get_descriptors(audio, text):
    # Convert CSV to dataframes and use onsets to target elements/descriptors
    df = createDataframe(text)    
    snd = parselmouth.Sound(audio)

    pitch = snd.to_pitch()
    intensity = snd.to_intensity()
    formant = snd.to_formant_burg()

    descriptors = []

    for t in df["onset_sec"]:
        pitch_val = pitch.get_value_at_time(t)
        intensity_val = intensity.get_value(t)
        f1 = formant.get_value_at_time(1, t)
        f2 = formant.get_value_at_time(2, t)

        descriptors.append((pitch_val, intensity_val, f1, f2))

    # Add descriptors to the original DataFrame
    df[["pitch", "intensity", "formant1", "formant2"]] = pd.DataFrame(descriptors, index=df.index)

    return df



# Pitch
# Pitch Range
# Pitch floor (Hz) 
# Pitch ceiling (Hz) 
# “octave jump cost”
# “octave cost”
# “voicing threshold”
# “silence threshold”
# Intensity
# Formant
# Pulses
# Jitter
# Shimmer
# CPPS (Soft Central Peak Prominence)
# GNE (glottal-to-noise excitation ratio)
# AVQI (Acoustic Voice Quality Index)
# Spectral Centroid
# Spectral Flux 



    

