"""
speechDescriptors
==============

.. autosummary::
    :toctree: generated/

    get_speech_descriptors
"""

import parselmouth
import librosa
import pandas as pd
import numpy as np

from pyampact.speechDescriptorsUtils import (
    compute_correlation_dimension,
    compute_dfa,
    compute_emd_features,
    compute_mfcc,
    compute_ppe,
    compute_recurrence_period_and_rpde,
    compute_spectral_spread,
    compute_tqwt_features,
    compute_wt_features,
    feature_spectral_flux,
    compute_vot,
    compute_bark_band_energies,
    compute_dpi,
    compute_avqi,
)

__all__ = ["get_speech_descriptors"]


def createDataframe(speech_file):
    df = pd.read_csv(speech_file, header=None, names=["onset_sec", "pitch_est", "duration", "word"])
    return df


def get_speech_descriptors(audio, df, speaker_type="female"):
    """
    Calculates list of speech descriptors with audio file and CSV provided by the force-align notebook
    https://github.com/pyampact/pyAMPACTtutorials/blob/main/Tutorial_05_pyAMPACT_ForceAlign.ipynb

    Parameters
    ----------
    audio : ndarray
        Audio time series of the speech file        

    df : DataFrame
        df of onset, pitch_est(Hz), duration, and word of each word in passage, output by force align

    speaker_type : str
        Determine the resonance ceiling for formant and CPPS calculation:
            - male = 5000
            - female = 5500
            - child = 8000            

    Returns
    -------
    globals_df : Pandas DataFrame
        The global inputs and values of the whole audio sample
    df : Pandas Dataframe
        All speech descriptors analyzed and output in the process

    """
    # df is a dict of DataFrames — get first part
    df = list(df.values())[0]
    df = df[["ONSET_SEC", "OFFSET_SEC", "AVG PITCH IN HZ", "DURATION", "WORD"]]

    snd = parselmouth.Sound(audio)

    # Analysis parameters (must match Praat script for primitives)
    time_step = 0.01
    pitch_floor = 120
    pitch_ceiling = 600
    octave_cost = 0.01
    octave_jump_cost = 0.35
    voicing_threshold = 0.45
    silence_threshold = 0.03

    # resonance ceiling (Praat script uses 5500 for DB/female)
    resonance_ceiling = {"male": 5000, "female": 5500, "child": 8000}.get(
        speaker_type.lower(), 5500
    )

    # Global pitch metrics (metadata only)
    pitch_cc_global = parselmouth.praat.call(snd, "To Pitch...", time_step, pitch_floor, pitch_ceiling)
    min_pitch_global = parselmouth.praat.call(pitch_cc_global, "Get minimum", 0, 0, "Hertz", "Parabolic")
    max_pitch_global = parselmouth.praat.call(pitch_cc_global, "Get maximum", 0, 0, "Hertz", "Parabolic")
    voiced_global = pitch_cc_global.selected_array["frequency"]
    voiced_global = voiced_global[voiced_global > 0]
    pitch_range = float(voiced_global.max() - voiced_global.min()) if voiced_global.size else 0.0

    # Output columns (keep your existing schema)
    out_cols = [
        "mean_pitch", "pitch_min", "pitch_max", "mean_intensity",
        "mean_f1", "mean_f2", "mean_f3", "mean_f4",
        "jitter_local", "jitter_percent", "shimmer_local", "shimmer_percent",
        "pulse_energy", "cpps", "cpp",
        "mean_spec_centroid", "mean_spec_flux",
        "gne_mean", "gne_max",
        "hnr_mean_db", "nhr_mean_db",
        "noise_intensity_db", "vfer",
        "ppe", "vot_ms",
        "ac_strength", "voicing",
        "mean_bark", "d2", "dfa_alpha",
        "pause_durations", "mean_dpi",
        "emd_n_imfs", "emd_energy_ratio", "emd_entropy",
        "energy", "mfcc",
        "rp", "rpde",
        "mean_spec_spread",
        "tqwt_e1", "tqwt_e2", "tqwt_e3", "tqwt_e4", "tqwt_e5",
        "tqwt_entropy", "tqwt_centroid", "tqwt_low_high_ratio",
        "wt_entropy", "wt_low_high_ratio",
    ]

    rows = []
    min_required = 6.4 / pitch_floor  # 0.053333...

    for i in df.index:
        onset = float(df.at[i, "ONSET_SEC"])
        offset = float(df.at[i, "OFFSET_SEC"])
        dur = offset - onset

        # default undefined (Praat uses undefined)
        row = {k: np.nan for k in out_cols}
        row["pause_durations"] = []
        row["mfcc"] = None

        if not np.isfinite(dur) or dur <= 0 or dur < min_required:
            rows.append(row)
            continue

        # segment with LOCAL time axis [0..dur]
        seg = snd.extract_part(from_time=onset, to_time=offset, preserve_times=False)
        sr = int(seg.sampling_frequency)
        mid = dur / 2.0

        # ========= PRIMITIVES (MATCH PRAAT SCRIPT) =========

        # Pitch (cc)
        p = parselmouth.praat.call(seg, "To Pitch...", time_step, pitch_floor, pitch_ceiling)
        row["pitch_min"] = parselmouth.praat.call(p, "Get minimum", 0, 0, "Hertz", "Parabolic")
        row["pitch_max"] = parselmouth.praat.call(p, "Get maximum", 0, 0, "Hertz", "Parabolic")

        # Mean pitch (not in Praat output table, but keep your column)
        f0 = p.selected_array["frequency"]
        f0v = f0[f0 > 0]
        row["mean_pitch"] = float(np.mean(f0v)) if f0v.size else np.nan

        # Intensity
        w_int = parselmouth.praat.call(seg, "To Intensity...", pitch_floor, time_step, "no")
        row["mean_intensity"] = parselmouth.praat.call(w_int, "Get mean", 0, 0, "dB")

        # PointProcess + jitter/shimmer/pulse energy
        w_pp = parselmouth.praat.call(seg, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
        minPeriod = 1.0 / pitch_ceiling
        maxPeriod = 1.0 / pitch_floor

        row["jitter_local"] = parselmouth.praat.call(
            w_pp, "Get jitter (local)", 0, 0, minPeriod, maxPeriod, 1.3
        )
        row["jitter_percent"] = row["jitter_local"] * 100.0 if np.isfinite(row["jitter_local"]) else np.nan

        row["shimmer_local"] = parselmouth.praat.call(
            [seg, w_pp], "Get shimmer (local)", 0, 0, minPeriod, maxPeriod, 1.3, 1.6
        )
        row["shimmer_percent"] = row["shimmer_local"] * 100.0 if np.isfinite(row["shimmer_local"]) else np.nan

        pulse = parselmouth.praat.call(w_pp, "To Sound (pulse train)", 44100, 1.0, 0.05, 2000)
        row["pulse_energy"] = parselmouth.praat.call(pulse, "Get energy", 0, 0)

        # Formants (midpoint sample, burg params must match)
        w_form = parselmouth.praat.call(seg, "To Formant (burg)...", time_step, 5, resonance_ceiling, 0.025, 50)
        # Keep your mean_f* column names, but fill with Praat midpoint semantics
        row["mean_f1"] = parselmouth.praat.call(w_form, "Get value at time", 1, mid, "Hertz", "Linear")
        row["mean_f2"] = parselmouth.praat.call(w_form, "Get value at time", 2, mid, "Hertz", "Linear")
        row["mean_f3"] = parselmouth.praat.call(w_form, "Get value at time", 3, mid, "Hertz", "Linear")
        row["mean_f4"] = parselmouth.praat.call(w_form, "Get value at time", 4, mid, "Hertz", "Linear")

        # CPPS (only if dur >= 0.1)
        if dur >= 0.1:
            pcg = parselmouth.praat.call(seg, "To PowerCepstrogram...", pitch_floor, time_step, resonance_ceiling, 50)
            minQ = 1.0 / pitch_ceiling
            maxQ = 1.0 / pitch_floor
            row["cpps"] = parselmouth.praat.call(
                pcg, "Get CPPS",
                "yes", 0.02, 0.0005,
                pitch_floor, pitch_ceiling,
                0.05, "parabolic",
                minQ, maxQ,
                "Straight", "Robust"
            )
        else:
            pcg = None

        # GNE (only if dur >= 0.1 in Praat script)
        if dur >= 0.1:
            gneObj = parselmouth.praat.call(seg, "To Harmonicity (gne)...", 500, 4500, 1000, 80)
            row["gne_max"] = parselmouth.praat.call(gneObj, "Get maximum")
            gneVals = np.array(gneObj.values, dtype=float)
            row["gne_mean"] = float(np.nanmean(gneVals)) if gneVals.size else np.nan
        else:
            gneVals = np.array([], dtype=float)

        # HNR / NHR (segment, cc)
        harm = parselmouth.praat.call(seg, "To Harmonicity (cc)...", time_step, pitch_floor, 0.1, 1.0)
        row["hnr_mean_db"] = parselmouth.praat.call(harm, "Get mean", 0, 0)
        row["nhr_mean_db"] = -row["hnr_mean_db"] if np.isfinite(row["hnr_mean_db"]) else np.nan

        # Voicing (AC pitch mean) — exact Praat params
        p_ac = parselmouth.praat.call(
            seg, "To Pitch (ac)",
            time_step, pitch_floor, 15, "off",
            silence_threshold, voicing_threshold,
            octave_cost, octave_jump_cost,
            0.14, pitch_ceiling
        )
        row["voicing"] = parselmouth.praat.call(p_ac, "Get mean", 0, 0, "Hertz")

        # Energy (segment)
        row["energy"] = parselmouth.praat.call(seg, "Get energy", 0, 0)


        # CPP (custom calc) only if you built pcg
        if pcg is not None:
            pcg_mat = np.array(parselmouth.praat.call(pcg, "To Matrix"))
            mean_ceps = np.mean(pcg_mat, axis=1)
            mean_ceps_db = 10 * np.log10(np.maximum(mean_ceps, 1e-12))
            n_quef = mean_ceps_db.size
            quef = np.linspace(0, 1.0 / pitch_floor, n_quef)
            mask_peak = (quef >= 1.0 / pitch_ceiling) & (quef <= 1.0 / pitch_floor)
            if np.any(mask_peak):
                peak_idx = int(np.argmax(mean_ceps_db[mask_peak]))
                peak_q = float(quef[mask_peak][peak_idx])
                peak_val = float(mean_ceps_db[mask_peak][peak_idx])
                mask_trend = (quef >= 0.001) & (quef <= 0.02)
                if np.sum(mask_trend) >= 2:
                    slope, intercept = np.polyfit(quef[mask_trend], mean_ceps_db[mask_trend], 1)
                    row["cpp"] = peak_val - (slope * peak_q + intercept)

        # Spectral features (librosa) on seg
        y = seg.values[0].astype(float, copy=False)
        if y.size:
            spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            row["mean_spec_centroid"] = float(np.mean(spec_centroid)) if spec_centroid.size else np.nan

            S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
            spec_flux = feature_spectral_flux(S, sr)
            row["mean_spec_flux"] = float(np.mean(spec_flux)) if np.size(spec_flux) else np.nan

            row["mean_bark"] = compute_bark_band_energies(S, sr, n_bands=24)

            freqs = np.linspace(0, sr / 2, S.shape[0])
            row["mean_spec_spread"] = compute_spectral_spread(S, freqs)
        else:
            S = None

        # Noise intensity + VFER from HNR (keep your definitions)
        if np.isfinite(row["hnr_mean_db"]) and np.isfinite(row["mean_intensity"]):
            hnr_lin = 10 ** (row["hnr_mean_db"] / 10.0)
            noise_fraction = 1.0 / (1.0 + hnr_lin)
            row["noise_intensity_db"] = row["mean_intensity"] + 10 * np.log10(noise_fraction)
            row["vfer"] = hnr_lin / (1.0 + hnr_lin)

        # PPE (use segment pitch object p on local grid)
        t_values = np.arange(0.0, dur, time_step)
        pitch_segment = np.array([p.get_value_at_time(t) for t in t_values], dtype=float)
        row["ppe"] = compute_ppe(pitch_segment)

        # VOT (use seg + p)
        row["vot_ms"] = compute_vot(seg, p, sr)

        # AC strength (Praat Pitch (ac) confidence)
        if "strength" in p_ac.selected_array.dtype.names:
            strength = p_ac.selected_array["strength"].astype(float, copy=False)
            strength = strength[~np.isnan(strength)]
            row["ac_strength"] = float(np.mean(strength)) if strength.size else np.nan
        else:
            row["ac_strength"] = np.nan


        # Correlation dimension / DFA / EMD on gneVals (as you had)
        if gneVals.size:
            row["d2"] = compute_correlation_dimension(gneVals)
            row["dfa_alpha"] = compute_dfa(gneVals)
            row["emd_n_imfs"], row["emd_energy_ratio"], row["emd_entropy"] = compute_emd_features(gneVals)

        # DPI (use seg + seg intensity + segment pitch p)
        if dur >= min_required:
            seg_intensity = seg.to_intensity(minimum_pitch=pitch_floor, time_step=time_step, subtract_mean=False)
            pause_durations, mean_dpi = compute_dpi(
                seg, p, seg_intensity, intensity_thresh=row["mean_intensity"] - 10
            )
            row["pause_durations"] = pause_durations
            row["mean_dpi"] = mean_dpi

        # RP / RPDE (use segment point process w_pp, local times)
        rp, rpde = compute_recurrence_period_and_rpde(w_pp, 0.0, dur)
        row["rp"] = rp
        row["rpde"] = rpde

        # TQWT / WT (keep)
        if y.size:
            (
                row["tqwt_e1"], row["tqwt_e2"], row["tqwt_e3"], row["tqwt_e4"], row["tqwt_e5"],
                row["tqwt_entropy"], row["tqwt_centroid"], row["tqwt_low_high_ratio"],
            ) = compute_tqwt_features(y=y, sr=sr, Q=1.0, r=3.0, J=8, n_keep=5)

            row["wt_entropy"], row["wt_low_high_ratio"] = compute_wt_features(y)

        # MFCC optional
        row["mfcc"] = None  # keep as you had (disabled)

        rows.append(row)

    # attach results
    out_df = pd.DataFrame(rows, index=df.index)
    df[out_cols] = out_df[out_cols]

    # ---------- REORDER COLUMNS FOR CSV OUTPUT ONLY ----------

    COLUMN_ORDER = [
        # identity / segmentation
        "WORD", "ONSET_SEC", "OFFSET_SEC", "DURATION",

        # pitch & voicing
        "mean_pitch", "pitch_min", "pitch_max", "voicing", "ac_strength", "ppe", "vot_ms",

        # intensity & energy
        "mean_intensity", "pulse_energy", "energy", "noise_intensity_db", "vfer",

        # perturbation
        "jitter_local", "jitter_percent", "shimmer_local", "shimmer_percent",

        # formants
        "mean_f1", "mean_f2", "mean_f3", "mean_f4",

        # cepstral / harmonic
        "cpps", "cpp", "gne_max", "gne_mean", "hnr_mean_db", "nhr_mean_db",

        # spectral
        "mean_spec_centroid", "mean_spec_flux", "mean_spec_spread", "mean_bark",

        # temporal / nonlinear
        "rp", "rpde", "d2", "dfa_alpha", "pause_durations", "mean_dpi",

        # decomposition
        "emd_n_imfs", "emd_energy_ratio", "emd_entropy",

        # transforms
        "tqwt_e1", "tqwt_e2", "tqwt_e3", "tqwt_e4", "tqwt_e5",
        "tqwt_entropy", "tqwt_centroid", "tqwt_low_high_ratio",
        "wt_entropy", "wt_low_high_ratio",

        # heavy payloads
        "mfcc",
    ]

    # keep only columns that exist (Praat vs Python compatibility)
    ordered_cols = [c for c in COLUMN_ORDER if c in df.columns]
    df = df.loc[:, ordered_cols]


    globals_df = pd.DataFrame([{
        "pitch_floor": pitch_floor,
        "pitch_ceiling": pitch_ceiling,
        "pitch_range": pitch_range,
        "octave_cost": octave_cost,
        "octave_jump_cost": octave_jump_cost,
        "voicing_threshold": voicing_threshold,
        "silence_threshold": silence_threshold,
        "formant_ceiling": resonance_ceiling,
        "time_step": time_step,
        "speaker_type": speaker_type,
        "global_pitch_min": float(min_pitch_global) if np.isfinite(min_pitch_global) else np.nan,
        "global_pitch_max": float(max_pitch_global) if np.isfinite(max_pitch_global) else np.nan,
    }])

    return globals_df, df
