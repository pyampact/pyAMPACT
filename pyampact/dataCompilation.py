"""
dataCompilation
===============

.. autosummary::
    :toctree: generated/

    data_compilation
"""

import os
import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib.patches import Patch
from scipy import signal

from pyampact.performance import estimate_perceptual_parameters
from pyampact.alignmentUtils import f0_est_weighted_sum_spec
from pyampact.symbolic import *

__all__ = [
    "data_compilation",
    "export_selected_columns",
    "visualise_alignment_from_nmat",
    "plot_piano_roll",
]


def data_compilation(
    y,
    original_sr,
    hop_length,
    winms,
    tsr,
    spec,
    nmat,
    piece,
    audio_file_path,
    force_pyin=False
):
    """
    Compile per-note perceptual descriptors from an aligned audio-score pair
    and write the results to disk in the appropriate format (.krn, .mei, or .csv).

    F0 estimation strategy is selected automatically based on the number of
    parts in the score: monophonic pieces use pyin directly; polyphonic pieces
    attempt pitch-separated F0 estimation via the reassigned spectrogram and
    fall back to pyin on failure. Set ``force_pyin=True`` to override and use
    pyin for all pieces regardless of polyphony.

    Parameters
    ----------
    y : ndarray
        Audio time series at the original sample rate.

    original_sr : int
        Sample rate of ``y``.

    hop_length : int
        Hop size in samples at ``tsr`` used during alignment (e.g. 32).

    winms : float
        Analysis window size in milliseconds (e.g. 100).

    tsr : int
        Target sample rate used during alignment (e.g. 4000).

    spec : ndarray
        Magnitude spectrogram produced by alignment, shape (freq x frames).

    nmat : dict
        Note matrix dict returned by ``run_alignment``, keyed by part name.
        Each value is a DataFrame with at minimum ONSET_SEC, OFFSET_SEC, and MIDI columns.

    piece : Score
        Score object returned by ``run_alignment`` / ``load_score``.

    audio_file_path : str
        Path to the source audio file. Used to derive the output folder name
        (``output_files/output_<stem>/``) and filename stem.

    force_pyin : bool, optional
        If True, use pyin F0 estimation for all notes regardless of whether
        the piece is monophonic or polyphonic. Default is False (auto-detect).

    Returns
    -------
    nmat : dict
        The input note matrix dict with perceptual descriptor columns appended
        to each part DataFrame, including: f0Vals, meanf0, ppitch1, ppitch2,
        jitter, vibratoDepth, vibratoRate, pwrVals, meanPwr, shimmer,
        specCentVals, meanSpecCent, specBandwidthVals, meanSpecBandwidth,
        specContrastVals, meanSpecContrast, specFlatnessVals, meanSpecFlatness,
        specRolloffVals, meanSpecRolloff.

    fileOutput : str
        Path to the primary output file written to disk:
        a .krn file for Humdrum kern scores, a .csv file for Tony CSV scores,
        or a .mei file (with companion .csv) for all other formats.

    Notes
    -----
    Spectral features (centroid, bandwidth, contrast, flatness, rolloff) are
    precomputed once over the full spectrogram and sliced per note to avoid
    redundant computation. F0 and RMS power are computed at ``original_sr``
    using window and hop sizes derived from the alignment spectrogram dimensions
    rather than the caller-supplied ``hop_length``, which ensures consistency
    with the DTW alignment grid.
    """
    # --- Derive output folder from the audio filename ---
    # e.g. "./test_files/B063_00-01.wav"  →  "output_files/output_B063_00-01/"
    audio_stem = os.path.splitext(os.path.basename(audio_file_path))[0]
    output_dir = os.path.join("output_files", f"output_{audio_stem}")
    os.makedirs(output_dir, exist_ok=True)
    # Base path for all output files (no extension — each writer appends its own)
    output_path = os.path.join(output_dir, audio_stem)

    all_note_vals = []
    all_note_ids = []

    # fft_len at tsr (alignment sample rate)
    fft_len_tsr = int(2 ** np.round(np.log(winms / 1000 * tsr) / np.log(2)))
    fft_len_tsr = max(256, fft_len_tsr)

    # Magnitude spectrogram (freq x frames) – produced at tsr by alignment
    S = np.abs(spec)

    # Derive the actual hop used by align_midi_wav from the spec dimensions and
    # the audio duration at tsr.  This is robust regardless of what hop_length
    # value the caller passes in (which is often the run_alignment default of 32,
    # not the true STFT hop of ~100).
    n_frames = S.shape[1]
    n_samples_tsr = int(round(len(y) * tsr / original_sr))
    # align_midi_wav uses boundary=None, padded=False:
    #   n_frames = 1 + (n_samples - fft_len) // hop_samp  =>  hop_samp = (n_samples - fft_len) // (n_frames - 1)
    if n_frames > 1:
        hop_samp_tsr = max(1, (n_samples_tsr - fft_len_tsr) // (n_frames - 1))
    else:
        hop_samp_tsr = fft_len_tsr

    # Frame time grid for spec (derived from actual hop at tsr)
    frame_times = np.arange(n_frames) * hop_samp_tsr / tsr

    # Convert to original_sr samples for librosa pyin / rms analysis
    hop_length_orig = max(1, int(round(hop_samp_tsr * original_sr / tsr)))
    fft_len_orig    = max(256, int(round(fft_len_tsr * original_sr / tsr)))

    # Global f0 (pyin) at original_sr
    f0_all, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=original_sr,
        frame_length=fft_len_orig,
        hop_length=hop_length_orig,
    )
    f0_times = librosa.frames_to_time(
        np.arange(len(f0_all)),
        sr=original_sr,
        hop_length=hop_length_orig,
    )

    # Global power (rms) at original_sr
    pwr_all = librosa.feature.rms(
        y=y,
        frame_length=fft_len_orig,
        hop_length=hop_length_orig,
    ).flatten()
    pwr_times = librosa.frames_to_time(
        np.arange(len(pwr_all)),
        sr=original_sr,
        hop_length=hop_length_orig,
    )

    # Precompute spectral features ONCE over the whole piece
    # Shapes:
    #   centroid/bandwidth/flatness/rolloff: (1, frames)
    #   contrast: (bands, frames)
    spec_centroid_all = librosa.feature.spectral_centroid(S=S)
    spec_bandwidth_all = librosa.feature.spectral_bandwidth(S=S)
    spec_contrast_all = librosa.feature.spectral_contrast(S=S)
    spec_flatness_all = librosa.feature.spectral_flatness(S=S)
    spec_rolloff_all = librosa.feature.spectral_rolloff(S=S)

    def slice_indices(times, onset, offset):
        i0 = np.searchsorted(times, onset, side="left")
        i1 = np.searchsorted(times, offset, side="right")
        if i1 <= i0:
            return None
        return i0, i1

    def slice_indices_vec(times, onsets, offsets):
        i0 = np.searchsorted(times, onsets, side="left")
        i1 = np.searchsorted(times, offsets, side="right")
        ok = i1 > i0
        return i0, i1, ok

    def slice_spec_dict(i0, i1):
        return {
            "spec_centroid": spec_centroid_all[..., i0:i1],
            "spec_bandwidth": spec_bandwidth_all[..., i0:i1],
            "spec_contrast": spec_contrast_all[..., i0:i1],
            "spec_flatness": spec_flatness_all[..., i0:i1],
            "spec_rolloff": spec_rolloff_all[..., i0:i1],
        }

    # is_single_part: true when there is only one instrument/voice part
    is_monophonic = len(nmat) == 1

    freqs_rs = times_rs = D_rs = None
    if not is_monophonic:
        freqs_rs, times_rs, D_rs = librosa.reassigned_spectrogram(
            y=y,
            sr=original_sr,
            hop_length=hop_length_orig,
        )

    for key, df in nmat.items():
        onsets = df["ONSET_SEC"].to_numpy(dtype=float)
        offsets = df["OFFSET_SEC"].to_numpy(dtype=float)
        midis = df["MIDI"].to_numpy(dtype=float)
        ids = df.index.to_list()

        note_vals = []
        note_ids = []

        # Vectorized index lookup (cuts python overhead)
        f0_i0, f0_i1, f0_ok = slice_indices_vec(f0_times, onsets, offsets)
        pwr_i0, pwr_i1, pwr_ok = slice_indices_vec(pwr_times, onsets, offsets)
        spec_i0, spec_i1, spec_ok = slice_indices_vec(frame_times, onsets, offsets)

        for i in range(len(df)):
            if not (f0_ok[i] and pwr_ok[i] and spec_ok[i]):
                note_vals.append(_nan_note())
                note_ids.append(ids[i])
                continue

            onset = onsets[i]
            offset = offsets[i]
            midi = midis[i]

            f0_seg = f0_all[f0_i0[i] : f0_i1[i]]
            pwr_seg = pwr_all[pwr_i0[i] : pwr_i1[i]]
            M_dict = slice_spec_dict(spec_i0[i], spec_i1[i])

            if is_monophonic or force_pyin == True:
                note_vals.append(
                    estimate_perceptual_parameters(
                        f0_seg,
                        pwr_seg,
                        M_dict,
                        original_sr,
                        hop_length_orig,
                        1,
                    )
                )
            else:
                # Try the pitch-separated path first; fall back to the global
                # pyin f0/rms segments when f0_est_weighted_sum_spec fails
                # (it throws on most notes, leaving descriptors all-NaN).
                try:
                    f0, pwr, t, _M_unused, xf = f0_est_weighted_sum_spec(
                        onset,
                        offset,
                        midi,
                        freqs_rs,
                        D_rs,
                        original_sr,
                    )
                    note_vals.append(
                        estimate_perceptual_parameters(
                            f0,
                            pwr,
                            M_dict,
                            original_sr,
                            hop_length_orig,
                            1,
                        )
                    )
                except Exception:
                    # Fall back to global pyin f0 + rms power for this note
                    note_vals.append(
                        estimate_perceptual_parameters(
                            f0_seg,
                            pwr_seg,
                            M_dict,
                            original_sr,
                            hop_length_orig,
                            1,
                        )
                    )

            note_ids.append(ids[i])

        all_note_vals.append(note_vals)
        all_note_ids.append(note_ids)

    loc = 0
    for key, df in nmat.items():
        vals = all_note_vals[loc]

        df["f0Vals"] = [v["f0_vals"] for v in vals]
        df["meanf0"] = [_safe_nanmean(v) for v in df["f0Vals"]]

        df["ppitch1"] = [v["ppitch"][0] for v in vals]
        df["ppitch2"] = [v["ppitch"][1] for v in vals]
        df["jitter"] = [v["jitter"] for v in vals]

        df["vibratoDepth"] = [v["vibrato_depth"] for v in vals]
        df["vibratoRate"] = [v["vibrato_rate"] for v in vals]

        df["pwrVals"] = [v["pwr_vals"] for v in vals]
        df["meanPwr"] = [_safe_nanmean(v) for v in df["pwrVals"]]
        df["shimmer"] = [v["shimmer"] for v in vals]

        df["specCentVals"] = [v["spec_centroid"] for v in vals]
        df["meanSpecCent"] = [_safe_nanmean(v) for v in df["specCentVals"]]

        df["specBandwidthVals"] = [v["spec_bandwidth"] for v in vals]
        df["meanSpecBandwidth"] = [_safe_nanmean(v) for v in df["specBandwidthVals"]]

        df["specContrastVals"] = [v["spec_contrast"] for v in vals]
        df["meanSpecContrast"] = [_safe_nanmean(v) for v in df["specContrastVals"]]

        df["specFlatnessVals"] = [v["spec_flatness"] for v in vals]
        df["meanSpecFlatness"] = [_safe_nanmean(v) for v in df["specFlatnessVals"]]

        df["specRolloffVals"] = [v["spec_rolloff"] for v in vals]
        df["meanSpecRolloff"] = [_safe_nanmean(v) for v in df["specRolloffVals"]]

        loc += 1

    nmat_export = _convert_nmat_for_export(nmat)
    ext = getattr(piece, "fileExtension", None)

    if ext == "krn":
        # Scalar descriptor columns to export as kern analysis spines.
        # ONSET_SEC / OFFSET_SEC give each note its audio-aligned timing.
        ANALYSIS_COLS = [
            'ONSET_SEC', 'OFFSET_SEC',
            'meanf0', 'ppitch1', 'ppitch2', 'jitter',
            'vibratoDepth', 'vibratoRate',
            'meanPwr', 'shimmer',
            'meanSpecCent', 'meanSpecBandwidth', 'meanSpecContrast',
            'meanSpecFlatness', 'meanSpecRolloff',
        ]

        # Check for a **harte spine imported from the source kern file.
        harte_raw = piece._analyses.get('harte', None)
        if isinstance(harte_raw, list) and len(harte_raw):
            harte_raw = harte_raw[0]
        has_harte = (
            harte_raw is not None
            and isinstance(harte_raw, pd.Series)
            and not harte_raw.empty
        )

        # Build a forward-filled harte lookup dict once (beat_offset → label).
        if has_harte:
            harte_clean = harte_raw.sort_index()
            harte_clean = harte_clean[~harte_clean.index.duplicated(keep='last')].ffill()

        # Check for a **harm spine (roman numeral analysis) in the source kern file.
        harm_raw = piece._analyses.get('harm', None)
        if isinstance(harm_raw, list) and len(harm_raw):
            harm_raw = harm_raw[0]
        has_harm = (
            harm_raw is not None
            and isinstance(harm_raw, pd.Series)
            and not harm_raw.empty
        )

        # Build a forward-filled harm lookup dict once (beat_offset → label).
        if has_harm:
            harm_clean = harm_raw.sort_index()
            harm_clean = harm_clean[~harm_clean.index.duplicated(keep='last')].ffill()

        part_dfs = []
        for part, df_part in nmat.items():
            available = [c for c in ANALYSIS_COLS if c in df_part.columns]
            dfc = df_part[available].copy()

            # Attach harm (roman numeral) labels by beat offset — prepend so
            # it appears as the first analysis spine in the output kern file.
            if has_harm and 'ONSET' in df_part.columns:
                unique_onsets = pd.Index(df_part['ONSET'].unique())
                combined_idx = harm_clean.index.union(unique_onsets)
                harm_lookup = harm_clean.reindex(combined_idx).ffill()
                onset_to_harm = harm_lookup.to_dict()
                dfc['harm'] = df_part['ONSET'].map(onset_to_harm).values
                dfc = dfc[['harm'] + [c for c in dfc.columns if c != 'harm']]

            # Attach harte labels by beat offset
            if has_harte and 'ONSET' in df_part.columns:
                unique_onsets = pd.Index(df_part['ONSET'].unique())
                combined_idx = harte_clean.index.union(unique_onsets)
                harte_lookup = harte_clean.reindex(combined_idx).ffill()
                onset_to_harte = harte_lookup.to_dict()
                dfc['harte'] = df_part['ONSET'].map(onset_to_harte).values
                # Keep harm first if both spines are present
                front_cols = [c for c in ['harm', 'harte'] if c in dfc.columns]
                dfc = dfc[front_cols + [c for c in dfc.columns if c not in front_cols]]

            # Re-index from XML_ID → global beat-offset
            if 'ONSET' in df_part.columns:
                dfc.index = df_part['ONSET'].values

            # Chord notes share an onset; keep last (highest MIDI = melody note)
            dfc = dfc[~dfc.index.duplicated(keep='last')]
            part_dfs.append(dfc)

        # Merge all parts: union of all beat offsets, first non-NaN wins per cell
        if part_dfs:
            combined = part_dfs[0]
            for other in part_dfs[1:]:
                combined = combined.combine_first(other)
            combined = combined.sort_index()
        else:
            combined = pd.DataFrame()

        toKern(
            piece,
            path_name=f"{output_path}.krn",
            include_lyrics=False,
            include_dynamics=False,
            analysis_dfs={'analysis': combined} if not combined.empty else None,
        )
        fileOutput = f"{output_path}.krn"
    elif ext in ("csv", "txt"):
        # CSV scores (speech/music Tony CSV) — skip MEI entirely, write nmat directly
        csv_path = f"{output_path}.csv"
        frames = []
        for part_name, df_part in nmat.items():
            frames.append(df_part)
        if frames:
            pd.concat(frames).to_csv(csv_path)
        fileOutput = csv_path

    else:
        mei_path = f"{output_path}.mei"
        insertAudioAnalysis(
            piece,
            mei_path,
            nmat_export,
            mimetype="audio/aiff",
            target=audio_file_path,
        )
        fileOutput = mei_path

        # Write a CSV alongside MEI output for non-krn, non-csv formats
        csv_path = f"{output_path}.csv"
        _mei_to_csv(mei_path, csv_path)

    return nmat, fileOutput    

# Utility functions for data processing, no documentation
def _safe_nanmean(x):
    if x is None:
        return np.nan

    x = np.asarray(x)

    # Empty array
    if x.size == 0:
        return np.nan

    # All values are NaN
    if np.isnan(x).all():
        return np.nan

    return np.nanmean(x)

def _nan_note():
    return {
        "f0_vals": np.nan,
        "ppitch": (np.nan, np.nan),
        "jitter": np.nan,
        "vibrato_depth": np.nan,
        "vibrato_rate": np.nan,
        "pwr_vals": np.nan,
        "shimmer": np.nan,
        "spec_centroid": np.nan,
        "spec_bandwidth": np.nan,
        "spec_contrast": np.nan,
        "spec_flatness": np.nan,
        "spec_rolloff": np.nan,
    }


def _convert_nmat_for_export(nmat):
    list_columns = [
        "f0Vals",
        "pwrVals",
        "specCentVals",
        "specBandwidthVals",
        "specContrastVals",
        "specFlatnessVals",
        "specRolloffVals",
    ]
    out = {}
    for part, df_part in nmat.items():
        dfc = df_part.copy()
        for col in list_columns:
            if col in dfc.columns:
                dfc[col] = dfc[col].astype(str)
        out[part] = dfc
    return out


# ---------- MEI → CSV helpers ----------

_MEI_NS = {"mei": "http://www.music-encoding.org/ns/mei"}

_PITCH_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

_CSV_EXPORT_COLS = [
    "xmlid",
    "MIDI",
    "pitch_from_midi",
    "ONSET_SEC",
    "OFFSET_SEC",
    "meanf0",
    "ppitch1",
    "ppitch2",
    "jitter",
    "vibratoDepth",
    "vibratoRate",
    "meanPwr",
    "shimmer",
    "meanSpecCent",
    "meanSpecBandwidth",
    "meanSpecContrast",
    "meanSpecFlatness",
    "meanSpecRolloff",
]


def _midi_to_pitch(midi):
    import math
    if midi is None or (isinstance(midi, float) and math.isnan(midi)):
        return ""
    midi = int(round(midi))
    octave = midi // 12 - 1
    return f"{_PITCH_NAMES[midi % 12]}{octave}"


def _extract_notes_from_mei(mei_path: str) -> "pd.DataFrame":
    import xml.etree.ElementTree as ET
    import json

    tree = ET.parse(mei_path)
    root = tree.getroot()
    rows = []
    for when in root.findall(".//mei:performance//mei:when", _MEI_NS):
        ext = when.find("mei:extData", _MEI_NS)
        if ext is None or ext.text is None:
            continue
        text = ext.text.strip()
        if text.startswith("<![CDATA["):
            text = text.replace("<![CDATA[", "").replace("]]>", "").strip()
        try:
            data = json.loads(text)
        except Exception:
            continue
        row = dict(data)
        row["xmlid"] = when.attrib.get("{http://www.w3.org/XML/1998/namespace}id")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _mei_to_csv(mei_path: str, csv_path: str) -> None:
    """Extract note-level descriptors from a pyAMPACT MEI file and write a CSV."""
    df = _extract_notes_from_mei(mei_path)
    if df.empty:
        print(f"  mei_to_csv: no extData found in {mei_path}, CSV not written.")
        return

    # Ensure all expected columns exist (fill missing with NA)
    for col in _CSV_EXPORT_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df["pitch_from_midi"] = df["MIDI"].apply(_midi_to_pitch)

    df_out = df[_CSV_EXPORT_COLS].copy()
    df_out = df_out.dropna(axis=1, how="all")
    df_out.to_csv(csv_path, index=False)


def export_selected_columns(nmat, columns, audio_file_path=None, output_path=None):
    """
    Export a user-defined subset of descriptor columns from a note matrix to CSV.

    Parameters
    ----------
    nmat : pd.DataFrame
        Note matrix containing selected olumns.

    columns : list of str
        Column names to include in the exported CSV. 

    audio_file_path : str, optional
        Path to the source audio file. When provided, the output is written to
        ``output_files/output_<stem>/<stem>_selected.csv`` alongside the other
        pyAMPACT output files for that recording.

    output_path : str, optional
        Explicit destination path for the CSV file. Takes precedence over the
        auto-derived path when both ``audio_file_path`` and ``output_path`` are
        given. If neither is provided, defaults to ``./output_selected_data.csv``.

    Returns
    -------
    None

    """
    if audio_file_path is not None:
        audio_stem = os.path.splitext(os.path.basename(audio_file_path))[0]
        output_dir = os.path.join("output_files", f"output_{audio_stem}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_path or os.path.join(output_dir, f"{audio_stem}_selected.csv")
    else:
        output_path = output_path or "./output_selected_data.csv"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    selected = []

    if isinstance(nmat, dict):
        dfs = nmat.values()
    elif isinstance(nmat, pd.DataFrame):
        dfs = [nmat]
    elif isinstance(nmat, np.ndarray) and all(isinstance(x, pd.DataFrame) for x in nmat):
        dfs = list(nmat)
    else:
        raise TypeError(f"Unsupported type for nmat: {type(nmat)}")

    for df in dfs:
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                df[col] = pd.NA
        selected.append(df[columns])

    if not selected:
        return

    combined = pd.concat(selected, ignore_index=True)
    combined.to_csv(output_path, index=False)


def midi_to_freq(midi):
    return 440.0 * (2 ** ((midi - 69) / 12))


enharmonic_map = {
    "A#": "Bb",
    "C#": "Db",
    "D#": "Eb",
    "F#": "Gb",
    "G#": "Ab",
}


def visualise_alignment_from_nmat(
    nmat_dict,
    y,
    original_sr,
    target_sr,
    hop_length,
    winms,
    audio_file_path,
):
    # --- Derive output folder from the audio filename ---
    audio_stem = os.path.splitext(os.path.basename(audio_file_path))[0]
    output_dir = os.path.join("output_files", f"output_{audio_stem}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, audio_stem)

    # --- FFT length computed at ORIGINAL sample rate ---
    win_sec = winms / 1000.0
    fft_len = int(2 ** np.round(np.log(win_sec * original_sr) / np.log(2)))
    fft_len = max(256, fft_len)

    # --- hop is already in samples at original_sr ---
    hop_sec = hop_length / target_sr
    hop_samp = int(hop_sec * original_sr)

    noverlap = fft_len - hop_samp

    freqs, times, Zxx = signal.stft(
        y,
        fs=original_sr,
        window="hamming",
        nperseg=fft_len,
        noverlap=noverlap,
        nfft=fft_len,
        boundary=None,
        padded=False,
    )

    D = np.abs(Zxx)
    D /= (D.max() if D.max() > 0 else 1.0)

    S = librosa.amplitude_to_db(D, ref=np.max)

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        S,
        x_coords=times,
        y_coords=freqs,
        x_axis="time",
        y_axis="log",
        cmap="gray_r",
    )


    plt.title(f"Spectrogram + Notes: {output_path}")

    base_colors = ["red","blue","green","orange","purple","cyan","magenta","yellow"]
    color_cycle = (base_colors * ((len(nmat_dict) // len(base_colors)) + 1))[:len(nmat_dict)]
    legend_elements = []

    for idx, (_, notes) in enumerate(nmat_dict.items()):
        if notes.empty:
            continue
        color = color_cycle[idx]
        legend_elements.append(Patch(facecolor=color, label=f"Part-{idx+1}"))

        for _, row in notes.iterrows():
            if any(k not in row.index for k in ("MIDI","ONSET_SEC","OFFSET_SEC")):
                continue
            freq = midi_to_freq(row["MIDI"])
            start = row["ONSET_SEC"]
            end = row["OFFSET_SEC"]
            plt.fill_between([start, end], freq - 15, freq + 15, color=color, alpha=0.4)

    plt.ylim(20, original_sr / 2)
    plt.xlim(0, times[-1])
    plt.colorbar(format="%+2.0f dB")
    plt.legend(handles=legend_elements, loc="upper right")
    plt.savefig(f"{output_path}.png", dpi=300)
    plt.close()


def plot_piano_roll(
    piece,
    nmat,
    audio_file_path,
    target_sr,
    hop_length,
    verbose=False
):
    """
    Build a piano-roll image from aligned note data and save it alongside the
    other output files for this audio file.

    The piano roll is painted in audio-seconds (x-axis) vs MIDI pitch (y-axis).
    Spine annotations present in the score (keys, harm, chord/harte, function)
    are extracted, remapped from quarter-note offsets to audio seconds, and
    printed to stdout for inspection — exactly as the original exampleScript did.

    Parameters
    ----------
    piece : Score
        The Score object returned by run_alignment / load_score.
    nmat : dict
        The aligned note-matrix dict returned by data_compilation.
    audio_file_path : str
        Path to the original audio file — used to derive the output folder name.
    target_sr : int
        Target sample rate used during alignment (e.g. 4000).
    hop_length : int
        Hop size in samples at target_sr used during alignment (e.g. 32).
    verbose : boolean
        If true then print the piano roll specs

    Returns
    -------
    pr : ndarray  (128 × n_cols, float32)
        The raw piano-roll matrix, in case the caller wants to post-process it.
    audio_axis : pd.Index
        The time axis (seconds) corresponding to the columns of pr.
    """
    # ── Derive output folder ──────────────────────────────────────────────────
    audio_stem = os.path.splitext(os.path.basename(audio_file_path))[0]
    output_dir = os.path.join("output_files", f"output_{audio_stem}")
    os.makedirs(output_dir, exist_ok=True)

    # ── Flatten all parts of nmat into one DataFrame ──────────────────────────
    nmat_symbolic = nmats(piece)
    all_notes = pd.concat(nmat_symbolic.values()).reset_index(drop=True)
    all_notes = all_notes.dropna(subset=["MIDI", "ONSET_SEC", "OFFSET_SEC"])
    all_notes = all_notes[all_notes["MIDI"] >= 0]

    if all_notes.empty:
        print("plot_piano_roll: no valid notes found, skipping.")
        return None, None

    # ── Audio time axis ───────────────────────────────────────────────────────
    audio_start = all_notes["ONSET_SEC"].min()
    audio_end   = all_notes["OFFSET_SEC"].max()
    n_cols      = int((audio_end - audio_start) * target_sr / hop_length) + 1
    col_times   = audio_start + np.arange(n_cols) * (hop_length / target_sr)
    audio_axis  = pd.Index(col_times)

    # ── QN → audio-seconds mapping (from aligned nmat) ───────────────────────
    qn_sec = (
        all_notes[["ONSET", "ONSET_SEC"]]
        .dropna()
        .drop_duplicates("ONSET")
        .sort_values("ONSET")
    )
    qn_pts  = qn_sec["ONSET"].values
    sec_pts = qn_sec["ONSET_SEC"].values

    def qn_to_sec(qn_arr):
        return np.interp(np.asarray(qn_arr, dtype=float), qn_pts, sec_pts)

    # ── Piano roll matrix (128 × n_cols) ─────────────────────────────────────
    pr = np.zeros((128, n_cols), dtype=np.float32)
    for _, row in all_notes.iterrows():
        midi = int(row["MIDI"])
        i0   = int(np.searchsorted(col_times, float(row["ONSET_SEC"])))
        i1   = int(np.searchsorted(col_times, float(row["OFFSET_SEC"])))
        i1   = max(i1, i0 + 1)
        pr[midi, max(i0, 0):min(i1, n_cols)] = 1.0

    # ── Convert spine annotations: QN offsets → audio seconds → audio_axis ───
    def spine_to_audio(raw):
        ser = raw[0].copy() if isinstance(raw, list) else raw.copy()
        if ser.empty:
            return pd.Series(dtype=object, index=audio_axis)
        # Strip humdrum dot continuation tokens
        ser = ser[ser != "."].dropna()
        if ser.empty:
            return pd.Series(dtype=object, index=audio_axis)
        # Remap QN index to audio seconds
        ser.index = pd.Index(qn_to_sec(ser.index.astype(float)))
        ser = ser[~ser.index.duplicated(keep="last")].sort_index()
        # Insert into NaN series covering full audio_axis, then ffill
        target   = pd.Series(np.nan, index=audio_axis, dtype=object)
        combined = ser.combine_first(target).sort_index().ffill()
        return combined.reindex(audio_axis)

    keys_audio      = spine_to_audio(piece._analyses.get("keys",     pd.Series(dtype=object)))
    harm_audio      = spine_to_audio(piece._analyses.get("harm",     []))
    chords_audio    = spine_to_audio(piece._analyses.get("chord",    piece._analyses.get("harte", [])))
    functions_audio = spine_to_audio(piece._analyses.get("function", []))

    if verbose==True:
        print("keys:");        print(keys_audio)
        print("\nharm:");      print(harm_audio)
        print("\nchords:");    print(chords_audio)
        print("\nfunctions:"); print(functions_audio)

    # ── Plot ──────────────────────────────────────────────────────────────────
    title = getattr(piece, "fileName", audio_stem)
    t_min = float(audio_axis[0])
    t_max = float(audio_axis[-1])

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#fafaf7")
    ax.imshow(
        pr,
        aspect="auto",
        origin="lower",
        extent=[t_min, t_max, 0, 128],
        cmap="Blues",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax.set_xlim(t_min, t_max)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.set_xlabel("Audio time  (seconds)", fontsize=9, color="#555")
    ax.set_ylabel("MIDI pitch", fontsize=9, color="#555")
    ax.set_title(
        f"pyAMPACT  \u2014  {title}",
        fontsize=11, fontweight="bold", pad=8, color="#1a1a1a",
    )
    ax.tick_params(labelsize=8, color="#aaa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#ddd")
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{audio_stem}_piano_roll.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return pr, audio_axis

