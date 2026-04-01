# import numpy as np
# import parselmouth

# def iaif_glottal_flow(seg_sound: parselmouth.Sound,
#                       vt_order: int = None,
#                       g_order: int = 3,
#                       time_step: float = 0.01,
#                       window_length: float = 0.03,
#                       preemph_from_hz: float = 50.0):
#     """
#     IAIF-lite using Praat ops via Parselmouth.
#     Returns (gflow, gflow_deriv) as parselmouth.Sound objects.

#     seg_sound: a parselmouth.Sound of the segment
#     vt_order: LPC order for vocal-tract (default ~= 2*Fs/1000 + 4)
#     g_order:  LPC order for glottal shaping (low, usually 2–4)
#     """

#     sr = seg_sound.sampling_frequency
#     if vt_order is None:
#         vt_order = int(round(2.0 * (sr / 1000.0) + 4))  # decent default for adult speech

#     # 0) Pre-emphasis (undo lip radiation)
#     s_pre = parselmouth.praat.call(seg_sound, "Filter (pre-emphasis)...", preemph_from_hz)

#     # 1) First-pass VT estimation on pre-emphasized speech
#     lpc_vt1 = parselmouth.praat.call(s_pre, "To LPC (burg)...",
#                                      time_step, window_length, vt_order, preemph_from_hz)

#     # 2) Inverse filter → coarse glottal flow derivative (Ug')
#     ugd1 = parselmouth.praat.call([s_pre, lpc_vt1], "Filter (inverse)")

#     # 3) Optional low-order glottal LPC to flatten LF tilt (simple IAIF step)
#     #    Estimate on ugd1, then inverse-filter ugd1 with low-order model to reduce source coloration
#     lpc_g = parselmouth.praat.call(ugd1, "To LPC (burg)...",
#                                    time_step, window_length, max(1, g_order), preemph_from_hz)
#     ugd2 = parselmouth.praat.call([ugd1, lpc_g], "Filter (inverse)")

#     # 4) Second-pass VT estimation on re-colored speech: filter original pre-emphasized speech
#     lpc_vt2 = parselmouth.praat.call(s_pre, "To LPC (burg)...",
#                                      time_step, window_length, vt_order, preemph_from_hz)
#     ugd = parselmouth.praat.call([s_pre, lpc_vt2], "Filter (inverse)")  # final Ug' estimate

#     # 5) De-emphasis (restore low end a bit; optional)
#     ugd = parselmouth.praat.call(ugd, "Filter (de-emphasis)...", preemph_from_hz)

#     # 6) Integrate Ug' → Ug (glottal flow). Do this in Python for precision.
#     y = ugd.values[0].astype(float)
#     T = 1.0 / sr
#     ug = np.cumsum(y) * T  # simple rectangle integration; ok for quotient metrics

#     # remove DC drift from integration
#     ug = ug - np.mean(ug)

#     gflow = parselmouth.Sound(ug, sampling_frequency=sr)
#     gflow_deriv = ugd
#     return gflow, gflow_deriv
