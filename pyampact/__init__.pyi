from .version import show_versions as show_versions

from .alignment import (
    run_alignment as run_alignment,
    run_DTW_alignment as run_DTW_alignment,
    align_midi_wav as align_midi_wav,
)

from .alignmentUtils import (
    dp as dp,
    gh as gh,
    g as g,
    orio_simmx as orio_simmx,
    maptimes as maptimes,
    f0_est_weighted_sum as f0_est_weighted_sum,
    f0_est_weighted_sum_spec as f0_est_weighted_sum_spec,
    trim_silences as trim_silences,
    merge_grace_notes as merge_grace_notes,
)

from .dataCompilation import (
    data_compilation as data_compilation,
    export_selected_columns as export_selected_columns,
    visualise_alignment_from_nmat as visualise_alignment_from_nmat,
    plot_piano_roll as plot_piano_roll,
)

from .performance import (
    estimate_perceptual_parameters as estimate_perceptual_parameters,
    calculate_vibrato as calculate_vibrato,
    perceived_pitch as perceived_pitch,
)

from .speechDescriptors import (
    get_speech_descriptors as get_speech_descriptors,
)

from .speechDescriptorsUtils import (
    compute_correlation_dimension as compute_correlation_dimension,
    compute_dfa as compute_dfa,
    compute_emd_features as compute_emd_features,
    compute_mfcc as compute_mfcc,
    compute_ppe as compute_ppe,
    compute_recurrence_period_and_rpde as compute_recurrence_period_and_rpde,
    compute_spectral_spread as compute_spectral_spread,
    compute_tqwt_features as compute_tqwt_features,
    compute_wt_features as compute_wt_features,
    feature_spectral_flux as feature_spectral_flux,
    compute_vot as compute_vot,
    compute_bark_band_energies as compute_bark_band_energies,
    compute_dpi as compute_dpi,
    compute_avqi as compute_avqi,
)

from .symbolic import (
    load_score as load_score,
    convert_attribs_to_str as convert_attribs_to_str,
    _assignM21Attributes as _assignM21Attributes,
    _import_other_spines as _import_other_spines,
    _partList as _partList,
    _parts as _parts,
    insertScoreDef as insertScoreDef,
    xmlIDs as xmlIDs,
    _lyricHelper as _lyricHelper,
    lyrics as lyrics,
    _m21Clefs as _m21Clefs,
    _clefs as _clefs,
    dynamics as dynamics,
    _priority as _priority,
    keys as keys,
    harm as harm,
    functions as functions,
    chords as chords,
    cdata as cdata,
    getSpines as getSpines,
    dez as dez,
    form as form,
    romanNumerals as romanNumerals,
    _m21ObjectsNoTies as _m21ObjectsNoTies,
    _measures as _measures,
    _barlines as _barlines,
    _keySignatures as _keySignatures,
    _timeSignatures as _timeSignatures,
    _beats as _beats,
    durations as durations,
    midi_ticks_durations as midi_ticks_durations,
    midiPitches as midiPitches,
    notes as notes,
    kernNotes as kernNotes,
    nmats as nmats,
    contextualize as contextualize,
    pianoRoll as pianoRoll,
    sampled as sampled,
    mask as mask,
    build_mask_from_nmat_seconds as build_mask_from_nmat_seconds,
    jsonCDATA as jsonCDATA,
    insertAudioAnalysis as insertAudioAnalysis,
    _meiStack as _meiStack,
    toKern as toKern,
    toMEI as toMEI,
    show as show,
)

from .symbolicUtils import (
    _escape_cdata as _escape_cdata,
    addMEINote as addMEINote,
    addTieBreakers as addTieBreakers,
    kernClefHelper as kernClefHelper,
    combineRests as combineRests,
    combineUnisons as combineUnisons,
    fromJSON as fromJSON,
    _id_gen as _id_gen,
    idGen as idGen,
    indentMEI as indentMEI,
    _kernChordHelper as _kernChordHelper,
    kernFooter as kernFooter,
    kernHeader as kernHeader,
    _kernNoteHelper as _kernNoteHelper,
    kernNRCHelper as kernNRCHelper,
    noteRestHelper as noteRestHelper,
    remove_namespaces as remove_namespaces,
    removeTied as removeTied,
    snapTo as snapTo,
    truncate_and_scale_onsOffsList as truncate_and_scale_onsOffsList,
    githubURLtoRaw as githubURLtoRaw,
    duration2MEI as duration2MEI,
    _duration2Kern as _duration2Kern,
    meiDeclaration as meiDeclaration,
    explode_kern_chords as explode_kern_chords,
)

