"""
symbolic
===================

.. autosummary::
   :toctree: generated/

   load_score
   convert_attribs_to_str
   _assignM21Attributes
   _import_other_spines
   _partList
   _parts
   insertScoreDef
   xmlIDs
   _lyricHelper
   lyrics
   _m21Clefs
   _clefs
   dynamics
   _priority
   keys
   harm
   functions
   chords
   cdata
   getSpines
   dez
   form
   romanNumerals
   _m21ObjectsNoTies
   _measures
   _barlines
   _keySignatures
   _timeSignatures
   _beats
   durations
   midi_ticks_durations
   midiPitches
   notes
   kernNotes
   nmats
   contextualize
   pianoRoll
   sampled
   mask
   jsonCDATA
   insertAudioAnalysis
   toKern
   toMEI
   show
   build_mask_from_nmat_seconds

"""

import sys, types, os, ast, base64, math, json, re, tempfile
from copy import deepcopy

import pandas as pd
import numpy as np
import librosa
import requests
import xml.etree.ElementTree as ET
import music21 as m21

from .symbolicUtils import *

m21.environment.set('autoDownload', 'allow')

imported_scores = {}
function_pattern = re.compile(r'^function\s*=\s*', re.IGNORECASE)
idx = pd.IndexSlice

__all__ = [
    "load_score",
    "convert_attribs_to_str",
    "_assignM21Attributes",
    "_import_other_spines",
    "_partList",
    "_parts",
    "insertScoreDef",
    "xmlIDs",
    "_lyricHelper",
    "lyrics",
    "_m21Clefs",
    "_clefs",
    "dynamics",
    "_priority",
    "keys",
    "harm",
    "functions",
    "chords",
    "cdata",
    "getSpines",
    "dez",
    "form",
    "romanNumerals",
    "_m21ObjectsNoTies",
    "_measures",
    "_barlines",
    "_keySignatures",
    "_timeSignatures",
    "_beats",
    "durations",
    "midi_ticks_durations",
    "midiPitches",
    "notes",
    "nmats",
    "contextualize",
    "pianoRoll",
    "sampled",
    "mask",
    "build_mask_from_nmat_seconds",
    "jsonCDATA",
    "insertAudioAnalysis",
    "_meiStack",
    "toKern",
    "kernNotes",
    "toMEI",
    "show",
]


def convert_attribs_to_str(element):
    for key in element.attrib:
        if isinstance(element.attrib[key], np.float64):
            element.attrib[key] = str(element.attrib[key])
    for child in element:
        convert_attribs_to_str(child)

def initialize_piece(piece):
    if not hasattr(piece, "_analyses"):
        piece._analyses = {}
    if not hasattr(piece, "_meiTree"):
        piece._meiTree = None
    if not hasattr(piece, "foundSpines"):
        piece.foundSpines = {}
    if not hasattr(piece, "partNames"):
        piece.partNames = [p.partName for p in piece.score.parts]
    if "_divisiStarts" not in piece._analyses:
        piece._analyses["_divisiStarts"] = pd.DataFrame()
    if "_divisiEnds" not in piece._analyses:
        piece._analyses["_divisiEnds"] = pd.DataFrame()
    return piece

def load_score(score_path):
    piece = types.SimpleNamespace()
    piece._analyses = {}
    if score_path.startswith('https://github.com/'):
        score_path = 'https://raw.githubusercontent.com/' + \
            score_path[19:].replace('/blob/', '/', 1)
    piece.path = score_path
    piece.fileName = score_path.rsplit('.', 1)[0].rsplit('/')[-1]
    piece.fileExtension = score_path.rsplit(
        '.', 1)[1] if '.' in score_path else ''
    piece.partNames = []
    piece.score = None
    piece._meiTree = None
    piece._kernSpineGroups = None  # set below for .krn files
    if piece.fileExtension == 'krn':
        # Explode chord-spines into monophonic spines so music21 sees one
        # Part per voice. Temp file only — nothing written to disk permanently.
        # spine_groups records the original grouping so toKern can reassemble.
        if score_path.startswith('http'):
            _krn_text = requests.get(piece.path).text
        else:
            with open(score_path, 'r') as _f:
                _krn_text = _f.read()
        _exploded, _spine_groups = explode_kern_chords(_krn_text)
        piece._kernSpineGroups = _spine_groups
        _fd, _tmp_path = tempfile.mkstemp(suffix='.krn')
        try:
            with os.fdopen(_fd, 'w') as _tmp:
                _tmp.write(_exploded)
            _assignM21Attributes(piece, _tmp_path)
            _import_other_spines(piece, _tmp_path)
        finally:
            os.remove(_tmp_path)
    # file is not an online kern file (can be either or neither but not both)
    elif piece.fileExtension == 'mid':
        # Convert MIDI to MusicXML
        with tempfile.NamedTemporaryFile(suffix=".musicxml.xml", delete=False) as tmp:
            try:                    
                score = m21.converter.parse(piece.path)
                score.write('musicxml', fp=tmp.name)
                piece.path = tmp.name
                piece.fileExtension = 'xml'
            except Exception as e:
                raise RuntimeError(f"Failed to convert MIDI to MusicXML: {e}")
        _assignM21Attributes(piece, )
        _import_other_spines(piece, )            
    else:            
        _assignM21Attributes(piece, )
        _import_other_spines(piece, )
    piece.public = '\n'.join([
        f'{prop.ljust(15)}{type(getattr(piece, prop))}'
        for prop in dir(piece) if not prop.startswith('_')
    ])
    _partList(piece, )
    piece = initialize_piece(piece)
    return piece


def _assignM21Attributes(piece, path=''):
    """
    Assign music21 attributes to a given object.

    :param obj: A music21 object.
    :return: None
    """
    if piece.path not in imported_scores:
        if path:   # parse humdrum files differently to extract their function, and harm spines if they have them
            imported_scores[piece.path] = m21.converter.parse(
                path, format='humdrum')
        # these files might be mei files and could lack elements music21 needs to be able to read them
        elif piece.fileExtension in ('xml', 'musicxml', 'mei', 'mxl'):
            if piece.path.startswith('http'):
                tree = ET.ElementTree(ET.fromstring(
                    requests.get(piece.path).text))
            else:
                tree = ET.parse(piece.path)
            remove_namespaces(tree)
            root = tree.getroot()
            hasFunctions = False
            _functions = root.findall('.//function')
            if len(_functions):
                hasFunctions = True

            # this is an mei file even if the fileExtension is .xml
            if root.tag.endswith('mei'):
                parseEdited = False
                piece._meiTree = deepcopy(root)
                # this mei file doesn't have a scoreDef element, so construct one and add it to the score
                if not root.find('.//scoreDef'):
                    parseEdited = True
                    insertScoreDef(piece, root)

                # make sure all events are contained in measures
                for section in root.iter('section'):
                    if section.find('measure') is None:
                        parseEdited = True
                        measure = ET.Element('measure')
                        measure.set('xml:id', next(idGen))
                        measure.extend(section)
                        section.clear()
                        section.append(measure)

                if parseEdited:
                    mei_string = ET.tostring(root, encoding='unicode')
                    imported_scores[piece.path] = m21.converter.subConverters.ConverterMEI(
                    ).parseData(mei_string)
                    parseEdited = False

            if hasFunctions:   # not an mei file, but an xml file that had functions
                try:
                    imported_scores[piece.path] = m21.converter.parse(
                        piece.path)
                except m21.harmony.HarmonyException:
                    print(
                        'There was an issue with the function texts so they were removed.')
                    for _function in _functions:
                        _function.text = ''
                    xml_string = ET.tostring(root, encoding='unicode')
                    imported_scores[piece.path] = m21.converter.parse(
                        xml_string, format='MusicXML')

        # read file/string as volpiano or tinyNotation if applicable
        elif piece.fileExtension in ('', 'txt'):
            if piece.path.startswith('http'):
                text = requests.get(piece.path).text
            else:
                with open(piece.path, 'r') as file:
                    text = file.read()
            rows = []
            for line in [l.strip() for l in text.strip().splitlines() if l.strip()]:
                cols = line.split()
                onset, offset, note_name = float(cols[0]), float(cols[1]), cols[2]
                duration = offset - onset
                try:
                    midi = int(round(m21.pitch.Pitch(note_name).midi))
                except Exception:
                    continue
                freq = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
                rows.append({'ONSET_SEC': onset, 'AVG PITCH IN HZ': freq,
                            'DURATION': duration, 'MIDI': midi})
            txt_df = pd.DataFrame(rows)
            piece._analyses['tony_csv'] = txt_df
            _p = pd.DataFrame(txt_df['MIDI'])
            _p.columns = ['Part-1']
            piece._analyses[('_parts', False, False, False, False)] = _p
            dur = pd.DataFrame(txt_df['DURATION'])
            dur.columns = ['Part-1']
            dur.index = pd.MultiIndex.from_tuples([(i, 0) for i in dur.index])
            piece._analyses[('durations', True)] = dur
            midiPitches = pd.DataFrame(txt_df['MIDI'])
            midiPitches.columns = ['Part-1']
            midiPitches.index = pd.MultiIndex.from_tuples([(i, 0) for i in midiPitches.index])
            piece._analyses[('midiPitches', True)] = midiPitches
            measures = pd.DataFrame([1])
            measures.columns = ['Part-1']
            piece._analyses[('_measures', False)] = measures
            xmlIDs = pd.DataFrame([next(idGen) for _ in range(len(txt_df))])
            xmlIDs.columns = ['Part-1']
            xmlIDs.index = pd.MultiIndex.from_tuples([(i, 0) for i in xmlIDs.index])
            piece._analyses['xmlIDs'] = xmlIDs
            piece.partNames = ['Part-1']
            piece._analyses['_partList'] = [None]
            piece.metadata = {'title': piece.fileName or 'Title not found',
                            'composer': 'Composer not found'}
            piece.score = None
            piece._flatParts = []
            return

        elif piece.fileExtension == 'csv':   # <-- CORRECT: top-level sibling elif
            _peek = pd.read_csv(piece.path, sep=None, engine='python', header=None, nrows=1)
            _has_header = pd.isna(pd.to_numeric(_peek.iloc[0, 0], errors='coerce'))
            _skip = 1 if _has_header else None
            csv_df = pd.read_csv(piece.path, sep=None, engine='python', header=None, index_col=False, skiprows=_skip)
            if len(csv_df.columns) == 4:
                csv_df.columns = ['ONSET_SEC', 'AVG PITCH IN HZ', 'DURATION', 'WORD']
            else:
                csv_df.columns = ['ONSET_SEC', 'AVG PITCH IN HZ', 'DURATION']
            csv_df['AVG PITCH IN HZ'] = pd.to_numeric(csv_df['AVG PITCH IN HZ'], errors='coerce')
            csv_df['MIDI'] = csv_df['AVG PITCH IN HZ'].map(
                lambda freq: librosa.hz_to_midi(freq)).round().astype('Int16')
            piece._analyses['tony_csv'] = csv_df
            _parts = pd.DataFrame(csv_df['MIDI'])
            _parts.columns = ['Part-1']
            piece._analyses[('_parts', False, False, False, False)] = _parts
            dur = pd.DataFrame(csv_df['DURATION'])
            dur.columns = ['Part-1']
            dur.index = pd.MultiIndex.from_tuples([(i, 0) for i in dur.index])
            piece._analyses[('durations', True)] = dur
            midiPitches = pd.DataFrame(csv_df['MIDI'])
            midiPitches.columns = ['Part-1']
            midiPitches.index = pd.MultiIndex.from_tuples([(i, 0) for i in midiPitches.index])
            piece._analyses[('midiPitches', True)] = midiPitches
            measures = pd.DataFrame([1])
            measures.columns = ['Part-1']
            piece._analyses[('_measures', False)] = measures
            xmlIDs = pd.DataFrame([next(idGen) for x in range(len(csv_df.index))])
            xmlIDs.columns = ['Part-1']
            xmlIDs.index = pd.MultiIndex.from_tuples([(i, 0) for i in xmlIDs.index])
            piece._analyses['xmlIDs'] = xmlIDs
            piece.partNames = ['Part-1']
            piece._analyses['_partList'] = [None]
            piece.metadata = {
                'title': piece.fileName or 'Title not found',
                'composer': 'Composer not found',
            }
            piece.score = None
            piece._flatParts = []
            return
    if piece.path not in imported_scores and piece.fileExtension != 'csv':   # check again to catch valid tree files
        if piece.path.startswith('http') and piece.fileExtension in ('mid', 'midi'):
            midi_bytes = requests.get(piece.path).content
            imported_scores[piece.path] = m21.converter.parse(midi_bytes)
        else:
            imported_scores[piece.path] = m21.converter.parse(piece.path)
    piece.score = imported_scores[piece.path]
    piece.metadata = {'title': "Title not found",
                     'composer': "Composer not found"}
    if piece.score.metadata is not None:
        piece.metadata['title'] = piece.score.metadata.title or 'Title not found'
        piece.metadata['composer'] = piece.score.metadata.composer or 'Composer not found'
    piece._partStreams = piece.score.getElementsByClass(m21.stream.Part)
    piece._flatParts = []
    piece.partNames = []
    for i, part in enumerate(piece._partStreams):
        flat = part.flatten()
        toRemove = [el for el in flat if el.offset < 0]
        flat.remove(toRemove)
        flat.makeMeasures(inPlace=True)
        flat.makeAccidentals(inPlace=True)
        # you have to flatten again after calling makeMeasures
        piece._flatParts.append(flat.flatten())
        name = flat.partName if (
            flat.partName and flat.partName not in piece.partNames) else f'Part-{i + 1}'
        piece.partNames.append(name)

def _partList(piece):
    """
    Return a list of series of the note, rest, and chord objects in each part.

    :return: A list of pandas Series, each representing a part in the score.
    """
    if '_partList' not in piece._analyses:            
        kernStrands = []
        parts = []
        isUnique = True
        divisiStarts = []
        divisiEnds = []
        for ii, flat_part in enumerate(piece._flatParts):                
            graces, graceOffsets = [], []
            notGraces = {}
            for nrc in flat_part.getElementsByClass(['Note', 'Rest', 'Chord']):                    
                if nrc.duration.isGrace:                                                                   
                    graceOffsets.append(round(float(nrc.offset), 5))
                else:
                    # get rid of really long rests TODO: make this get rid of rests longer than the prevailing measure
                    if (nrc.isRest and nrc.quarterLength > 18):
                        continue
                    offset = round(float(nrc.offset), 5)
                    if offset in notGraces:
                        notGraces[offset].append(nrc)
                    else:
                        notGraces[offset] = [nrc]

            ser = pd.Series(notGraces)
            if ser.empty:   # no note, rest, or chord objects detected in this part                                        
                ser.name = piece.partNames[ii]
                parts.append(ser)
                continue
            # make each cell a row resulting in a df where each col is a separate synthetic voice
            df = ser.apply(pd.Series)
            # swap elements in cols at this offset until all of them fill the space left before the next note in each col
            if len(df.columns > 1):
                for jj, ndx in enumerate(df.index):
                    # calculate dur inside the loop to avoid having to swap its elements like we do for df
                    dur = df.map(lambda cell: round(
                        float(cell.quarterLength), 5), na_action='ignore')
                    for thisCol in range(len(df.columns) - 1):
                        if isinstance(df.iat[jj, thisCol], float):  # ignore NaNs
                            continue
                        thisDur = dur.iat[jj, thisCol]
                        thisNextNdx = df.iloc[jj+1:, thisCol].first_valid_index(
                        ) or piece.score.highestTime
                        thisPrevNdx = df.iloc[:jj,
                                              thisCol].last_valid_index() or 0
                        if thisPrevNdx > 0:
                            thisPrevDur = dur[thisCol].at[thisPrevNdx]
                            # current note happens before previous note ended so swap for a NaN if there is one
                            if thisPrevNdx + thisPrevDur - ndx > .00003:
                                for otherCol in range(thisCol + 1, len(df.columns)):
                                    if isinstance(df.iat[jj, otherCol], float):
                                        df.iloc[jj, [thisCol, otherCol]] = df.iloc[jj, [
                                            otherCol, thisCol]]
                                        break
                        # this nrc takes up the amount of time expected in this col so no need to swap
                        if abs(thisNextNdx - ndx - thisDur) < .00003:
                            continue
                        # look for an nrc in another col with the duration thisCol needs
                        for otherCol in range(thisCol + 1, len(df.columns)):
                            # once we get a nan there's no hope of finding a valid swap at this index
                            if isinstance(df.iat[jj, otherCol], float):
                                break
                            otherDur = dur.iat[jj, otherCol]
                            if abs(thisNextNdx - ndx - otherDur) < .00003:  # found a valid swap
                                df.iloc[jj, [thisCol, otherCol]
                                        ] = df.iloc[jj, [otherCol, thisCol]]
                                break

            if len(graces):  # add all the grace notes found to col0                    
                part0 = pd.concat((pd.Series(
                    graces, graceOffsets), df.iloc[:, 0].dropna())).sort_index(kind='mergesort')
                isUnique = False
            else:
                part0 = df.iloc[:, 0].dropna()
            part0.name = piece.partNames[ii]
            parts.append(part0)
            kernStrands.append(part0)                

            strands = []
            # if df has more than 1 column, iterate over the non-first columns
            for col in range(1, len(df.columns)):
                part = df.iloc[:, col].dropna()
                _copy = part.copy()
                _copy.name = f'{part0.name}_{col}'
                parts.append(_copy)
                dur = part.apply(lambda nrc: nrc.quarterLength).astype(
                    float).round(5)
                prevEnds = (dur + dur.index).shift()
                startI = 0
                for endI, endNdx in enumerate(part.index[startI:]):
                    endNdx = round(float(endNdx), 5)
                    nextNdx = piece.score.highestTime if len(
                        part) - 1 == endI else part.index[endI + 1]
                    thisDur = part.iat[endI].quarterLength
                    if abs(nextNdx - endNdx - thisDur) > .00003:
                        strand = part.iloc[startI:endI + 1].copy()
                        strand.name = f'{piece.partNames[ii]}__{len(strands) + 1}'
                        divisiStarts.append(pd.Series(
                            ('*^', '*^'), index=(strand.name, piece.partNames[ii]), name=part.index[startI], dtype='string'))
                        joinNdx = endNdx + thisDur        # find a suitable endpoint to rejoin this strand
                        divisiEnds.append(pd.Series(('*v', '*v'), index=(
                            strand.name, piece.partNames[ii]), name=(strand.name, joinNdx), dtype='string'))
                        strands.append(strand)
                        startI = endI + 1
            kernStrands.extend(
                sorted(strands, key=lambda _strand: _strand.last_valid_index()))

        piece._analyses['_divisiStarts'] = pd.DataFrame(
            divisiStarts).fillna('*').sort_index()
        de = pd.DataFrame(divisiEnds)
        if not de.empty:
            de = de.reset_index(level=1)
            de = de.reindex(
                [prt.name for prt in kernStrands if prt.name not in piece.partNames]).set_index('level_1')
        piece._analyses['_divisiEnds'] = de
        if not isUnique:
            addTieBreakers(parts)
            addTieBreakers(kernStrands)
        piece._analyses['_partList'] = parts
        piece._analyses['_kernStrands'] = kernStrands
    return piece._analyses['_partList']

def _parts(piece, multi_index=False, kernStrands=False, compact=False,
           number=False):
    """
    Return a DataFrame of the note, rest, and chord objects in the score.

    The difference between parts and kernStrands is that parts can have voices
    whereas kernStrands cannot. If there are voices in the _parts DataFrame, the
    kernStrands DataFrame will include all these notes by adding additional
    columns.

    :param multi_index: Boolean, default False. If True, the returned DataFrame
        will have a MultiIndex.
    :param kernStrands: Boolean, default False. If True, the method will use the
        '_kernStrands' analysis.
    :param compact: Boolean, default False. If True, the method will keep chords
        unified rather then expanding them into separate columns.
    :param number: Boolean, default False. If True, the method will 1-index
        the part names and the voice names making the columns a MultiIndex. Only
        applies if `compact` is also True.
    :return: A DataFrame of the note, rest, and chord objects in the score.
    """
    key = ('_parts', multi_index, kernStrands, compact, number)

    # if piece.fileExtension == 'csv':
    #     return
    if key not in piece._analyses:
        toConcat = []
        if kernStrands:
            toConcat = piece._analyses['_kernStrands']
        elif compact:
            toConcat = _partList(piece, )
            if number:
                partNameToNum = {part: i + 1 for i,
                                 part in enumerate(piece.partNames)}
                colTuples = []
                for part in toConcat:
                    names = part.name.split('_')
                    if len(names) == 1:
                        colTuples.append((partNameToNum[names[0]], 1))
                    else:
                        colTuples.append(
                            (partNameToNum[names[0]], int(names[1]) + 1))
                mi = pd.MultiIndex.from_tuples(
                    colTuples, names=('Staff', 'Layer'))
        else:
            for part in _partList(piece, ):
                if part.empty:
                    toConcat.append(part)
                    continue
                listify = part.apply(
                    lambda nrc: nrc.notes if nrc.isChord else [nrc])
                expanded = listify.apply(pd.Series)
                expanded.columns = [f'{part.name}:{i}' if i > 0 else part.name for i in range(
                    len(expanded.columns))]
                toConcat.append(expanded)
        df = pd.concat(toConcat, axis=1, sort=True) if len(
            toConcat) else pd.DataFrame(columns=piece.partNames)
        if not multi_index and isinstance(df.index, pd.MultiIndex):
            df.index = df.index.droplevel(1)
        if compact and number:
            df.columns = mi
        piece._analyses[key] = df        
    return piece._analyses[key]

def _import_other_spines(piece, path=''):
    """
    Import the harmonic function spines from a given path.

    :param path: A string representing the path to the file containing the
        harmonic function spines.
    :return: A pandas DataFrame representing the harmonic function spines.
    """
    if piece.fileExtension == 'krn' or path:
        humFile = m21.humdrum.spineParser.HumdrumFile(path or piece.path)
        humFile.parseFilename()
        foundSpines = set()
        keyVals, keyPositions = [], []
        gotKeys = False
        for spine in humFile.spineCollection:
            if spine.spineType in ('kern', 'text', 'dynam'):
                continue
            foundSpines.add(spine.spineType)
            start = False
            vals, valPositions = [], []
            if len(keyVals):
                gotKeys = True
            for i, event in enumerate(spine.eventList):
                contents = event.contents
                if contents.endswith(':') and contents.startswith('*'):
                    start = True
                    # there usually won't be any m21 objects at the same position as the key events,
                    # so use the position from the next item in eventList if there is a next item.
                    if not gotKeys and i + 1 < len(spine.eventList):
                        # [1:-1] to remove the * and : characters
                        keyVals.append(contents[1:-1])
                        keyPositions.append(spine.eventList[i+1].position)
                    continue
                elif not start and spine.spineType not in ('function', 'harm') and not contents.startswith('*'):
                    start = True
                    if not contents.startswith('!') and not contents.startswith('='):
                        vals.append(contents)
                        valPositions.append(event.position)
                elif not start or '!' in contents or '=' in contents or '*' in contents:
                    continue
                else:
                    if spine.spineType == 'function':
                        functionLabel = function_pattern.sub('', contents)
                        if len(functionLabel):
                            vals.append(functionLabel)
                        else:
                            continue
                    else:
                        vals.append(contents)
                    valPositions.append(event.position)

            df1 = _priority(piece, )
            name = spine.spineType.title()
            if name == 'Cdata':
                df2 = pd.DataFrame([ast.literal_eval(val)
                                   for val in vals], index=valPositions)
            else:
                df2 = pd.DataFrame({name: vals}, index=valPositions)
            joined = df1.join(df2, on='Priority')
            if name != 'Cdata':   # get all the columns from the third to the end. Usually just 1 col except for cdata
                res = joined.iloc[:, 2].copy()
            else:
                res = joined.iloc[:, 2:].copy()
            res.index = joined['Offset']
            res.index.name = ''
            if spine.spineType not in piece._analyses:
                piece._analyses[spine.spineType] = [res]
            else:
                piece._analyses[spine.spineType].append(res)
            if not gotKeys and len(keyVals):
                keyName = 'keys'
                # key records are usually not found at a kern line with notes so take the next valid one
                keyPositions = [df1.iat[np.where(df1.Priority >= kp)[
                    0][0], 0] for kp in keyPositions]
                df3 = pd.DataFrame({keyName: keyVals}, index=keyPositions)
                joined = df1.join(df3, on='Priority')
                ser = joined.iloc[:, 2].copy()
                ser.index = joined['Offset']
                ser.index.name = ''
                piece._analyses[keyName] = ser
                gotKeys = True
        if len(foundSpines):
            piece.foundSpines = foundSpines

    for spine in ('function', 'harm', 'keys', 'chord'):
        if spine not in piece._analyses:
            piece._analyses[spine] = pd.Series(dtype='string')
    if 'cdata' not in piece._analyses:
        piece._analyses['cdata'] = pd.DataFrame()

def insertScoreDef(piece, root):
    """
    Insert a scoreDef element into an MEI (Music Encoding Initiative) document.

    This function inserts a scoreDef element into an MEI document if one is
    not already present. It modifies the input element in-place.

    :param root: An xml.etree.ElementTree.Element representing the root of the
        MEI document.
    :return: None
    """
    if root.find('.//scoreDef') is None:
        if piece.score is not None:
            clefs = _m21Clefs(piece, )
            ksigs = _keySignatures(piece, False)
            tsigs = _timeSignatures(piece, False)
            tsig1 = tsigs.iat[0, 0]
            scoreDef = ET.Element('scoreDef', {'xml:id': next(idGen), 'n': '1',
                                               'meter.count': f'{tsig1.numerator}', 'meter.unit': f'{tsig1.denominator}'})
        else:
            scoreDef = ET.Element(
                'scoreDef', {'xml:id': next(idGen), 'n': '1'})
        pgHead = ET.SubElement(scoreDef, 'pgHead')
        rend1 = ET.SubElement(
            pgHead, 'rend', {'halign': 'center', 'valign': 'top'})
        rend_title = ET.SubElement(
            rend1, 'rend', {'type': 'title', 'fontsize': 'x-large'})
        rend_title.text = 'Untitled score'
        ET.SubElement(rend1, 'lb')
        rend_subtitle = ET.SubElement(
            rend1, 'rend', {'type': 'subtitle', 'fontsize': 'large'})
        rend_subtitle.text = 'Subtitle'
        rend2 = ET.SubElement(
            pgHead, 'rend', {'halign': 'right', 'valign': 'bottom'})
        rend_composer = ET.SubElement(rend2, 'rend', {'type': 'composer'})
        rend_composer.text = 'Composer / arranger'
        staffGrp = ET.SubElement(scoreDef, 'staffGrp', {
                                 'xml:id': next(idGen), 'n': '1', 'symbol': 'bracket'})
        if not len(piece.partNames):
            piece.partNames = sorted(
                {f'Part-{staff.attrib.get("n")}' for staff in root.iter('staff')})
        for i, staff in enumerate(piece.partNames):
            attribs = {'label': staff, 'n': str(
                i + 1), 'xml:id': next(idGen), 'lines': '5'}
            if piece.score is not None:
                clef = clefs.iloc[0, i]
                attribs['clef.line'] = f'{clef.line}'
                attribs['clef.shape'] = clef.sign
                if clef.octaveChange != 0:
                    attribs['clef.dis'] = f'{abs(clef.octaveChange) * 8}'
                    attribs['clef.dis.place'] = 'below' if clef.octaveChange < 0 else 'above'
                ksig = ksigs.iloc[0, i] if not ksigs.empty else None
                if ksig:
                    val = len(ksig.alteredPitches)
                    if val > 0 and ksig.alteredPitches[0].accidental.modifier == '-':
                        attribs['key.sig'] = f'{val}f'
                    elif val > 0 and ksig.alteredPitches[0].accidental.modifier == '#':
                        attribs['key.sig'] = f'{val}s'
            staffDef = ET.SubElement(staffGrp, 'staffDef', attribs)
            label = ET.SubElement(staffDef, 'label', {
                                  'xml:id': next(idGen)})
            label.text = staff
        scoreEl = root.find('.//score')
        if scoreEl is not None:
            scoreEl.insert(0, scoreDef)

def xmlIDs(piece):
    """
    Return xml ids per part in a pandas.DataFrame time-aligned with the
    objects offset. If the file is not xml or mei, or an idString wasn't found,
    return a DataFrame of the ids of the music21 objects.

    :return: A pandas DataFrame representing the xml ids in the score.

    See Also
    --------
    :meth:`nmats`
    """
    if 'xmlIDs' in piece._analyses:
        return piece._analyses['xmlIDs']
    if piece.fileExtension in ('xml', 'mei'):
        tree = ET.parse(piece.path)
        root = tree.getroot()
        idString = [key for key in root.attrib.keys()
                    if key.endswith('}id')]
        if len(idString):
            idString = idString[0]
            data = {}
            dotCoefficients = {None: 1, '1': 1.5,
                               '2': 1.75, '3': 1.875, '4': 1.9375}
            for staff in root.findall('.//staff'):
                # doesn't need './/' because only looks for direct children of staff elements
                for layer in staff.findall('layer'):
                    column_name = f"Staff{staff.get('n')}_Layer{layer.get('n')}"
                    if column_name not in data:
                        data[column_name] = []
                    for nrb in layer:
                        if nrb.tag.endswith('note') or nrb.tag.endswith('rest') or nrb.tag.endswith('mRest'):
                            data[column_name].append(nrb.get(idString))
                        elif nrb.tag.endswith('beam'):
                            for nr in nrb:
                                data[column_name].append(nr.get(idString))
            ids = pd.DataFrame.from_dict(data, orient='index').T
            cols = []
            parts = _parts(piece, multi_index=True).copy()
            for i in range(len(parts.columns)):
                part = parts.iloc[:, i].dropna()
                idCol = ids.iloc[:, i].dropna()
                idCol.index = part.index
                cols.append(idCol)
            df = pd.concat(cols, axis=1)
            df.columns = parts.columns
            piece._analyses['xmlIDs'] = df
            return df
    # either not xml/mei, or an idString wasn't found
    df = _parts(piece, multi_index=True).map(
        lambda obj: f'{obj.id}', na_action='ignore')
    piece._analyses['xmlIDs'] = df
    return df

def _lyricHelper(piece, cell, strip):
    """
    Helper function for the lyrics method.

    :param cell: A music21 object.
    :return: The lyric of the music21 object.
    """
    if hasattr(cell, 'lyric'):
        lyr = cell.lyric
        if lyr and strip and len(lyr) > 1:
            lyr = lyr.strip(' \n\t-_')
        return lyr
    return np.nan

def lyrics(piece, strip=True):
    """
    Extract the lyrics from the score.

    The lyrics are extracted from each part and returned as a pandas DataFrame
    where each column represents a part and each row represents a lyric. The
    DataFrame is indexed by the offset of the lyrics.

    :param strip: Boolean, default True. If True, the method will strip leading
        and trailing whitespace from the lyrics.
    :return: A pandas DataFrame representing the lyrics in the score.

    See Also
    --------
    :meth:`dynamics`
    """
    key = ('lyrics', strip)
    if key not in piece._analyses:
        df = _parts(piece).map(
            lambda cell: _lyricHelper(piece, cell, strip),
            na_action='ignore'
        )
        piece._analyses[key] = df
    return piece._analyses[key].copy()

def _m21Clefs(piece):
    """
    Extract the clefs from the score.

    The clefs are extracted from each part and returned as a pandas DataFrame
    where each column represents a part and each row represents a clef. The
    DataFrame is indexed by the offset of the clefs.

    :return: A pandas DataFrame of the clefs in the score in music21's format.
    """
    if '_m21Clefs' not in piece._analyses:
        parts = []
        isUnique = True
        for i, flat_part in enumerate(piece._flatParts):
            ser = pd.Series(flat_part.getElementsByClass(
                ['Clef']), name=piece.partNames[i])
            ser.index = ser.apply(
                lambda nrc: nrc.offset).astype(float).round(5)
            ser = ser[~ser.index.duplicated(keep='last')]

            if not ser.index.is_unique:
                isUnique = False
            parts.append(ser)
        if not isUnique:
            for part in parts:
                tieBreakers = []
                nexts = part.index.to_series().shift(-1)
                for i in range(-1, -1 - len(part.index), -1):
                    if part.index[i] == nexts.iat[i]:
                        tieBreakers.append(tieBreakers[-1] - 1)
                    else:
                        tieBreakers.append(0)
                tieBreakers.reverse()
                part.index = pd.MultiIndex.from_arrays(
                    (part.index, tieBreakers))
        clefs = pd.concat(parts, axis=1)
        if isinstance(clefs.index, pd.MultiIndex):
            clefs = clefs.droplevel(1)
        piece._analyses['_m21Clefs'] = clefs
    return piece._analyses['_m21Clefs']

def _clefs(piece):
    """
    Extract the clefs from the score.

    The clefs are extracted from each part and returned as a pandas DataFrame
    where each column represents a part and each row represents a clef. The
    DataFrame is indexed by the offset of the clefs.

    :return: A pandas DataFrame of the clefs in the score in kern format.
    """
    if '_clefs' not in piece._analyses:
        piece._analyses['_clefs'] = _m21Clefs(piece, ).map(
            kernClefHelper, na_action='ignore')
    return piece._analyses['_clefs']

def dynamics(piece):
    """
    Extract the dynamics from the score.

    The dynamics are extracted from each part and returned as a pandas DataFrame
    where each column represents a part and each row represents a dynamic
    marking. The DataFrame is indexed by the offset of the dynamic markings.

    :return: A pandas DataFrame representing the dynamics in the score.

    See Also
    --------
    :meth:`lyrics`
    """
    if 'dynamics' not in piece._analyses:
        dyns = [pd.Series({obj.offset: obj.value for obj in sf.getElementsByClass(
            'Dynamic')}, dtype='string') for sf in piece._flatParts]
        dyns = pd.concat(dyns, axis=1)
        dyns.columns = piece.partNames
        dyns.dropna(how='all', axis=1, inplace=True)
        piece._analyses['dynamics'] = dyns
    return piece._analyses['dynamics'].copy()

def _priority(piece):
    """
    For .krn files, get the line numbers of the events in the piece, which
    music21 often calls "priority". For other encoding formats return an
    empty dataframe.

    :return: A DataFrame containing the priority values.
    """
    if '_priority' not in piece._analyses:
        if piece.fileExtension != 'krn':
            priority = pd.DataFrame()
        else:
            # use compact to avoid losing priorities of chords
            parts = _parts(piece, compact=True)                
            if parts.empty:
                priority = pd.DataFrame()
            else:
                priority = parts.map(lambda cell: cell.priority, na_action='ignore').ffill(
                    axis=1).iloc[:, -1].astype('Int16')
                priority = pd.DataFrame(
                    {'Priority': priority.values, 'Offset': priority.index})
        piece._analyses['_priority'] = priority
    return piece._analyses['_priority']

def keys(piece, snap_to=None, filler='forward', output='array'):
    """
    Get the key signature portion of the **harm spine in a kern file if there
    is one and return it as an array or a time-aligned pandas Series. This is
    similar to the .harm, .functions, .chords, and .cdata methods. The default
    is for the results to be returned as a 1-d array, but you can set `output='series'`
    for a pandas series instead. If want to get the results of a different spine
    type (i.e. not one of the ones listed above), see :meth:`getSpines`.

    See Also
    --------
    :meth:`cdata`
    :meth:`chords`
    :meth:`functions`
    :meth:`harm`
    :meth:`getSpines`
    """
    if snap_to is not None:
        output = 'series'
    return snapTo(piece._analyses['keys'], snap_to, filler, output)

def harm(piece, snap_to=None, filler='forward', output='array'):
    """
    Get the harmonic analysis portion of the **harm spine in a kern file if there
    is one and return it as an array or a time-aligned pandas Series. The prevailing
    key signature information is not included here from the harm spine, but that key
    information is available in the .keys method. This is similar to the
    .keys, .functions, .chords, and .cdata methods. The default is for the
    results to be returned as a 1-d array, but you can set `output='series'`
    for a pandas series instead which is helpful if you're going to concatenate
    the results to a dataframe. If want to get the results of a different spine
    type (i.e. not one of the ones listed above), see :meth:`getSpines`.

    If you want to align these results so that they match the columnar (time) axis
    of the pianoRoll, sampled, or mask results, you can pass the pianoRoll or mask
    that you want to align to as the `snap_to` parameter. Doing that makes it easier
    to combine these results with any of the pianoRoll, sampled, or mask tables to
    have both in a single table which can make data analysis easier. Passing a `snap_to`
    argument will automatically cause the return value to be a pandas series since
    that's facilitates combining the two. Here's how you would use the `snap_to`
    parameter and then combine the results with the pianoRoll to create a single table.

    See Also
    --------
    :meth:`cdata`
    :meth:`chords`
    :meth:`functions`
    :meth:`keys`
    """
    return snapTo(piece._analyses['harm'], snap_to, filler, output)

def functions(piece, snap_to=None, filler='forward', output='array'):
    """
    Get the harmonic function labels from a **function spine in a kern file if there
    is one and return it as an array or a time-aligned pandas Series. This is
    similar to the .harm, .keys, .chords, and .cdata methods. The default
    is for the results to be returned as a 1-d array, but you can set `output='series'`
    for a pandas series instead. If want to get the results of a different spine
    type (i.e. not one of the ones listed above), see :meth:`getSpines`.

    See Also
    --------
    :meth:`cdata`
    :meth:`chords`
    :meth:`harm`
    :meth:`keys`
    """
    if snap_to is not None:
        output = 'series'
    return snapTo(piece._analyses['function'], snap_to, filler, output)

def chords(piece, snap_to=None, filler='forward', output='array'):
    """
    Get the chord labels from the **chord spine in a kern file if there
    is one and return it as an array or a time-aligned pandas Series. This is
    similar to the .functions, .harm, .keys, and .cdata methods. The default
    is for the results to be returned as a 1-d array, but you can set `output='series'`
    for a pandas series instead. If want to get the results of a different spine
    type (i.e. not one of the ones listed above), see :meth:`getSpines`.

    See Also
    --------
    :meth:`cdata`
    :meth:`functions`
    :meth:`harm`
    :meth:`keys`
    """
    if snap_to is not None:
        output = 'series'
    return snapTo(piece._analyses['chord'], snap_to, filler, output)

def cdata(piece, snap_to=None, filler='forward', output='dataframe'):
    """
    Get the cdata records from **cdata spines in a kern file if there
    are any and return it as a pandas DataFrame. This is
    similar to the .harm, .functions, .chords, and .keys methods, with the
    exception that this method defaults to returning a dataframe since there are
    often more than one cdata spine in a kern score. If want to get the results
    of a different spine type (i.e. not one of the ones listed above), see
    :meth:`getSpines`.

    See Also
    --------
    :meth:`chords`
    :meth:`functions`
    :meth:`harm`
    :meth:`keys`
    :meth:`getSpines`
    """
    if snap_to is not None:
        output = 'dataframe'
    return snapTo(piece._analyses['cdata'], snap_to, filler, output)

def getSpines(piece, spineType):
    """
    Return a pandas DataFrame of a less common spine type. This method is a
    window into the vast ecosystem of Humdrum tools making them accessible to
    pyAMPACT.

    :param spineType: A string representing the spine type to return. You can
        pass the spine type with or without the "**" prefix.
    :return: A pandas DataFrame of the given spine type.

    Similar to the .harm, .keys, .functions, .chords, and .cdata methods, this
    method returns the contents of a specific spine type from a kern file. This is
    a generic method that can be used to get the contents of any spine type other
    than: **kern, **dynam, **text, **cdata, **chord, **harm, or **function. Many
    of the other spine types that you may be interested provide partwise data.
    For example, the results of Humlib's Renaissance dissonance analysis are given
    as one "**cdata-rdiss" spine per part. Note that a **cdata-rdiss spine is not
    the same as a **cdata spine. This is why we return a DataFrame rather than
    an array or series. If there is just one spine of the spine type you request,
    the data will still be given as a 1-column dataframe. When you import a kern
    file, it automatically gets scanned for other spine types and if any are found
    you can see them with the `foundSpines` attribute.

    This example takes a score with **cdata-rdiss spines (Renaissance dissonance
    analysis), and makes a DataFrame of just the **cdata-rdiss spines. The full
    score with color-coded dissonance labels can be seen on the Verovio Humdrum
    Viewer `here <https://verovio.humdrum.org/?k=ey&filter=dissonant%20--color&file=jrp:Tin2004>`_.

    See Also
    --------
    :meth:`cdata`
    :meth:`chords`
    :meth:`dynamics`
    :meth:`functions`
    :meth:`harm`
    :meth:`keys`
    :meth:`lyrics`
    """
    if spineType.startswith('**'):
        spineType = spineType[2:]
    if hasattr(piece, 'foundSpines') and spineType in piece.foundSpines:
        ret = snapTo(piece._analyses[spineType],
                     filler='nan', output='dataframe')
        ret.dropna(how='all', inplace=True)
        if len(ret.columns) == len(piece.partNames):
            ret.columns = piece.partNames
        return ret
    if piece.fileExtension != 'krn':
        print(f'\t***This is not a kern file so there are no spines to import.***')
    else:
        print(f'\t***No {spineType} spines were found.***')

def dez(piece, path=''):
    """
    Get the labels data from a .dez file/url and return it as a dataframe. Calls
    fromJSON to do this. The "meta" portion of the dez file is ignored. If no
    path is provided, the last dez table imported with this method is returned.

    :param path: A string representing the path to the .dez file.
    :return: A pandas DataFrame representing the labels in the .dez file.
    """
    if 'dez' not in piece._analyses:
        if not path:
            print(
                'No path was provided and no prior analysis was found. Please provide a path to a .dez file.')
            return
        elif not path.endswith('.dez'):
            print('The file provided is not a .dez file.')
            return
        elif not path.startswith('http') and not os.path.exists(path):
            print('The file provided does not exist.')
            return
        else:
            piece._analyses['dez'] = {path: fromJSON(path)}
    else:
        if not path:   # return the last dez table
            return next(reversed(piece._analyses['dez'].values()))
        else:
            if path not in piece._analyses['dez']:
                piece._analyses['dez'][path] = fromJSON(path)
    return piece._analyses['dez'][path]

def form(piece, snap_to=None, filler='forward', output='array', dez_path=''):
    """
    Get the "Structure" labels from a .dez file/url and return it as an array or a
    time-aligned pandas Series. The default is for the results to be returned as a 1-d
    array, but you can set `output='series'` for a pandas series instead. If you want to align
    these results so that they match the columnar (time) axis of the pianoRoll, sampled, or
    mask results, you can pass the pianoRoll or mask that you want to align to as the `snap_to`
    parameter. Doing that makes it easier to combine these results with any of the pianoRoll,
    sampled, or mask tables to have both in a single table which can make data analysis easier.
    This example shows how to get the form analysis from a .dez file.
    """
    if not dez_path and 'dez' not in piece._analyses:
        print('No .dez file was found.')
    else:
        dez_result = dez(piece, dez_path)
        df = dez_result.set_index('start').rename_axis(None)
        df = df.loc[(df['type'] == 'Structure'), 'tag']
        if df.empty:
            print('No "Structure" analysis was found in the .dez file.')
        else:
            return snapTo(df, snap_to, filler, output)

def romanNumerals(piece, snap_to=None, filler='forward', output='array', dez_path=''):
    """
    Get the roman numeral labels from a .dez file/url or **harm spine and return it as an array
    or a time-aligned pandas Series. The default is for the results to be returned as a 1-d
    array, but you can set `output='series'` for a pandas series instead. If you want to align
    these results so that they match the columnar (time) axis of the pianoRoll, sampled, or mask
    results, you can pass the pianoRoll or mask that you want to align to as the `snap_to` parameter.
    Doing that makes it easier to combine these results with any of the pianoRoll, sampled, or mask
    tables to have both in a single table which can make data analysis easier. This example shows
    how to get the roman numeral analysis from a kern score that has a **harm spine.
    """
    if dez_path or 'dez' in piece._analyses:
        dez_result = dez(piece, dez_path)
        if dez_result is not None:
            df = dez_result.set_index('start').rename_axis(None)
            df = df.loc[(df['type'] == 'Harmony'), 'tag']
            if df.empty:
                print(
                    'No "Harmony" analysis was found in the .dez file, checking for a **harm spine.')
            else:
                return snapTo(df, snap_to, filler, output)
    if 'harm' in piece._analyses and len(piece._analyses['harm']):
        return harm(piece, snap_to=snap_to, filler=filler, output=output)
    print('Neither a dez nor a **harm spine was found so using music21 to get roman numerals...')
    key = piece.score.analyze('key')
    chords = piece.score.chordify().recurse().getElementsByClass('Chord')
    offsets = [ch.offset for ch in chords]
    figures = [m21.roman.romanNumeralFromChord(
        ch, key).figure for ch in chords]
    ser = pd.Series(figures, index=offsets, name='Roman Numerals')
    ser = ser[ser != ser.shift()]   # remove consecutive duplicates
    return snapTo(ser, snap_to, filler, output)

def _m21ObjectsNoTies(piece):
    """
    Remove tied notes in a given voice. Only the first note in a tied group
    will be kept.

    :param voice: A music21 stream Voice object.
    :return: A list of music21 objects with ties removed.
    """
    if '_m21ObjectsNoTies' not in piece._analyses:
        piece._analyses['_m21ObjectsNoTies'] = _parts(piece, 
            multi_index=True).map(removeTied).dropna(how='all')
    return piece._analyses['_m21ObjectsNoTies']

def _measures(piece, compact=False):
    """
    Return a DataFrame of the measure starting points.

    :param compact: Boolean, default False. If True, the method will keep
        chords unified rather then expanding them into separate columns.
    :return: A DataFrame where each column corresponds to a part in the score,
        and each row index is the offset of a measure start. The values are
        the measure numbers.
    """
    if ('_measures', compact) not in piece._analyses:
        partCols = _parts(piece, compact=compact).columns
        partMeasures = []
        for i, part in enumerate(piece._flatParts):
            meas = {m.offset: m.measureNumber for m in part.makeMeasures(
            ) if isinstance(m, m21.stream.Measure)}
            ser = [pd.Series(meas, dtype='Int16')]
            voiceCount = len(
                [col for col in partCols if col.startswith(piece.partNames[i])])
            partMeasures.extend(ser * voiceCount)
        df = pd.concat(partMeasures, axis=1)
        df.columns = partCols
        piece._analyses[('_measures', compact)] = df
    return piece._analyses[('_measures', compact)].copy()

def _barlines(piece):
    """
    Return a DataFrame of barlines specifying which barline type.

    Double barline, for example, can help detect section divisions, and the
    final barline can help process the `highestTime` similar to music21.

    :return: A DataFrame where each column corresponds to a part in the score,
        and each row index is the offset of a barline. The values are the
        barline types.
    """
    if "_barlines" not in piece._analyses:
        partBarlines = [pd.Series({bar.offset: bar.type for bar in part.getElementsByClass(['Barline'])})
                        for i, part in enumerate(piece._flatParts)]
        df = pd.concat(partBarlines, axis=1, sort=True)
        df.columns = piece.partNames
        piece._analyses["_barlines"] = df
    return piece._analyses["_barlines"]

def _keySignatures(piece, kern=True):
    """
    Return a DataFrame of key signatures for each part in the score.

    :param kern: Boolean, default True. If True, the key signatures are
        returned in the **kern format.
    :return: A DataFrame where each column corresponds to a part in the score,
        and each row index is the offset of a key signature. The values are
        the key signatures.
    """
    if ('_keySignatures', kern) not in piece._analyses:
        kSigs = []
        for i, part in enumerate(piece._flatParts):
            kSigs.append(pd.Series({ky.offset: ky for ky in part.getElementsByClass(
                ['KeySignature'])}, name=piece.partNames[i]))
        df = pd.concat(kSigs, axis=1).sort_index(kind='mergesort')
        if kern:
            df = '*k[' + df.map(lambda ky: ''.join(
                [_note.name for _note in ky.alteredPitches]).lower(), na_action='ignore') + ']'
        piece._analyses[('_keySignatures', kern)] = df
    return piece._analyses[('_keySignatures', kern)]

def _timeSignatures(piece, ratio=True):
    """
    Return a DataFrame of time signatures for each part in the score.

    :return: A DataFrame where each column corresponds to a part in the score,
        and each row index is the offset of a time signature. The values are
        the time signatures in ratio string format.
    """
    if ('_timeSignatures', ratio) not in piece._analyses:
        tsigs = []
        for i, part in enumerate(piece._flatParts):
            if not ratio:
                tsigs.append(pd.Series(
                    {ts.offset: ts for ts in part.getTimeSignatures()}, name=piece.partNames[i]))
            else:
                tsigs.append(pd.Series(
                    {ts.offset: ts.ratioString for ts in part.getTimeSignatures()}, name=piece.partNames[i]))
        df = pd.concat(tsigs, axis=1).sort_index(kind='mergesort')
        piece._analyses[('_timeSignatures', ratio)] = df
    return piece._analyses[('_timeSignatures', ratio)]

def durations(piece, multi_index=False, df=None):
    """
    Return a DataFrame of durations of note and rest objects in the piece.

    If a DataFrame is provided as `df`, the method calculates the difference
    between cell offsets per column in the passed DataFrame, skipping
    memoization.

    :param multi_index: Boolean, default False. If True, the returned DataFrame
        will have a MultiIndex.
    :param df: Optional DataFrame. If provided, the method calculates the
        difference between cell offsets per column in this DataFrame.
    :return: A DataFrame of durations of note and rest objects in the piece.

    See Also
    --------
    :meth:`notes`
        Return a DataFrame of the notes and rests given in American Standard
        Pitch Notation
    """
    if df is None:
        key = ('durations', multi_index)
        if key not in piece._analyses:
            m21objs = _m21ObjectsNoTies(piece, )
            res = m21objs.map(lambda nrc: nrc.quarterLength,
                              na_action='ignore').astype(float).round(5)
            if not multi_index and isinstance(res.index, pd.MultiIndex):
                res = res.droplevel(1)
            piece._analyses[key] = res
        return piece._analyses[key]
    else:   # df is not None so calculate diff between cell offsets per column in passed df, skip memoization
        sers = []
        for col in range(len(df.columns)):
            part = df.iloc[:, col].dropna()
            ndx = part.index.get_level_values(0)
            if len(part) > 1:
                vals = (ndx[1:] - ndx[:-1]).to_list()
            else:
                vals = []
            if not part.empty:
                vals.append(piece.score.highestTime - ndx[-1])
            sers.append(pd.Series(vals, part.index, dtype='float64'))
        res = pd.concat(sers, axis=1, sort=True)
        if not multi_index and isinstance(res.index, pd.MultiIndex):
            res = res.droplevel(1)
        res.columns = df.columns
        return res

def midi_ticks_durations(piece, i=1, df=None):
    """
    Replaces the placeholder ONSET_SEC and OFFSET_SEC columns with timing information
    directly extracted from the music21 stream.

    Parameters
    ----------
    i : int
        Part index (1-based)
    df : pd.DataFrame or None
        Optional DataFrame with rows matching notes in the selected part.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with ONSET_SEC, OFFSET_SEC, DURATION, and TEMPO columns.
    """

    # --- Step 1: Get BPM from music21 score ---
    bpm = None
    try:
        for el in piece.score.flatten().getElementsByClass(m21.tempo.MetronomeMark):
            if el.number:
                bpm = float(el.number)
                break
    except Exception:
        pass

    if bpm is None:
        try:
            for part in piece.score.parts:
                for el in part.flatten().getElementsByClass(m21.tempo.MetronomeMark):
                    if el.number:
                        bpm = float(el.number)
                        break
                if bpm is not None:
                    break
        except Exception:
            pass

    if bpm is None or bpm <= 0:
        bpm = 120

    # --- Step 2: Clamp part index to valid range ---
    # nmats() calls midi_ticks_durations(piece, i+1, df) where i comes from
    # enumerate(_parts(piece).columns). _parts() expands chords into extra
    # columns, so i+1 can exceed len(_flatParts). Clamp to last valid part.
    num_parts = len(piece._flatParts)
    part_index = max(0, min(i - 1, num_parts - 1))  # convert to 0-based, clamp

    # --- Step 3: Extract onset/offset seconds directly from music21 ---
    onsOffsList = []
    try:
        part_for_seconds = piece.score.parts[part_index]
        seconds_map = part_for_seconds.secondsMap
        for entry in seconds_map:
            el = entry['element']
            if isinstance(el, (m21.note.Note, m21.chord.Chord)):
                onset = float(entry['offsetSeconds'])
                offset = float(entry['endTimeSeconds'])
                onsOffsList.append([onset, offset])
    except Exception:
        pass

    if not onsOffsList:
        # Fallback: convert quarter-note offsets to seconds via bpm
        seconds_per_beat = 60.0 / bpm
        try:
            part_stream = piece._flatParts[part_index]
            for nrc in part_stream.getElementsByClass(['Note', 'Chord']):
                onset = float(nrc.offset) * seconds_per_beat
                offset = onset + float(nrc.quarterLength) * seconds_per_beat
                onsOffsList.append([onset, offset])
        except Exception:
            df['TEMPO'] = bpm
            return df

    if not onsOffsList:
        df['TEMPO'] = bpm
        return df

    # --- Step 4: Match length to df rows ---
    onsOffsList = truncate_and_scale_onsOffsList(onsOffsList, len(df.index))

    res = pd.DataFrame(
        onsOffsList, columns=['ONSET_SEC', 'OFFSET_SEC'], index=df.index)

    res = res.sort_values(by='ONSET_SEC')

    # --- Step 5: Enforce minimum duration and clamp overlaps ---
    grace_threshold = 0.03
    min_gap = 0.025

    onsets = res['ONSET_SEC'].values.copy()
    offsets = res['OFFSET_SEC'].values.copy()

    for j in range(len(res) - 1):
        dur = offsets[j] - onsets[j]
        if dur < grace_threshold:
            continue
        next_onset = onsets[j + 1]
        if offsets[j] < next_onset:
            offsets[j] = next_onset
        if offsets[j] > next_onset - min_gap:
            offsets[j] = max(onsets[j], next_onset - min_gap)

    res['OFFSET_SEC'] = offsets
    res['DURATION'] = res['OFFSET_SEC'] - res['ONSET_SEC']
    df['ONSET_SEC'] = res['ONSET_SEC']
    df['OFFSET_SEC'] = res['OFFSET_SEC']
    df['DURATION'] = res['DURATION']
    df['TEMPO'] = bpm

    return df

def contextualize(piece, df, offsets=True, measures=True, beats=True):
    """
    Add measure and beat numbers to a DataFrame.

    :param df: A DataFrame to which to add measure and beat numbers.
    :param measures: Boolean, default True. If True, measure numbers will be added.
    :param beats: Boolean, default True. If True, beat numbers will be added.
    :return: A DataFrame with measure and beat numbers added.
    """
    _copy = df.copy()
    if _copy.index.names[0] == 'XML_ID':
        _copy['XML_ID'] = _copy.index.get_level_values(0)
        _copy.index = _copy['ONSET']
    col_names = _copy.columns
    _copy.columns = range(len(_copy.columns))
    _copy = _copy[~_copy.index.duplicated(keep='last')]
    toConcat = [_copy]
    cols = []
    if measures:
        meas = _measures(piece, ).iloc[:, 0]
        meas.name = 'Measure'
        toConcat.append(meas)
        cols.append('Measure')
    if beats:
        bts = _beats(piece, ).apply(
            lambda row: row[row.first_valid_index()], axis=1)
        bts = bts.loc[~bts.index.duplicated(keep='last')]
        bts.name = 'Beat'
        toConcat.append(bts)
        cols.append('Beat')
    ret = pd.concat(toConcat, axis=1).sort_index()
    if offsets:
        ret.index.name = 'Offset'
    if measures:
        ret['Measure'] = ret['Measure'].ffill()
    ret = ret.set_index(cols, append=True).dropna(how='all')
    if not offsets:
        ret.index = ret.index.droplevel(0)
    ret.columns = col_names
    return ret

def _beats(piece):
    """
    Return a DataFrame of beat numbers for each part in the score.

    :return: A DataFrame where each column corresponds to a part in the score,
        and each row index is the offset of a beat. The values are the beat
        numbers.
    """
    if '_beats' not in piece._analyses:
        df = _parts(piece, compact=True).map(
            lambda obj: obj.beat, na_action='ignore')
        piece._analyses['_beats'] = df
    return piece._analyses['_beats']

def midiPitches(piece, multi_index=False):
    """
    Return a DataFrame of notes and rests as MIDI pitches.

    MIDI does not have a representation for rests, so -1 is used as a
    placeholder.

    :param multi_index: Boolean, default False. If True, the returned DataFrame
        will have a MultiIndex.
    :return: A DataFrame of notes and rests as MIDI pitches. Rests are
        represented as -1.

    See Also
    --------
    :meth:`kernNotes`
        Return a DataFrame of the notes and rests given in kern notation.
    :meth:`notes`
        Return a DataFrame of the notes and rests given in American Standard
        Pitch Notation
    """
    key = ('midiPitches', multi_index)
    if key not in piece._analyses:
        midiPitches = _m21ObjectsNoTies(piece, ).map(
            lambda nr: -1 if nr.isRest else nr.pitch.midi, na_action='ignore')
        if not multi_index and isinstance(midiPitches.index, pd.MultiIndex):
            midiPitches = midiPitches.droplevel(1)
        piece._analyses[key] = midiPitches
    return piece._analyses[key]

def notes(piece, combine_rests=True, combine_unisons=False):
    """
    Return a DataFrame of the notes and rests given in American Standard Pitch
    Notation where middle C is C4. Rests are designated with the string "r".

    If `combine_rests` is True (default), non-first consecutive rests will be
    removed, effectively combining consecutive rests in each voice.
    `combine_unisons` works the same way for consecutive attacks on the same
    pitch in a given voice, however, `combine_unisons` defaults to False.

    :param combine_rests: Boolean, default True. If True, non-first consecutive
        rests will be removed.
    :param combine_unisons: Boolean, default False. If True, consecutive attacks
        on the same pitch in a given voice will be combined.
    :return: A DataFrame of notes and rests in American Standard Pitch Notation.

    See Also
    --------
    :meth:`kernNotes`
        Return a DataFrame of the notes and rests given in kern notation.
    :meth:`midiPitches`
    """
    if 'notes' not in piece._analyses:
        df = _m21ObjectsNoTies(piece, ).map(noteRestHelper, na_action='ignore')
        piece._analyses['notes'] = df
    ret = piece._analyses['notes'].copy()
    if combine_rests:
        ret = ret.apply(combineRests)
    if combine_unisons:
        ret = ret.apply(combineUnisons)
    if isinstance(ret.index, pd.MultiIndex):
        ret = ret.droplevel(1)
    return ret

def kernNotes(piece):
    """
    Return a DataFrame of the notes and rests given in kern notation.

    This is not the same as creating a kern format of a score, but is an
    important step in that process.

    :return: A DataFrame of notes and rests in kern notation.

    See Also
    --------
    :meth: `midiPitches`
    :meth:`notes`
        Return a DataFrame of the notes and rests given in American Standard
        Pitch Notation
    """
    if 'kernNotes' not in piece._analyses:
        piece._analyses['kernNotes'] = _parts(piece, 
            True, True).map(kernNRCHelper, na_action='ignore')
    return piece._analyses['kernNotes'].copy()

def nmats(piece, json_path=None, include_cdata=False):
    """
    Return a dictionary of DataFrames, one for each voice, with information
    about the notes and rests in that voice.

    Each DataFrame has the following columns:

    MEASURE  ONSET  DURATION  PART  MIDI  ONSET_SEC  OFFSET_SEC

    In the MIDI column, notes are represented
    with their MIDI pitch numbers (0 to 127), and rests are represented with -1s.
    The ONSET_SEC and OFFSET_SEC columns are taken from the audio analysis from
    the `json_path` file if one is given. The XML_IDs of each note or rest serve
    as the index for this DataFrame. If `include_cdata` is True and a `json_path`
    is provided, the cdata from the json file is included in the DataFrame.

    :param json_path: Optional path to a JSON file containing audio analysis data.
    :param include_cdata: Boolean, default False. If True and a `json_path` is
        provided, the cdata from the json file is included in the DataFrame.
    :return: A dictionary of DataFrames, one for each voice.

    See Also
    --------
    :meth:`fromJSON`
    :meth:`insertAudioAnalysis`
    :meth:`jsonCDATA`
    :meth:`xmlIDs`

    """
    if not json_path:   # user must pass a json_path if they want the cdata to be included
        include_cdata = False
    key = ('nmats', json_path, include_cdata)

    if key not in piece._analyses:
        nmats = {}
        included = {}
        dur = durations(piece, multi_index=True)
        mp = midiPitches(piece, multi_index=True)
        ms = _measures(piece, )
        ids = xmlIDs(piece, )
        if json_path:
            if json_path.lower().endswith('.json'):
                data = fromJSON(json_path) if json_path else pd.DataFrame()
            elif json_path.lower().endswith('.csv'):
                data = pd.read_csv(githubURLtoRaw(json_path), header=None)
                col_names = ('ONSET_SEC', 'MIDI', 'DURATION')                    
                # sometimes these files only have two columns instead of three
                data.columns = col_names[:len(data.columns)]
                data['MIDI'] = data['MIDI'].map(
                    librosa.hz_to_midi, na_action='ignore').round().astype('Int16')

        if isinstance(ids.index, pd.MultiIndex):
            ms.index = pd.MultiIndex.from_product((ms.index, (0,)))

        for i, partName in enumerate(_parts(piece, ).columns):                
            meas = ms.iloc[:, i]
            midi = mp.iloc[:, i].dropna()                
            onsetBeat = pd.Series(midi.index.get_level_values(
                0), index=midi.index, dtype='float64')             
            durBeat = dur.iloc[:, i].dropna()
            part = pd.Series(partName, midi.index, dtype='string')
            xmlID = ids.iloc[:, i].dropna()
            if piece.fileExtension == 'csv':
                csv_data = piece._analyses['tony_csv']                                   
                csv_data.index = part.index
                onsetSec = csv_data['ONSET_SEC']
                offsetSec = csv_data['ONSET_SEC'] + csv_data['DURATION']

                # For music            
                concat_list = [meas, onsetBeat, durBeat, part, midi, onsetSec, offsetSec, xmlID]
                column_names = ['MEASURE', 'ONSET', 'DURATION', 'PART',
                                'MIDI', 'ONSET_SEC', 'OFFSET_SEC', 'XML_ID']

                # For speech
                if 'WORD' in csv_data.columns:
                    words = csv_data['WORD']
                    avgPitch = csv_data['AVG PITCH IN HZ']
                    concat_list.append(words)
                    concat_list.append(avgPitch)
                    column_names.append('WORD')
                    column_names.append('AVG PITCH IN HZ')                        

            else:
                onsetSec = onsetBeat.copy()
                offsetSec = onsetBeat + durBeat
                concat_list = [meas, onsetBeat, durBeat, part, midi, onsetSec, offsetSec, xmlID]
                column_names = ['MEASURE', 'ONSET', 'DURATION', 'PART',
                                'MIDI', 'ONSET_SEC', 'OFFSET_SEC', 'XML_ID']

            df = pd.concat(concat_list, axis=1, sort=True)
            df.columns = column_names
            df['MEASURE'] = df['MEASURE'].ffill()
            df.dropna(how='all', inplace=True, subset=df.columns[1:5])
            df = df.set_index('XML_ID')                                

            if piece.fileExtension != 'csv':
                # Remove rows where MIDI == -1.0                                 
                df = df[df['MIDI'] != -1.0]                    
                df = midi_ticks_durations(piece, i+1, df)

            if json_path is not None:   # add json data if a json_path is provided
                if len(data.index) > len(df.index):
                    data = data.iloc[:len(df.index), :]
                    print(
                        '\n\n*** Warning ***\n\nThe json data has more observations than there are notes in this part so the data was truncated.\n')
                elif len(data.index) < len(df.index):
                    df = df.iloc[:len(data.index), :]
                    print(
                        '\n\n*** Warning ***\n\nThere are more events than there are json records in this part.\n')
                data.index = df.index
                if json_path.lower().endswith('.json'):
                    df.iloc[:len(data.index), 5] = data.index
                    if len(data.index) > 1:
                        df.iloc[:len(data.index) - 1, 6] = data.index[1:]
                    data.index = df.index[:len(data.index)]
                    df = pd.concat((df, data), axis=1)
                    included[partName] = df
                    df = df.iloc[:, :7].copy()

                elif json_path.lower().endswith('.csv'):
                    df[['ONSET_SEC', 'MIDI']] = data[['ONSET_SEC', 'MIDI']]
                    if 'DURATION' in data.columns:
                        df.OFFSET_SEC = df.ONSET_SEC + data['DURATION']
                    included[partName] = df

            nmats[partName] = df
        piece._analyses[('nmats', json_path, False)] = nmats
        if json_path:
            piece._analyses[('nmats', json_path, True)] = included
    return piece._analyses[key]

def pianoRoll(piece):
    """
    Construct a MIDI piano roll. This representation of a score plots midi pitches
    on the y-axis (rows) and time on the x-axis (columns). Midi pitches are given
    as integers from 0 to 127 inclusive, and time is given in quarter notes counting
    up from the beginning of the piece. At any given time in the piece (column), all
    the sounding pitches are shown as 1s in the corresponding rows. There is no
    midi representation of rests so these are not shown in the pianoRoll. Similarly,
    in this representation you can't tell if a single voice is sounding a given note,
    of if multiple voices are sounding the same note. The end result looks like a
    player piano roll but 1s are used instead of holes. This method is primarily
    used as an intermediate step in the construction of a mask.

    Note: There are 128 possible MIDI pitches.

    :return: A DataFrame representing the MIDI piano roll. Each row corresponds
        to a MIDI pitch (0 to 127), and each column corresponds to an offset in
        the score. The values are 1 for a note onset and 0 otherwise.

    See Also
    --------
    :meth:`mask`
    :meth:`sampled`
    """

    if 'pianoRoll' not in piece._analyses:
        if piece.fileExtension == 'csv':
            mp = piece._analyses['nmats', None, False][
                'Part-1']['MIDI']
            df = mp.reset_index().rename(
                columns={'ONSET_SEC': 'Index', 'MIDI': 'Part-1'})
            df.set_index('Index', inplace=True)
            mp = df
        else:
            mp = midiPitches(piece, )

        # remove non-last offset repeats and forward-fill
        mp = mp[~mp.index.duplicated(keep='last')].ffill()
        pianoRoll = pd.DataFrame(index=range(128), columns=mp.index.values)
        for offset in mp.index:
            for pitch in mp.loc[offset]:
                if pitch >= 0:
                    pianoRoll.at[pitch, offset] = 1
        pianoRoll = pianoRoll.infer_objects(copy=False).fillna(0)
        piece._analyses['pianoRoll'] = pianoRoll
    return piece._analyses['pianoRoll']

def sampled(piece, bpm=60, obs=24):
    """
    Sample the score according to the given beats per minute (bpm) and the
    desired observations per second (obs). This method is primarily used as an
    intermediate step in the construction of a mask. It builds on the pianoRoll
    by sampling the time axis (columns) at the desired rate. The result is a
    DataFrame where each row corresponds to a MIDI pitch (0 to 127), and each
    column corresponds to a timepoint in the sampled score. The difference
    between this and the pianoRoll is that the columns are sampled at a regular
    time intervals, rather than at each new event as they are in the pianoRoll.

    :param bpm: Integer, default 60. The beats per minute to use for sampling.
    :param obs: Integer, default 24. The desired observations per second.
    :return: A DataFrame representing the sampled score. Each row corresponds
        to a MIDI pitch (0 to 127), and each column corresponds to a timepoint
        in the sampled score. The values are 1 for a note onset and 0 otherwise.

    See Also
    --------
    :meth:`mask`
    :meth:`pianoRoll`
    """
    key = ('sampled', bpm, obs)

    if key not in piece._analyses:
        if piece.fileExtension == 'csv':
            highestTime = (piece._analyses['nmats', None, False][
                'Part-1']['OFFSET_SEC'].max())
            slices = 60/bpm * obs
            timepoints = pd.Index(
                [t/slices for t in range(0, int(highestTime * slices))])

        else:
            slices = 60/bpm * obs
            timepoints = pd.Index(
                [t/slices for t in range(0, int(piece.score.highestTime * slices))])
        pr = pianoRoll(piece, ).copy()
        pr.columns = [col if col in timepoints else timepoints.asof(
            col) for col in pr.columns]
        pr = pr.T
        pr = pr.iloc[~pr.index.duplicated(keep='last')]
        pr = pr.T
        sampled_pr = pr.reindex(columns=timepoints, method='ffill')
        piece._analyses[key] = sampled_pr
    return piece._analyses[key]

def mask(piece, winms=100, sample_rate=4000, num_harmonics=1, width=0,
         bpm=60, aFreq=440, base_note=0, tuning_factor=1, obs=24):
    """
    Construct a mask from the sampled piano roll using width and harmonics. This
    builds on the intermediate representations of the pianoRoll and sampled
    methods. The sampled method already put the x-axis (columns) in regular
    time intervals. The mask keeps these columns and then alters the y-axis (rows)
    into frequency bins. The number of bins is determined by the winms and sample_rate
    values, and is˚ equal to some power of 2 plus 1. The frequency bins serve to "blur"
    the sampled pitch data that we expect from the score. This allows us to detect
    real performed sounds in audio recordings that are likely slightly above or below
    the precise notated pitches. The mask is what allows pyAMPACT to connect
    symbolic events in a score to observed sounds in an audio recording. Increasing
    the `num_harmonics` will also include that many harmonics of a notated score
    pitch in the mask. Note that the first harmonic is the fundamental frequency
    which is why the `num_harmonics` parameter defaults to 1. The `width` parameter
    controls how broad or "blurry" the mask is compared to the notated score.

    :param winms: Integer, default 100. The window size in milliseconds.
    :param sample_rate: Integer, default 2000. The sample rate in Hz.
    :param num_harmonics: Integer, default 1. The number of harmonics to use.
    :param width: Integer, default 0. The width of the mask.
    :param bpm: Integer, default 60. The beats per minute to use for sampling.
    :param aFreq: Integer, default 440. The frequency of A4 in Hz.
    :param base_note: Integer, default 0. The base MIDI note to use.
    :param tuning_factor: Float, default 1. The tuning factor to use.
    :param obs: Integer, default 24. The desired observations per second.
    :return: A DataFrame representing the mask. Each row corresponds to a
        frequency bin, and each column corresponds to a timepoint in the
        sampled score. The values are 1 for a note onset and 0 otherwise.

    See Also
    --------
    :meth:`pianoRoll`
    :meth:`sampled`
    """
    key = ('mask', winms, sample_rate, num_harmonics,
           width, bpm, aFreq, base_note, tuning_factor)

    data = piece._analyses


    # Function to replace the index with ONSET_SEC and reset the multi-index
    def replace_index_with_onset_sec(df):
        if isinstance(df, pd.DataFrame):  # If it's a DataFrame
            if 'ONSET_SEC' in df.columns:
                # Set 'ONSET_SEC' as index and reset it without keeping the old index
                df.set_index('ONSET_SEC', inplace=True)
                return df  # No need to reset index, 'ONSET_SEC' is now the index
        elif isinstance(df, pd.Series):  # If it's a Series
            return df.reset_index(drop=False)  # Reset index for series
        return df

    # Apply the transformation
    processed_data = {}
    if piece.fileExtension == 'csv':

        # Process each entry in the dictionary
        processed_data['tony_csv'] = replace_index_with_onset_sec(
            data['tony_csv'])
        processed_data[('_parts', False, False, False, False)] = replace_index_with_onset_sec(
            data['_parts', False, False, False, False])
        processed_data[('durations', True)] = replace_index_with_onset_sec(
            data[('durations', True)])
        processed_data[('midiPitches', True)] = replace_index_with_onset_sec(
            data[('midiPitches', True)])
        processed_data[('_measures', False)] = replace_index_with_onset_sec(
            data[('_measures', False)])
        processed_data['xmlIDs'] = replace_index_with_onset_sec(
            data['xmlIDs'])

        # Process the nmats dictionary similarly
        processed_data[('nmats', None, False)] = {
            key: replace_index_with_onset_sec(df) for key, df in data[('nmats', None, False)].items()
        }

        piece._analyses = processed_data

    if key not in piece._analyses:
        width_semitone_factor = 2 ** ((width / 2) / 12)
        sampled_pr = sampled(piece, bpm, obs)
        num_rows = int(
            2 ** round(math.log(winms / 1000 * sample_rate) / math.log(2) - 1)) + 1
        mask = pd.DataFrame(index=range(
            num_rows), columns=sampled_pr.columns).infer_objects(copy=False).fillna(0)
        fftlen = 2**round(math.log(winms / 1000 *
                                   sample_rate) / math.log(2))

        for row in range(base_note, sampled_pr.shape[0]):
            note = base_note + row
            # MIDI note to Hz: MIDI 69 = 440 Hz = A4
            freq = tuning_factor * \
                (2 ** (note / 12)) * aFreq / (2 ** (69 / 12))
            if sampled_pr.loc[row, :].sum() > 0:
                mcol = pd.Series(0, index=range(num_rows))
                for harm in range(1, num_harmonics + 1):
                    minbin = math.floor(
                        harm * freq / width_semitone_factor / sample_rate * fftlen)
                    maxbin = math.ceil(
                        harm * freq * width_semitone_factor / sample_rate * fftlen)
                    if minbin <= num_rows:
                        maxbin = min(maxbin, num_rows)
                        mcol.loc[minbin: maxbin] = 1
                mask.iloc[np.where(mcol)[0], np.where(
                    sampled_pr.iloc[row])[0]] = 1
        piece._analyses[key] = mask

    return piece._analyses[key]

def jsonCDATA(piece, json_path):
    """
    Return a dictionary of pandas DataFrames, one for each voice. These
    DataFrames contain the cdata from the JSON file designated in `json_path`
    with each nested key in the JSON object becoming a column name in the
    DataFrame. The outermost keys of the JSON cdata will become the "absolute"
    column. While the columns are different, there are as many rows in these
    DataFrames as there are in those of the nmats DataFrames for each voice.

    :param json_path: Path to a JSON file containing cdata.
    :return: A dictionary of pandas DataFrames, one for each voice.

    See Also
    --------
    :meth:`fromJSON`
    :meth:`insertAudioAnalysis`
    :meth:`nmats`
    :meth:`xmlIDs`
    """
    key = ('jsonCDATA', json_path)
    if key not in piece._analyses:
        nmats = nmats(piece, json_path=json_path, include_cdata=True)
        cols = ['ONSET_SEC', *next(iter(nmats.values())).columns[7:]]
        post = {}
        for partName, df in nmats.items():
            res = df[cols].copy()
            res.rename(columns={'ONSET_SEC': 'absolute'}, inplace=True)
            post[partName] = res
        piece._analyses[key] = post
    return piece._analyses[key]

def insertAudioAnalysis(piece, output_path, data, mimetype='', target='', mei_tree=None):
    """
    Insert a <performance> element into the MEI score given the analysis data
    (`data`) in the format of a json file or an nmat dictionary with audio data
    already included. If the original score is not an MEI file, a new MEI file
    will be created and used. The JSON data will be extracted via the `.nmats()`
    method. If provided, the `mimetype` and `target` get passed as
    attributes to the <avFile> element. The performance element will nest
    the DataFrame data in the <performance> element as a child of  <music>
    and a sibling of <body>. A new file will be saved to the
    `output_filename` in the current working directory.

    .. parsed-literal::

        <music>
            <performance xml:id="pyAMPACT-1">
                <recording xml:id="pyAMPACT-2">
                    <avFile mimetype="audio/aiff" target="song.wav" xml:id="pyAMPACT-3" />
                    <when absolute="00:00:12:428" xml:id="pyAMPACT-4" data="#note_1">
                        <extData xml:id="pyAMPACT-5">
                            <![CDATA[>
                                {"ppitch":221.30926295063591, "jitter":0.7427361, ...}
                            ]]>
                        </extData>
                    </when>
                    <when absolute="00:00:12:765" xml:id="pyAMPACT-6" data="#note_2">
                    ...
                </recording>
            </performance>
            <body>
                ...
            </body>
        </music>

    :param output_filename: The name of the output file.
    :param data: Path to a JSON file containing analysis data or an nmats dictionary.
    :param mimetype: Optional MIME type to be set as an attribute to the <avFile>
        element.
    :param target: Optional target to be set as an attribute to the <avFile>
        element.
    :param mei_tree: Optional ElementTree object to use as the base for the new file.
    :return: None but a new file is written

    See Also
    --------
    :meth:`nmats`
    :meth:`toKern`
    :meth:`toMEI`
    """
    performance = ET.Element('performance', {'xml:id': next(idGen)})
    recording = ET.SubElement(performance, 'recording', {
        'xml:id': next(idGen)})
    avFile = ET.SubElement(recording, 'avFile', {'xml:id': next(idGen)})
    if mimetype:
        avFile.set('mimetype', mimetype)
    if target:
        avFile.set('target', target)
    if isinstance(data, dict):   # this is the case for nmats
        jsonCDATA = data
        for part_name, part_df in jsonCDATA.items():
            for i, ndx in enumerate(part_df.index):
                when = ET.SubElement(recording, 'when', {
                    'absolute': part_df.at[ndx, 'ONSET_SEC'], 'xml:id': next(idGen), 'data': f'#{ndx}'})
                extData = ET.SubElement(
                    when, 'extData', {'xml:id': next(idGen)})
                extData.text = f' <![CDATA[ {json.dumps(part_df.iloc[i, 1:].to_dict())} ]]> '
    else:
        jsonCDATA = jsonCDATA(piece, data)
    if mei_tree is None:
        if piece._meiTree is None:
            toMEI(piece, )   # this will save the MEI tree to piece._meiTree
        mei_tree = piece._meiTree
    musicEl = mei_tree.find('.//music')
    musicEl.insert(0, performance)
    # if not output_path.endswith('.mei.xml'):
    #     output_path = output_path.split('.', 1)[0] + '.mei.xml'

    indentMEI(piece._meiTree.getroot())
    # get header/xml descriptor from original file
    lines = []
    if piece.path.endswith('.mei.xml') or piece.path.endswith('.mei'):
        with open(piece.path, 'r') as f:
            for line in f:
                if '<mei ' in line:
                    break
                lines.append(line)
        header = ''.join(lines)
    else:
        convert_attribs_to_str(piece._meiTree.getroot())
        xml_string = ET.tostring(
            piece._meiTree.getroot(), encoding='unicode')
        score_lines = xml_string.split('\n')
        for line in score_lines:
            if '<mei ' in line:
                break
            lines.append(line)
    header = ''.join(lines)
    with open(f'{output_path}', 'w') as f:
        f.write(header)
        ET.ElementTree(piece._meiTree.getroot()).write(
            f, encoding='unicode')

def show(piece, start=None, stop=None):
    """
    Print a VerovioHumdrumViewer link to the score in between the `start` and
    `stop` measures (inclusive).

    :param start: Optional integer representing the starting measure. If `start`
        is greater than `stop`, they will be swapped.
    :param stop: Optional integer representing the last measure.
    :return: None but a url is printed out

    See Also
    --------
    :meth:`toKern`
    """
    if isinstance(start, int) and isinstance(stop, int) and start > stop:
        start, stop = stop, start
    tk = toKern(piece, )
    if start and start > 1:
        header = tk[:tk.index('\n=') + 1]
        headerColCount = header.rsplit('\n', 1)[-1].count('\t')
        startIndex = tk.index(f'={start}')
        fromStart = tk[startIndex:]
        fromStartColCount = fromStart.split('\n', 1)[0].count('\t')
        # add the last divisi line to try to get the column count right
        if fromStartColCount > headerColCount:
            divisi = [fromStart]
            firstLines = tk[:startIndex - 1].split('\n')
            for line in reversed(firstLines):
                if '*^' in line:
                    divisi.append(line)
                    if fromStartColCount - len(divisi) < headerColCount:
                        break
            fromStart = '\n'.join(reversed(divisi))
        tk = header + fromStart
    if stop and stop + 1 < _measures(piece, ).iloc[:, 0].max():
        tk = tk[:tk.index(f'={stop + 1}')]
    encoded = base64.b64encode(tk.encode()).decode()
    if len(encoded) > 2000:
        print(f'''\nAt {len(encoded)} characters, this excerpt is too long to be passed in a url. Instead,\
        \n to see the whole score you can run .toKern("your_file_name"), then drag and drop\
        \nthat file to VHV: https://verovio.humdrum.org/''')
    else:
        print(f'https://verovio.humdrum.org/?t={encoded}')

def toKern(piece, path_name='', data=None, include_lyrics=True, include_dynamics=True, analysis_dfs=None):
    """
    Create a kern representation of the score. If no `path_name` variable is
    passed, then returns a pandas DataFrame of the kern representation.
    Otherwise a file is created or overwritten at the `path_name` path. If
    path_name does not end in '.krn' then this file extension will be added
    to the path. If `lyrics` is `True` (default) then the lyrics for each part
    will be added to the output, if there are lyrics. The same applies to
    `dynamics`.

    :param path_name: Optional string representing the path to save the kern
        file.
    :param data: Optional string representing the data to be converted to kern
        format.
    :param lyrics: Boolean, default True. If True, lyrics for each part will
        be added.
    :param dynamics: Boolean, default True. If True, dynamics for each part
        will be added.
    :return: String of new kern score if no `path_name` is given, or None if
        writing the new kern file to the location of `path_name`

    See Also
    --------
    :meth:`show`
    """
    piece_lyrics = lyrics
    piece_dynamics = dynamics

    key = ('toKern',
           data,
           include_lyrics,
           include_dynamics,
           bool(analysis_dfs))

    if key not in piece._analyses:

        me = _measures(piece).map(
            lambda cell: f'={cell}-' if cell == 0 else f'={cell}',
            na_action='ignore'
        )

        events = kernNotes(piece)

        # ── Collapse exploded kern spines → original chord-token structure ──
        # kernNotes returns one column per exploded voice (Part-1, Part-2 …).
        # _kernSpineGroups maps each original spine to its exploded output indices,
        # e.g. [[0],[1],[2,3,4],[5]] for **function **harte **kern(3-voice) **kern(1-voice).
        # Non-kern original spines produce NO Part columns; only **kern ones do.
        # Strategy: the total indices across all groups minus len(piece.partNames)
        # gives the count of leading non-kern indices to skip. The remaining
        # indices map 1-to-1 onto piece.partNames in order.
        # IMPORTANT: piece.partNames is NEVER mutated — only the local events df.
        _sg = getattr(piece, '_kernSpineGroups', None)
        if _sg:
            _all_idxs = [i for g in _sg for i in g]
            _n_skip = len(_all_idxs) - len(piece.partNames)
            _kern_idxs = _all_idxs[_n_skip:]
            # music21 reads kern spines RIGHT-TO-LEFT: the last spine becomes Part-1.
            # So kern_idxs [2,3,4,5] maps REVERSED onto partNames:
            # idx 5 (rightmost) -> Part-1, idx 2 (leftmost) -> Part-4.
            _n_kern = len(_kern_idxs)
            _idx_to_part = {eidx: piece.partNames[_n_kern - 1 - pos]
                            for pos, eidx in enumerate(_kern_idxs)}
            _sibling_cols = []
            for _group in _sg:
                _pcols = [_idx_to_part[i] for i in _group if i in _idx_to_part]
                if len(_pcols) <= 1:
                    continue
                _primary, _siblings = _pcols[0], _pcols[1:]
                import re as _re_mg
                _REST_PAT = _re_mg.compile(r'^\d+\.?r$')
                def _mg(row, p=_primary, sibs=_siblings):
                    toks = []
                    for c in [p] + sibs:
                        t = str(row[c])
                        if t in ('.', 'nan') or _REST_PAT.match(t):
                            continue
                        toks.append(t)
                    return ' '.join(toks) if toks else '.'
                events[_primary] = events.apply(_mg, axis=1)
                _sibling_cols.extend(_siblings)
            if _sibling_cols:
                events = events.drop(columns=_sibling_cols)
            # Reorder remaining columns so highest Part-N (bass, leftmost in file)
            # comes first and lowest Part-N (treble, rightmost in file) comes last.
            # The _cols loop below iterates reversed, so the last column here
            # becomes the first column in _cols → rightmost spine in output → top staff.
            def _part_num(col_name):
                try: return int(col_name.split('-')[1])
                except: return 0
            events = events[sorted(events.columns, key=_part_num, reverse=True)]
        # ── end collapse ─────────────────────────────────────────────────────

        isMI = isinstance(events.index, pd.MultiIndex)

        includeLyrics, includeDynamics = False, False

        if include_lyrics and not piece_lyrics(piece).empty:
            includeLyrics = True
            lyr = piece_lyrics(piece)
            if isMI:
                lyr.index = pd.MultiIndex.from_arrays(
                    (lyr.index, [0] * len(lyr.index))
                )

        if include_dynamics and not piece_dynamics(piece).empty:
            includeDynamics = True
            dyn = piece_dynamics(piece)
            if isMI:
                dyn.index = pd.MultiIndex.from_arrays(
                    (dyn.index, [0] * len(dyn.index))
                )

        _cols = []
        firstTokens = []
        partNumbers = []
        staves = []
        instruments = []
        partNames = []
        shortNames = []

        # Build a mapping from collapsed column name → output spine number.
        # When kern spines were exploded and collapsed, piece.partNames still has
        # the exploded names (Part-1..Part-N). We want sequential numbering of
        # the collapsed kern spines in their output order (left to right = 1, 2…).
        # events.columns after collapse = primaries in ascending Part-N order.
        # The _cols loop reverses them, so output spine 1 = events.columns[-1], etc.
        _collapsed_kern_cols = list(events.columns)  # e.g. [Part-4, Part-1]
        _n_collapsed = len(_collapsed_kern_cols)
        # Output order (reversed): _collapsed_kern_cols[-1], …, _collapsed_kern_cols[0]
        # Spine number in output = position in that reversed list + 1
        _col_to_spine_num = {
            col: _n_collapsed - i
            for i, col in enumerate(_collapsed_kern_cols)
        }

        for i in range(len(events.columns), 0, -1):

            col = events.columns[i - 1]
            _cols.append(events[col])

            partNum = _col_to_spine_num.get(col, i)

            firstTokens.append('**kern')
            partNumbers.append(f'*part{partNum}')
            staves.append(f'*staff{partNum}')
            instruments.append('*Ivox')
            # Use a clean spine-number label when kern was exploded/collapsed,
            # otherwise use the internal Part-N name (unexploded scores).
            _spine_label = f'Part {partNum}' if _sg else col
            partNames.append(f'*I"{_spine_label}')
            shortNames.append(f"*I'{_spine_label[0]}")

            if includeLyrics and col in lyr.columns:
                lyrCol = lyr[col]
                lyrCol.name = 'Text_' + lyrCol.name
                _cols.append(lyrCol)
                firstTokens.append('**text')
                partNumbers.append(f'*part{partNum}')
                staves.append(f'*staff{partNum}')
                instruments.append('*')
                partNames.append('*')
                shortNames.append('*')

            if includeDynamics and col in dyn.columns:
                dynCol = dyn[col]
                dynCol.name = 'Dynam_' + dynCol.name
                _cols.append(dynCol)
                firstTokens.append('**dynam')
                partNumbers.append(f'*part{partNum}')
                staves.append(f'*staff{partNum}')
                instruments.append('*')
                partNames.append('*')
                shortNames.append('*')

        events = pd.concat(_cols, axis=1)

        # ------------------------
        # ADD ANALYSIS SPINES
        # ------------------------

        if analysis_dfs is not None:

            if isinstance(analysis_dfs, pd.DataFrame):
                analysis_dfs = {'analysis': analysis_dfs}

            for tag, df in analysis_dfs.items():

                _df = df.copy()
                _df = _df.reindex(events.index)

                _df = _df.map(
                    lambda x: '.' if pd.isna(x) else str(x)
                )

                for col in _df.columns:
                    firstTokens.append(f'**{col}')
                    partNumbers.append('*')
                    staves.append('*')
                    instruments.append('*')
                    partNames.append('*')
                    shortNames.append('*')

                addTieBreakers((events, _df))
                events = pd.concat((events, _df), axis=1)

        # ------------------------

        ba = _barlines(piece)
        ba = ba[ba != 'regular'].dropna().replace(
            {'double': '||', 'final': '=='}
        )
        ba.loc[piece.score.highestTime, :] = '=='

        me = pd.concat([me.iloc[:, 0]] * len(events.columns), axis=1)
        ba = pd.concat([ba.iloc[:, 0]] * len(events.columns), axis=1)

        me.columns = events.columns
        ba.columns = events.columns

        ds = piece._analyses['_divisiStarts']
        ds = ds.reindex(events.columns, axis=1).fillna('*')

        de = piece._analyses['_divisiEnds']
        de = de.reindex(events.columns, axis=1).fillna('*')

        clefs = _clefs(piece).reindex(events.columns, axis=1).fillna('*')
        ts = '*M' + _timeSignatures(piece)
        ts = ts.reindex(events.columns, axis=1).fillna('*')
        ks = _keySignatures(piece).reindex(events.columns, axis=1).fillna('*')

        partTokens = pd.DataFrame(
            [
                firstTokens,
                partNumbers,
                staves,
                instruments,
                partNames,
                shortNames,
                ['*-'] * len(events.columns)
            ],
            index=[
                -12, -11, -10, -9, -8, -7,
                int(piece.score.highestTime + 1)
            ]
        ).fillna('*')

        partTokens.columns = events.columns

        to_concat = [
            partTokens,
            de,
            me,
            ds,
            clefs,
            ks,
            ts,
            events,
            ba
        ]

        if isinstance(events.index, pd.MultiIndex):
            addTieBreakers(to_concat)

        body = pd.concat(to_concat).sort_index(kind='mergesort')

        if isinstance(body.index, pd.MultiIndex):
            body = body.droplevel(1)

        body = body.fillna('.')

        result = [kernHeader(piece.metadata)]

        result.extend(
            body.apply(
                lambda row: '\t'.join(
                    row.dropna().astype(str)
                ),
                axis=1
            )
        )

        result.extend((kernFooter(piece.fileExtension),))

        result = '\n'.join(result)

        piece._analyses[key] = result

    if not path_name:
        return piece._analyses[key]

    if not path_name.endswith('.krn'):
        path_name += '.krn'

    with open(path_name, 'w') as f:
        f.write(piece._analyses[key])

def _meiStack(piece):
    """
    Return a DataFrame stacked to be a multi-indexed series containing the
    score elements to be processed into the MEI format. This is used  for
    MEI output. Only used internally.

    :return: A Series of the score in MEI format

    See Also
    --------
    :meth:`toMEI`
    """
    if '_meiStack' not in piece._analyses:
        # assign column names in format (partNumber, voiceNumer) with no splitting up of chords
        events = _parts(piece, compact=True, number=True).copy()
        clefs = _m21Clefs(piece, ).copy()
        ksigs = _keySignatures(piece, False).copy()
        tsigs = _timeSignatures(piece, ratio=False).copy()
        mi = pd.MultiIndex.from_tuples([(x, 1) for x in range(
            1, len(clefs.columns) + 1)], names=['Staff', 'Layer'])
        for i, staffInfo in enumerate((clefs, ksigs, tsigs)):
            if 0.0 in staffInfo.index:
                staffInfo.drop(0.0, inplace=True)
            staffInfo.index = pd.MultiIndex.from_product(
                [staffInfo.index, [i - 9]])
            staffInfo.columns = mi

        me = _measures(piece, compact=True)
        me.columns = events.columns
        parts = []
        for i, partName in enumerate(events.columns):
            ei = events.iloc[:, i]
            mi = me.iloc[:, i]
            mi.name = 'Measure'
            addTieBreakers((ei, mi))
            if partName in clefs.columns:
                ci = clefs.loc[:, partName].dropna()
                ki = ksigs.loc[:, partName].dropna()
                ti = tsigs.loc[:, partName].dropna()
                ei = pd.concat((ci, ki, ti, ei)).sort_index()
            # force measures to come before any grace notes. # TODO: check case of nachschlag grace notes
            mi.index = mi.index.set_levels([-10], level=1)
            part = pd.concat((ei, mi), axis=1)
            part = part.dropna(how='all').sort_index(level=[0, 1])
            part.Measure = part.Measure.ffill()
            parts.append(part.set_index('Measure', append=True))
        df = pd.concat(parts, axis=1).sort_index().droplevel([0, 1])
        df.columns = events.columns
        stack = df.stack((0, 1), future_stack=True).dropna(
        ).sort_index(level=[0, 1, 2])
        piece._analyses['_meiStack'] = stack
    return piece._analyses['_meiStack']

def _coreMEIElements(piece):
    root = ET.Element(
        'mei', {'xmlns': 'http://www.music-encoding.org/ns/mei', 'meiversion': '5.1-dev'})

    meiHead = ET.SubElement(root, 'meiHead')
    fileDesc = ET.SubElement(meiHead, 'fileDesc')
    titleStmt = ET.SubElement(fileDesc, 'titleStmt')
    title = ET.SubElement(titleStmt, 'title')
    title.text = piece.metadata['title']
    composer = ET.SubElement(titleStmt, 'composer')
    composer.text = piece.metadata['composer']
    pubStmt = ET.SubElement(fileDesc, 'pubStmt')
    unpub = ET.SubElement(pubStmt, 'unpub')
    unpub.text = f'This mei file was converted from a .{piece.fileExtension} file by pyAMPACT'
    music = ET.SubElement(root, 'music')
    # insert performance element here
    body = ET.SubElement(music, 'body')
    mdiv = ET.SubElement(body, 'mdiv')
    score = ET.SubElement(mdiv, 'score')
    section = ET.SubElement(score, 'section')
    insertScoreDef(piece, root)
    return root

def toMEI(piece, file_name='', indentation='\t', data='', start=None, stop=None, dfs=None, analysis_tag='annot'):
    """
    Write or return an MEI score optionally including analysis data.

    If no `file_name` is passed then returns a string of the MEI representation.
    Otherwise a file called `file_name` is created or overwritten in the current working
    directory. If `file_name` does not end in '.mei.xml' or '.mei', then the `.mei.xml`
    file extension will be added to the `file_name`.

    :param file_name: Optional string representing the name to save the new
        MEI file to the current working directory.
    :param data: Optional string of the path of score data in json format to
        be added to the the new mei file.
    :param start: Optional integer representing the starting measure. If `start`
        is greater than `stop`, they will be swapped.
    :param stop: Optional integer representing the last measure.
    :param dfs: Optional dictionary of pandas DataFrames to be added to the
        new MEI file. The keys of the dictionary will be used as the `@type`
        attribute of the `analysis_tag` parameter element.
    :param analysis_tag: Optional string representing the name of the tag to
        be used for the analysis data.
    :return: String of new MEI score if no `file_name` is given, or None if
        writing the new MEI file to the current working directory.

    See Also
    --------
    :meth:`toKern`
    """
    key = ('toMEI', data, start, stop)
    if isinstance(dfs, pd.DataFrame):
        dfs = {'analysis': dfs}
    if key not in piece._analyses:
        root = _coreMEIElements(piece, )
        section = root.find('.//section')
        stack = _meiStack(piece, )
        if isinstance(start, int) or isinstance(stop, int):
            stack = stack.copy()
        if isinstance(start, int) and isinstance(stop, int) and start > stop:
            start, stop = stop, start
        if isinstance(start, int):
            stack = stack.loc[start:]
        if isinstance(stop, int):
            stack = stack.loc[:stop]
        uniqueStaves = stack.index.get_level_values(1).unique()
        uniqueLayers = stack.index.get_level_values(2).unique()
        for measure in stack.index.get_level_values(0).unique():
            meas_el = ET.SubElement(
                section, 'measure', {'n': f'{measure}'})
            for staff in uniqueStaves:
                staff_el = ET.SubElement(
                    meas_el, 'staff', {'n': f'{staff}'})
                for layer in uniqueLayers:
                    if (measure, staff, layer) not in stack.index:
                        continue
                    layer_el = ET.SubElement(
                        staff_el, 'layer', {'n': f'{layer}'})
                    parent = layer_el
                    for el in stack.loc[[(measure, staff, layer)]].values:
                        if hasattr(el, 'beams') and el.beams.beamsList and el.beams.beamsList[0].type == 'start':
                            parent = ET.SubElement(
                                layer_el, 'beam', {'xml:id': next(idGen)})
                        if hasattr(el, 'isNote') and el.isNote:
                            addMEINote(el, parent)
                        elif hasattr(el, 'isRest') and el.isRest:
                            rest_el = ET.SubElement(parent, 'rest', {'xml:id': f'{el.id}',
                                                                     'dur': duration2MEI[el.duration.type], 'dots': f'{el.duration.dots}'})
                        elif hasattr(el, 'isChord') and el.isChord:
                            chord_el = ET.SubElement(parent, 'chord')
                            for note in el.notes:
                                addMEINote(note, chord_el)
                        if hasattr(el, 'expressions'):
                            for exp in el.expressions:
                                if exp.name == 'fermata':
                                    ferm_el = ET.SubElement(meas_el, 'fermata',
                                                            {'xml:id': next(idGen), 'startid': parent[-1].get('xml:id')})
                        if hasattr(el, 'getSpannerSites'):
                            for spanner in el.getSpannerSites():
                                if isinstance(spanner, m21.spanner.Slur) and el == spanner[0]:
                                    ET.SubElement(meas_el, 'slur', {'xml:id': next(idGen),
                                                                    'startid': f'{el.id}', 'endid': f'{spanner.getLast().id}'})
                        if hasattr(el, 'beams') and el.beams.beamsList and el.beams.beamsList[0].type == 'stop':
                            parent = layer_el
                            continue

                        if isinstance(el, m21.clef.Clef):
                            clef_el = ET.SubElement(parent, 'clef', {'xml:id': next(
                                idGen), 'shape': el.sign, 'line': f'{el.line}'})
                        elif isinstance(el, m21.meter.TimeSignature):
                            attrs_el = ET.SubElement(parent, 'attributes', {
                                'xml:id': next(idGen)})
                            tsig_el = ET.SubElement(
                                attrs_el, 'time', {'xml:id': next(idGen)})
                            numerator_el = ET.SubElement(tsig_el, 'beats')
                            numerator_el.text = f'{el.numerator}'
                            denominator_el = ET.SubElement(
                                tsig_el, 'beatType')
                            denominator_el.text = f'{el.denominator}'
                        elif isinstance(el, m21.key.KeySignature):
                            score_def_el = ET.Element(
                                'scoreDef', {'xml:id': next(idGen)})
                            key_sig_el = ET.SubElement(score_def_el, 'keySig', {
                                'xml:id': next(idGen)})
                            if el.sharps >= 0:
                                key_sig_el.set('sig', f'{el.sharps}s')
                            else:
                                key_sig_el.set('sig', f'{abs(el.sharps)}f')
                            section.insert(len(section) - 1, score_def_el)

        indentMEI(root, indentation)
        convert_attribs_to_str(root)
        piece._analyses[key] = ET.ElementTree(root)
        if piece._meiTree is None:
            piece._meiTree = piece._analyses[key]

    if dfs is None:
        ret = piece._analyses[key]
    else:   # add analysis data
        ret = deepcopy(piece._analyses[key])
        if any((start, stop)):
            for measure in ret.findall('.//measure'):
                measure_number = int(measure.get('n'))
                if (start and measure_number < start) or (stop and measure_number > stop):
                    measure.getparent().remove(measure)
        events = _parts(piece, compact=True, number=True)
        for ii, (tag, df) in enumerate(dfs.items()):
            _df = contextualize(piece, 
                df, offsets=True, measures=True, beats=True)
            _df.columns = events.columns[:len(_df.columns)]
            if any((start, stop)):   # trim _df to start and stop
                if start and stop:
                    _df = _df.loc[idx[:, start:stop, :]]
                elif start:
                    _df = _df.loc[idx[:, start:, :]]
                else:
                    _df = _df.loc[idx[:, :stop, :]]
            dfstack = _df.stack((0, 1), future_stack=True).dropna()
            for measure in dfstack.index.get_level_values(1).unique():
                meas_el = ret.find(f'.//measure[@n="{measure}"]')
                if not meas_el:
                    continue
                for ndx in dfstack.index:
                    if ndx[1] > measure:
                        break
                    if ndx[1] < measure:
                        continue
                    val = dfstack.at[ndx]
                    properties = {'xml:id': next(idGen), 'type': tag, 'tstamp': f'{ndx[2]}',
                                  'staff': f'{ndx[3]}', 'layer': f'{ndx[4]}'}
                    if ndx[4] % 2 == 1 and ii % 2 == 0:
                        properties['place'] = 'below'
                    else:
                        properties['place'] = 'above'
                    analysis_el = ET.SubElement(
                        meas_el, analysis_tag, properties)
                    analysis_el.text = f'{val}'
        newRoot = ret.getroot()
        indentMEI(newRoot, indentation)
        ret = ET.ElementTree(newRoot)

    if not file_name:
        return ret
    else:
        if file_name.endswith('.mei'):
            file_name += '.xml'
        elif not file_name.endswith('.mei.xml'):
            file_name += '.mei.xml'
        with open(f'./{file_name}', 'w') as f:
            f.write(meiDeclaration)
            ret.write(f, encoding='unicode')

def build_mask_from_nmat_seconds(nmat, sample_rate, num_harmonics, width, tres, n_freqs):
    """
    Build a binary harmonic mask matrix from a note matrix for use in DTW alignment.

    Parameters
    ----------
    nmat : pd.DataFrame
        Note matrix dictionary keyed by part name. 

    sample_rate : int
        Sample rate in Hz used to determine the Nyquist frequency

    num_harmonics : int
        Number of harmonic partials (including the fundamental)

    width : int
        Base half-width in frequency bins around each partial centre.

    tres : float
        Time resolution in seconds per column of the mask matrix. Should match
        the hop size used in the corresponding STFT.

    n_freqs : int
        Number of frequency bins in the mask (i.e. number of rows), matching
        the frequency axis of the spectrogram it will be compared against.

    Returns
    -------
    M : np.ndarray, shape (n_freqs, n_cols)
        The harmonic mask matrix as a float32 array
    """
    # nmat is dict of dfs
    events = []
    for _, df in nmat.items():
        if not {"ONSET_SEC", "OFFSET_SEC", "MIDI"} <= set(df.columns):
            continue
        for ons, off, midi in zip(df["ONSET_SEC"].values, df["OFFSET_SEC"].values, df["MIDI"].values):
            if pd.isna(ons) or pd.isna(off) or pd.isna(midi):
                continue
            ons = float(ons); off = float(off)
            if off <= ons:
                # Fix zero-duration notes by adding minimum duration
                off = ons + 0.01  # 10ms minimum
            m = float(midi)
            if m < 0:
                continue
            f0 = 440.0 * (2.0 ** ((m - 69.0) / 12.0))
            events.append((ons, off, f0))

    if not events:
        raise ValueError("No events found in nmat for mask construction")

    total_dur = max(e[1] for e in events)
    times = np.arange(0.0, total_dur + tres, tres)

    nyquist = sample_rate / 2.0
    freqs = np.linspace(0.0, nyquist, int(n_freqs))

    M = np.zeros((len(freqs), len(times)), dtype=np.float32)

    for start, end, f0 in events:
        t0 = int(np.floor(start / tres))
        t1 = int(np.ceil(end / tres))
        t0 = max(0, min(t0, len(times) - 1))
        t1 = max(0, min(t1, len(times) - 1))

        # Ensure we have at least one time frame for very short notes
        if t1 <= t0:
            t1 = t0 + 1

        for h in range(1, int(num_harmonics) + 1):
            f = f0 * h
            if f >= nyquist:
                break
            fi = int(np.argmin(np.abs(freqs - f)))

            # Adaptive width based on frequency (wider at higher frequencies)
            adaptive_width = max(1, int(width * (1 + f / 2000.0)))
            lo = max(0, fi - adaptive_width)
            hi = min(len(freqs), fi + adaptive_width + 1)

            # Apply harmonic weighting (fundamental stronger than harmonics)
            harmonic_weight = 1.0 / h

            # Gaussian-like weighting within the frequency band for smoother mask
            for freq_idx in range(lo, hi):
                freq_distance = abs(freq_idx - fi)
                freq_weight = np.exp(-0.5 * (freq_distance / adaptive_width) ** 2)
                M[freq_idx, t0:t1 + 1] += harmonic_weight * freq_weight

    # Apply light temporal smoothing to reduce alignment sensitivity
    try:
        from scipy import ndimage
        M = ndimage.gaussian_filter(M, sigma=(0.3, 0.3))
    except ImportError:
        # Fallback if scipy is not available
        pass

    # Normalize with better dynamic range preservation
    mx = float(M.max())
    if mx > 0:
        # Use square root normalization to preserve relative intensities better
        M = np.sqrt(M / mx)

    return M
