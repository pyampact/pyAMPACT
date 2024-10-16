"""
symbolic
==============


.. autosummary::
    :toctree: generated/

    Score

"""

from pyampact.symbolicUtils import *
from fractions import Fraction
import tempfile
import mido
import xml.etree.ElementTree as ET
import requests
import re
import pandas as pd
import numpy as np
import json
import librosa
import tempfile
from .symbolicUtils import *
import os
import ast
import base64
from copy import deepcopy
import math
import music21 as m21
m21.environment.set('autoDownload', 'allow')

# utils
idx = pd.IndexSlice

# Comment for package build

# Uncomment for package build
__all__ = [
    "Score",
    "_assignM21Attributes",
    "_partList",
    "_parts",
    "_import_other_spines",
    "insertScoreDef",
    "xmlIDs",
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
    "_timeSignatures",
    "durations",
    "midi_ticks_durations",
    "midiPitches",
    "notes",
    "kernNotes",
    "nmats",
    "pianoRoll",
    "sampled",
    "mask",
    "jsonCDATA",
    "insertAudioAnalysis",
    "show",
    "toKern",
    "_meiStack",
    "toMEI"
]


def convert_attribs_to_str(element):
    for key in element.attrib:
        if isinstance(element.attrib[key], np.float64):
            element.attrib[key] = str(element.attrib[key])
    for child in element:
        convert_attribs_to_str(child)


class Score:
    """
    A class to import a score via music21 and expose pyAMPACT's analysis utilities.

    The analysis utilities are generally formatted as pandas dataframes. This 
    class also ports over some matlab code to help with alignment of scores in 
    symbolic notation and audio analysis of recordings of those scores. `Score` 
    objects can insert analysis into an MEI file, and can export any type of 
    file to a kern format, optionally also including analysis from a JSON file. 
    Similarly, `Score` objects can serve clickable URLs of short excerpts of 
    their associated score in symbolic notation. These links open in the Verovio 
    Humdrum Viewer.

    :param score_path: A string representing the path to the score file.
    :return: A Score object.

    Example
    --------
    .. code-block:: python

        url_or_path = 'https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn'
        piece = Score(url_or_path)
    """

    def __init__(self, score_path):
        self._analyses = {}
        if score_path.startswith('https://github.com/'):
            score_path = 'https://raw.githubusercontent.com/' + \
                score_path[19:].replace('/blob/', '/', 1)
        self.path = score_path
        self.fileName = score_path.rsplit('.', 1)[0].rsplit('/')[-1]
        self.fileExtension = score_path.rsplit(
            '.', 1)[1] if '.' in score_path else ''
        self.partNames = []
        self.score = None
        self._meiTree = None
        if score_path.startswith('http') and self.fileExtension == 'krn':
            fd, tmp_path = tempfile.mkstemp()
            try:
                with os.fdopen(fd, 'w') as tmp:
                    response = requests.get(self.path)
                    tmp.write(response.text)
                    tmp.seek(0)
                    self._assignM21Attributes(tmp_path)
                    self._import_other_spines(tmp_path)
            finally:
                os.remove(tmp_path)
        # file is not an online kern file (can be either or neither but not both)
        else:
            self._assignM21Attributes()
            self._import_other_spines()
        self.public = '\n'.join([f'{prop.ljust(15)}{type(getattr(self, prop))}' for prop in dir(
            self) if not prop.startswith('_')])
        self._partList()

    def _assignM21Attributes(self, path=''):
        """
        Assign music21 attributes to a given object.

        :param obj: A music21 object.
        :return: None
        """
        if self.path not in imported_scores:
            if path:   # parse humdrum files differently to extract their function, and harm spines if they have them
                imported_scores[self.path] = m21.converter.parse(
                    path, format='humdrum')
            # these files might be mei files and could lack elements music21 needs to be able to read them
            elif self.fileExtension in ('xml', 'musicxml', 'mei', 'mxl'):
                if self.path.startswith('http'):
                    tree = ET.ElementTree(ET.fromstring(
                        requests.get(self.path).text))
                else:
                    tree = ET.parse(self.path)
                remove_namespaces(tree)
                root = tree.getroot()
                hasFunctions = False
                _functions = root.findall('.//function')
                if len(_functions):
                    hasFunctions = True

                # this is an mei file even if the fileExtension is .xml
                if root.tag.endswith('mei'):
                    parseEdited = False
                    self._meiTree = deepcopy(root)
                    # this mei file doesn't have a scoreDef element, so construct one and add it to the score
                    if not root.find('.//scoreDef'):
                        parseEdited = True
                        self.insertScoreDef(root)

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
                        imported_scores[self.path] = m21.converter.subConverters.ConverterMEI(
                        ).parseData(mei_string)
                        parseEdited = False

                if hasFunctions:   # not an mei file, but an xml file that had functions
                    try:
                        imported_scores[self.path] = m21.converter.parse(
                            self.path)
                    except m21.harmony.HarmonyException:
                        print(
                            'There was an issue with the function texts so they were removed.')
                        for _function in _functions:
                            _function.text = ''
                        xml_string = ET.tostring(root, encoding='unicode')
                        imported_scores[self.path] = m21.converter.parse(
                            xml_string, format='MusicXML')

            # read file/string as volpiano or tinyNotation if applicable
            elif self.fileExtension in ('', 'txt'):
                temp = None
                text = self.path
                if self.fileExtension == 'txt':
                    if self.path.startswith('http'):
                        text = requests.get(self.path).text
                    else:
                        with open(self.path, 'r') as file:
                            text = file.read()
                if text.startswith('volpiano: ') or re.match(volpiano_pattern, text):
                    temp = m21.converter.parse(text, format='volpiano')
                elif text.startswith('tinyNotation: ') or re.match(tinyNotation_pattern, text):
                    temp = m21.converter.parse(text, format='tinyNotation')
                if temp is not None:
                    _score = m21.stream.Score()
                    _score.insert(0, temp)
                    imported_scores[self.path] = _score

            elif self.fileExtension == 'csv':   # read csv file as a pandas DataFrame with no header and no index
                csv_df = pd.read_csv(self.path, header=None, index_col=False)
                if len(csv_df.columns) != 3:
                    print(
                        "This csv file is not a 3-column Tony file so it can't be imported.")
                    return
                else:  # assume it's a Tony file if it has 3 columns
                    # sometimes these files only have two columns instead of three
                    csv_df.columns = ('ONSET_SEC', 'AVG PITCH IN HZ', 'DURATION')[
                        :len(csv_df.columns)]
                    csv_df['MIDI'] = csv_df['AVG PITCH IN HZ'].map(
                        lambda freq: librosa.hz_to_midi(freq)).round().astype('Int16')
                    self._analyses['tony_csv'] = csv_df
                    _parts = pd.DataFrame(csv_df['MIDI'])
                    _parts.columns = ['Part-1']
                    self._analyses[('_parts', False, False,
                                    False, False)] = _parts
                    dur = pd.DataFrame(csv_df['DURATION'])
                    dur.columns = ['Part-1']
                    dur.index = pd.MultiIndex.from_tuples(
                        [(i, 0) for i in dur.index])
                    self._analyses[('durations', True)] = dur
                    midiPitches = pd.DataFrame(csv_df['MIDI'])
                    midiPitches.columns = ['Part-1']
                    midiPitches.index = pd.MultiIndex.from_tuples(
                        [(i, 0) for i in midiPitches.index])
                    self._analyses[('midiPitches', True)] = midiPitches
                    measures = pd.DataFrame([1])
                    measures.columns = ['Part-1']
                    self._analyses[('_measures', False)] = measures
                    xmlIDs = pd.DataFrame([next(idGen)
                                          for x in range(len(csv_df.index))])
                    xmlIDs.columns = ['Part-1']
                    xmlIDs.index = pd.MultiIndex.from_tuples(
                        [(i, 0) for i in xmlIDs.index])
                    self._analyses['xmlIDs'] = xmlIDs
                    self.partNames = ['Part-1']
                    self._analyses['_partList'] = [None]
                    return

        if self.path not in imported_scores:   # check again to catch valid tree files
            if self.path.startswith('http') and self.fileExtension in ('mid', 'midi'):
                midi_bytes = requests.get(self.path).content
                imported_scores[self.path] = m21.converter.parse(midi_bytes)
            else:
                imported_scores[self.path] = m21.converter.parse(self.path)
        self.score = imported_scores[self.path]
        self.metadata = {'title': "Title not found",
                         'composer': "Composer not found"}
        if self.score.metadata is not None:
            self.metadata['title'] = self.score.metadata.title or 'Title not found'
            self.metadata['composer'] = self.score.metadata.composer or 'Composer not found'
        self._partStreams = self.score.getElementsByClass(m21.stream.Part)
        self._flatParts = []
        self.partNames = []
        for i, part in enumerate(self._partStreams):
            flat = part.flatten()
            toRemove = [el for el in flat if el.offset < 0]
            flat.remove(toRemove)
            flat.makeMeasures(inPlace=True)
            flat.makeAccidentals(inPlace=True)
            # you have to flatten again after calling makeMeasures
            self._flatParts.append(flat.flatten())
            name = flat.partName if (
                flat.partName and flat.partName not in self.partNames) else f'Part-{i + 1}'
            self.partNames.append(name)

    def _partList(self):
        """
        Return a list of series of the note, rest, and chord objects in each part.

        :return: A list of pandas Series, each representing a part in the score.
        """
        if '_partList' not in self._analyses:
            kernStrands = []
            parts = []
            isUnique = True
            divisiStarts = []
            divisiEnds = []
            for ii, flat_part in enumerate(self._flatParts):
                graces, graceOffsets = [], []
                notGraces = {}
                for nrc in flat_part.getElementsByClass(['Note', 'Rest', 'Chord']):
                    if nrc.duration.isGrace:
                        graces.append(nrc)
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
                    ser.name = self.partNames[ii]
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
                            ) or self.score.highestTime
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
                part0.name = self.partNames[ii]
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
                        nextNdx = self.score.highestTime if len(
                            part) - 1 == endI else part.index[endI + 1]
                        thisDur = part.iat[endI].quarterLength
                        if abs(nextNdx - endNdx - thisDur) > .00003:
                            strand = part.iloc[startI:endI + 1].copy()
                            strand.name = f'{self.partNames[ii]}__{len(strands) + 1}'
                            divisiStarts.append(pd.Series(
                                ('*^', '*^'), index=(strand.name, self.partNames[ii]), name=part.index[startI], dtype='string'))
                            joinNdx = endNdx + thisDur        # find a suitable endpoint to rejoin this strand
                            divisiEnds.append(pd.Series(('*v', '*v'), index=(
                                strand.name, self.partNames[ii]), name=(strand.name, joinNdx), dtype='string'))
                            strands.append(strand)
                            startI = endI + 1
                kernStrands.extend(
                    sorted(strands, key=lambda _strand: _strand.last_valid_index()))

            self._analyses['_divisiStarts'] = pd.DataFrame(
                divisiStarts).fillna('*').sort_index()
            de = pd.DataFrame(divisiEnds)
            if not de.empty:
                de = de.reset_index(level=1)
                de = de.reindex(
                    [prt.name for prt in kernStrands if prt.name not in self.partNames]).set_index('level_1')
            self._analyses['_divisiEnds'] = de
            if not isUnique:
                addTieBreakers(parts)
                addTieBreakers(kernStrands)
            self._analyses['_partList'] = parts
            self._analyses['_kernStrands'] = kernStrands
        return self._analyses['_partList']

    def _parts(self, multi_index=False, kernStrands=False, compact=False,
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
        if key not in self._analyses:
            toConcat = []
            if kernStrands:
                toConcat = self._analyses['_kernStrands']
            elif compact:
                toConcat = self._partList()
                if number:
                    partNameToNum = {part: i + 1 for i,
                                     part in enumerate(self.partNames)}
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
                for part in self._partList():
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
                toConcat) else pd.DataFrame(columns=self.partNames)
            if not multi_index and isinstance(df.index, pd.MultiIndex):
                df.index = df.index.droplevel(1)
            if compact and number:
                df.columns = mi
            self._analyses[key] = df
        return self._analyses[key]

    def _import_other_spines(self, path=''):
        """
        Import the harmonic function spines from a given path.

        :param path: A string representing the path to the file containing the 
            harmonic function spines.
        :return: A pandas DataFrame representing the harmonic function spines.
        """
        if self.fileExtension == 'krn' or path:
            humFile = m21.humdrum.spineParser.HumdrumFile(path or self.path)
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

                df1 = self._priority()
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
                if spine.spineType not in self._analyses:
                    self._analyses[spine.spineType] = [res]
                else:
                    self._analyses[spine.spineType].append(res)
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
                    self._analyses[keyName] = ser
                    gotKeys = True
            if len(foundSpines):
                self.foundSpines = foundSpines

        for spine in ('function', 'harm', 'keys', 'chord'):
            if spine not in self._analyses:
                self._analyses[spine] = pd.Series(dtype='string')
        if 'cdata' not in self._analyses:
            self._analyses['cdata'] = pd.DataFrame()

    def insertScoreDef(self, root):
        """
        Insert a scoreDef element into an MEI (Music Encoding Initiative) document.

        This function inserts a scoreDef element into an MEI document if one is
        not already present. It modifies the input element in-place.

        :param root: An xml.etree.ElementTree.Element representing the root of the
            MEI document.
        :return: None
        """
        if root.find('.//scoreDef') is None:
            if self.score is not None:
                clefs = self._m21Clefs()
                ksigs = self._keySignatures(False)
                tsigs = self._timeSignatures(False)
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
            if not len(self.partNames):
                self.partNames = sorted(
                    {f'Part-{staff.attrib.get("n")}' for staff in root.iter('staff')})
            for i, staff in enumerate(self.partNames):
                attribs = {'label': staff, 'n': str(
                    i + 1), 'xml:id': next(idGen), 'lines': '5'}
                if self.score is not None:
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

    def xmlIDs(self):
        """
        Return xml ids per part in a pandas.DataFrame time-aligned with the
        objects offset. If the file is not xml or mei, or an idString wasn't found,
        return a DataFrame of the ids of the music21 objects.

        :return: A pandas DataFrame representing the xml ids in the score.

        See Also
        --------
        :meth:`nmats`

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            piece.xmlIDs()
        """
        if 'xmlIDs' in self._analyses:
            return self._analyses['xmlIDs']
        if self.fileExtension in ('xml', 'mei'):
            tree = ET.parse(self.path)
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
                parts = self._parts(multi_index=True).copy()
                for i in range(len(parts.columns)):
                    part = parts.iloc[:, i].dropna()
                    idCol = ids.iloc[:, i].dropna()
                    idCol.index = part.index
                    cols.append(idCol)
                df = pd.concat(cols, axis=1)
                df.columns = parts.columns
                self._analyses['xmlIDs'] = df
                return df
        # either not xml/mei, or an idString wasn't found
        df = self._parts(multi_index=True).map(
            lambda obj: f'{obj.id}', na_action='ignore')
        self._analyses['xmlIDs'] = df
        return df

    def _lyricHelper(self, cell, strip):
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

    def lyrics(self, strip=True):
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

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/Busnoys_In_hydraulis.krn')
            piece.lyrics()
        """
        key = ('lyrics', strip)
        if key not in self._analyses:
            df = self._parts().map(self._lyricHelper, na_action='ignore',
                                   **{'strip': strip}).dropna(how='all')
            self._analyses[key] = df
        return self._analyses[key].copy()

    def _m21Clefs(self):
        """
        Extract the clefs from the score. 

        The clefs are extracted from each part and returned as a pandas DataFrame 
        where each column represents a part and each row represents a clef. The 
        DataFrame is indexed by the offset of the clefs.

        :return: A pandas DataFrame of the clefs in the score in music21's format.
        """
        if '_m21Clefs' not in self._analyses:
            parts = []
            isUnique = True
            for i, flat_part in enumerate(self._flatParts):
                ser = pd.Series(flat_part.getElementsByClass(
                    ['Clef']), name=self.partNames[i])
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
            self._analyses['_m21Clefs'] = clefs
        return self._analyses['_m21Clefs']

    def _clefs(self):
        """
        Extract the clefs from the score. 

        The clefs are extracted from each part and returned as a pandas DataFrame 
        where each column represents a part and each row represents a clef. The 
        DataFrame is indexed by the offset of the clefs.

        :return: A pandas DataFrame of the clefs in the score in kern format.
        """
        if '_clefs' not in self._analyses:
            self._analyses['_clefs'] = self._m21Clefs().map(
                kernClefHelper, na_action='ignore')
        return self._analyses['_clefs']

    def dynamics(self):
        """
        Extract the dynamics from the score. 

        The dynamics are extracted from each part and returned as a pandas DataFrame 
        where each column represents a part and each row represents a dynamic 
        marking. The DataFrame is indexed by the offset of the dynamic markings.

        :return: A pandas DataFrame representing the dynamics in the score.

        See Also
        --------
        :meth:`lyrics`

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            piece.dynamics()
        """
        if 'dynamics' not in self._analyses:
            dyns = [pd.Series({obj.offset: obj.value for obj in sf.getElementsByClass(
                'Dynamic')}, dtype='string') for sf in self._flatParts]
            dyns = pd.concat(dyns, axis=1)
            dyns.columns = self.partNames
            dyns.dropna(how='all', axis=1, inplace=True)
            self._analyses['dynamics'] = dyns
        return self._analyses['dynamics'].copy()

    def _priority(self):
        """
        For .krn files, get the line numbers of the events in the piece, which 
        music21 often calls "priority". For other encoding formats return an 
        empty dataframe.

        :return: A DataFrame containing the priority values.
        """
        if '_priority' not in self._analyses:
            if self.fileExtension != 'krn':
                priority = pd.DataFrame()
            else:
                # use compact to avoid losing priorities of chords
                parts = self._parts(compact=True)
                if parts.empty:
                    priority = pd.DataFrame()
                else:
                    priority = parts.map(lambda cell: cell.priority, na_action='ignore').ffill(
                        axis=1).iloc[:, -1].astype('Int16')
                    priority = pd.DataFrame(
                        {'Priority': priority.values, 'Offset': priority.index})
            self._analyses['_priority'] = priority
        return self._analyses['_priority']

    def keys(self, snap_to=None, filler='forward', output='array'):
        """
        Get the key signature portion of the **harm spine in a kern file if there
        is one and return it as an array or a time-aligned pandas Series. This is
        similar to the .harm, .functions, .chords, and .cdata methods. The default
        is for the results to be returned as a 1-d array, but you can set `output='series'`
        for a pandas series instead. If want to get the results of a different spine
        type (i.e. not one of the ones listed above), see :meth:`getSpines`.

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            piece.keys()

        If you want to align these results so that they match the columnar (time) axis
        of the pianoRoll, sampled, or mask results, you can pass the pianoRoll or mask
        that you want to align to as the `snap_to` parameter. Doing that makes it easier
        to combine these results with any of the pianoRoll, sampled, or mask tables to
        have both in a single table which can make data analysis easier. Passing a `snap_to`
        argument will automatically cause the return value to be a pandas series since
        that's facilitates combining the two. Here's how you would use the `snap_to`
        parameter and then combine the results with the pianoRoll to create a single table.

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            pianoRoll = piece.pianoRoll()
            keys = piece.keys(snap_to=pianoRoll)
            combined = pd.concat((pianoRoll, keys))

        The `sampled` and `mask` dfs often have more observations than the spine 
        contents, so you may want to fill in these new empty slots somehow. The kern 
        format uses '.' as a filler token so you can pass this as the `filler` 
        parameter to fill all the new empty slots with this as well. If you choose 
        some other value, say `filler='_'`, then in addition to filling in the empty 
        slots with underscores, this will also replace the kern '.' observations with 
        '_'. If you want to fill them in with NaN's as pandas usually does, you can 
        pass `filler='nan'` as a convenience. If you want to "forward fill" these 
        results, you can pass `filler='forward'` (default). This will propagate the 
        last non-period ('.') observation until a new one is found. Finally, you can 
        pass filler='drop' to drop all empty observations (both NaNs and humdrum
        periods).

        :param snap_to: A pandas DataFrame to align the results to. Default is None.
        :param filler: A string representing the filler token. Default is 'forward'.
        :param output: A string representing the output format. Default is 'array'.
        :return: A numpy array or pandas Series representing the harmonic keys
            analysis.

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
        return snapTo(self._analyses['keys'], snap_to, filler, output)

    def harm(self, snap_to=None, filler='forward', output='array'):
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

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            pianoRoll = piece.pianoRoll()
            harm = piece.harm(snap_to=pianoRoll, output='series')
            combined = pd.concat((pianoRoll, harm))

        The `sampled` and `mask` dfs often have more observations than the spine 
        contents, so you may want to fill in these new empty slots somehow. The kern 
        format uses '.' as a filler token so you can pass this as the `filler` 
        parameter to fill all the new empty slots with this as well. If you choose 
        some other value, say `filler='_'`, then in addition to filling in the empty 
        slots with underscores, this will also replace the kern '.' observations with 
        '_'. If you want to fill them in with NaN's as pandas usually does, you can 
        pass `filler='nan'` as a convenience. If you want to "forward fill" these 
        results, you can pass `filler='forward'` (default). This will propagate the 
        last non-period ('.') observation until a new one is found. Finally, you can 
        pass filler='drop' to drop all empty observations (both NaNs and humdrum
        periods).

        :param snap_to: A pandas DataFrame to align the results to. Default is None.
        :param filler: A string representing the filler token. Default is 'forward'.
        :param output: A string representing the output format. Default is 'array'.
        :return: A numpy array or pandas Series representing the harmonic keys
            analysis.

        See Also
        --------
        :meth:`cdata`
        :meth:`chords`
        :meth:`functions`
        :meth:`keys`
        """
        return snapTo(self._analyses['harm'], snap_to, filler, output)

    def functions(self, snap_to=None, filler='forward', output='array'):
        """
        Get the harmonic function labels from a **function spine in a kern file if there
        is one and return it as an array or a time-aligned pandas Series. This is
        similar to the .harm, .keys, .chords, and .cdata methods. The default
        is for the results to be returned as a 1-d array, but you can set `output='series'`
        for a pandas series instead. If want to get the results of a different spine
        type (i.e. not one of the ones listed above), see :meth:`getSpines`.

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            piece.functions()

        If you want to align these results so that they match the columnar (time) axis
        of the pianoRoll, sampled, or mask results, you can pass the pianoRoll or mask
        that you want to align to as the `snap_to` parameter. Doing that makes it easier
        to combine these results with any of the pianoRoll, sampled, or mask tables to
        have both in a single table which can make data analysis easier. Passing a `snap_to`
        argument will automatically cause the return value to be a pandas series since
        that's facilitates combining the two. Here's how you would use the `snap_to`
        parameter and then combine the results with the pianoRoll to create a single table.

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            pianoRoll = piece.pianoRoll()
            functions = piece.functions(snap_to=pianoRoll)
            combined = pd.concat((pianoRoll, functions))

        The `sampled` and `mask` dfs often have more observations than the spine 
        contents, so you may want to fill in these new empty slots somehow. The kern 
        format uses '.' as a filler token so you can pass this as the `filler` 
        parameter to fill all the new empty slots with this as well. If you choose 
        some other value, say `filler='_'`, then in addition to filling in the empty 
        slots with underscores, this will also replace the kern '.' observations with 
        '_'. If you want to fill them in with NaN's as pandas usually does, you can 
        pass `filler='nan'` as a convenience. If you want to "forward fill" these 
        results, you can pass `filler='forward'` (default). This will propagate the 
        last non-period ('.') observation until a new one is found. Finally, you can 
        pass filler='drop' to drop all empty observations (both NaNs and humdrum
        periods).

        :param snap_to: A pandas DataFrame to align the results to. Default is None.
        :param filler: A string representing the filler token. Default is 'forward'.
        :param output: A string representing the output format. Default is 'array'.
        :return: A numpy array or pandas Series representing the harmonic keys
            analysis.

        See Also
        --------
        :meth:`cdata`
        :meth:`chords`
        :meth:`harm`
        :meth:`keys`
        """
        if snap_to is not None:
            output = 'series'
        return snapTo(self._analyses['function'], snap_to, filler, output)

    def chords(self, snap_to=None, filler='forward', output='array'):
        """
        Get the chord labels from the **chord spine in a kern file if there
        is one and return it as an array or a time-aligned pandas Series. This is
        similar to the .functions, .harm, .keys, and .cdata methods. The default
        is for the results to be returned as a 1-d array, but you can set `output='series'`
        for a pandas series instead. If want to get the results of a different spine
        type (i.e. not one of the ones listed above), see :meth:`getSpines`.

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            piece.chords()

        If you want to align these results so that they match the columnar (time) axis
        of the pianoRoll, sampled, or mask results, you can pass the pianoRoll or mask
        that you want to align to as the `snap_to` parameter. Doing that makes it easier
        to combine these results with any of the pianoRoll, sampled, or mask tables to
        have both in a single table which can make data analysis easier. Passing a `snap_to`
        argument will automatically cause the return value to be a pandas series since
        that's facilitates combining the two. Here's how you would use the `snap_to`
        parameter and then combine the results with the pianoRoll to create a single table.

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            pianoRoll = piece.pianoRoll()
            chords = piece.chords(snap_to=pianoRoll)
            combined = pd.concat((pianoRoll, chords))

        The `sampled` and `mask` dfs often have more observations than the spine 
        contents, so you may want to fill in these new empty slots somehow. The kern 
        format uses '.' as a filler token so you can pass this as the `filler` 
        parameter to fill all the new empty slots with this as well. If you choose 
        some other value, say `filler='_'`, then in addition to filling in the empty 
        slots with underscores, this will also replace the kern '.' observations with 
        '_'. If you want to fill them in with NaN's as pandas usually does, you can 
        pass `filler='nan'` as a convenience. If you want to "forward fill" these 
        results, you can pass `filler='forward'` (default). This will propagate the 
        last non-period ('.') observation until a new one is found. Finally, you can 
        pass filler='drop' to drop all empty observations (both NaNs and humdrum
        periods).

        :param snap_to: A pandas DataFrame to align the results to. Default is None.
        :param filler: A string representing the filler token. Default is 'forward'.
        :param output: A string representing the output format. Default is 'array'.
        :return: A numpy array or pandas Series representing the harmonic keys
            analysis.

        See Also
        --------
        :meth:`cdata`
        :meth:`functions`
        :meth:`harm`
        :meth:`keys`
        """
        if snap_to is not None:
            output = 'series'
        return snapTo(self._analyses['chord'], snap_to, filler, output)

    def cdata(self, snap_to=None, filler='forward', output='dataframe'):
        """
        Get the cdata records from **cdata spines in a kern file if there
        are any and return it as a pandas DataFrame. This is
        similar to the .harm, .functions, .chords, and .keys methods, with the
        exception that this method defaults to returning a dataframe since there are
        often more than one cdata spine in a kern score. If want to get the results
        of a different spine type (i.e. not one of the ones listed above), see
        :meth:`getSpines`.

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            piece.cdata()

        If you want to align these results so that they match the columnar (time) axis
        of the pianoRoll, sampled, or mask results, you can pass the pianoRoll or mask
        that you want to align to as the `snap_to` parameter. Doing that makes it easier
        to combine these results with any of the pianoRoll, sampled, or mask tables to
        have both in a single table which can make data analysis easier. Passing a `snap_to`
        argument will automatically cause the return value to be a pandas series since
        that's facilitates combining the two. Here's how you would use the `snap_to`
        parameter and then combine the results with the pianoRoll to create a single table.

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            pianoRoll = piece.pianoRoll()
            cdata = piece.cdata(snap_to=pianoRoll)
            combined = pd.concat((pianoRoll, cdata))

        The `sampled` and `mask` dfs often have more observations than the spine 
        contents, so you may want to fill in these new empty slots somehow. The kern 
        format uses '.' as a filler token so you can pass this as the `filler` 
        parameter to fill all the new empty slots with this as well. If you choose 
        some other value, say `filler='_'`, then in addition to filling in the empty 
        slots with underscores, this will also replace the kern '.' observations with 
        '_'. If you want to fill them in with NaN's as pandas usually does, you can 
        pass `filler='nan'` as a convenience. If you want to "forward fill" these 
        results, you can pass `filler='forward'` (default). This will propagate the 
        last non-period ('.') observation until a new one is found. Finally, you can 
        pass filler='drop' to drop all empty observations (both NaNs and humdrum
        periods).

        :param snap_to: A pandas DataFrame to align the results to. Default is None.
        :param filler: A string representing the filler token. Default is 'forward'.
        :param output: A string representing the output format. Default is 'array'.
        :return: A numpy array or pandas Series representing the harmonic keys
            analysis.

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
        return snapTo(self._analyses['cdata'], snap_to, filler, output)

    def getSpines(self, spineType):
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

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/O_Virgo_Miserere.krn')
            rdiss = piece.getSpines('cdata-rdiss')

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
        if hasattr(self, 'foundSpines') and spineType in self.foundSpines:
            ret = snapTo(self._analyses[spineType],
                         filler='nan', output='dataframe')
            ret.dropna(how='all', inplace=True)
            if len(ret.columns) == len(self.partNames):
                ret.columns = self.partNames
            return ret
        if self.fileExtension != 'krn':
            print(f'\t***This is not a kern file so there are no spines to import.***')
        else:
            print(f'\t***No {spineType} spines were found.***')

    def dez(self, path=''):
        """
        Get the labels data from a .dez file/url and return it as a dataframe. Calls
        fromJSON to do this. The "meta" portion of the dez file is ignored. If no
        path is provided, the last dez table imported with this method is returned.

        :param path: A string representing the path to the .dez file.
        :return: A pandas DataFrame representing the labels in the .dez file.
        """
        if 'dez' not in self._analyses:
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
                self._analyses['dez'] = {path: fromJSON(path)}
        else:
            if not path:   # return the last dez table
                return next(reversed(self._analyses['dez'].values()))
            else:
                if path not in self._analyses['dez']:
                    self._analyses['dez'][path] = fromJSON(path)
        return self._analyses['dez'][path]

    def form(self, snap_to=None, filler='forward', output='array', dez_path=''):
        """
        Get the "Structure" labels from a .dez file/url and return it as an array or a
        time-aligned pandas Series. The default is for the results to be returned as a 1-d
        array, but you can set `output='series'` for a pandas series instead. If you want to align
        these results so that they match the columnar (time) axis of the pianoRoll, sampled, or
        mask results, you can pass the pianoRoll or mask that you want to align to as the `snap_to`
        parameter. Doing that makes it easier to combine these results with any of the pianoRoll,
        sampled, or mask tables to have both in a single table which can make data analysis easier.
        This example shows how to get the form analysis from a .dez file.

        Example
        -------
        .. code-block:: python

            piece = Score('test_files/K279-1.krn')
            form = piece.form(dez_path='test_files/K279-1_harmony_texture.dez')
        """
        if not dez_path and 'dez' not in self._analyses:
            print('No .dez file was found.')
        else:
            dez = self.dez(dez_path)
            df = dez.set_index('start').rename_axis(None)
            df = df.loc[(df['type'] == 'Structure'), 'tag']
            if df.empty:
                print('No "Structure" analysis was found in the .dez file.')
            else:
                return snapTo(df, snap_to, filler, output)

    def romanNumerals(self, snap_to=None, filler='forward', output='array', dez_path=''):
        """
        Get the roman numeral labels from a .dez file/url or **harm spine and return it as an array
        or a time-aligned pandas Series. The default is for the results to be returned as a 1-d
        array, but you can set `output='series'` for a pandas series instead. If you want to align
        these results so that they match the columnar (time) axis of the pianoRoll, sampled, or mask
        results, you can pass the pianoRoll or mask that you want to align to as the `snap_to` parameter.
        Doing that makes it easier to combine these results with any of the pianoRoll, sampled, or mask
        tables to have both in a single table which can make data analysis easier. This example shows
        how to get the roman numeral analysis from a kern score that has a **harm spine.

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            pianoRoll = piece.pianoRoll()
            romanNumerals = piece.romanNumerals(snap_to=pianoRoll)

        The next example shows how to get the roman numeral analysis from a .dez file.

        Example
        -------
        .. code-block:: python

            piece = Score('test_files/K279-1.krn')
            romanNumerals = piece.romanNumerals(dez_path='test_files/K279-1_harmony_texture.dez')
        """
        if dez_path or 'dez' in self._analyses:
            dez = self.dez(dez_path)
            if dez is not None:
                df = dez.set_index('start').rename_axis(None)
                df = df.loc[(df['type'] == 'Harmony'), 'tag']
                if df.empty:
                    print(
                        'No "Harmony" analysis was found in the .dez file, checking for a **harm spine.')
                else:
                    return snapTo(df, snap_to, filler, output)
        if 'harm' in self._analyses and len(self._analyses['harm']):
            return self.harm(snap_to=snap_to, filler=filler, output=output)
        print('Neither a dez nor a **harm spine was found so using music21 to get roman numerals...')
        key = self.score.analyze('key')
        chords = self.score.chordify().recurse().getElementsByClass('Chord')
        offsets = [ch.offset for ch in chords]
        figures = [m21.roman.romanNumeralFromChord(
            ch, key).figure for ch in chords]
        ser = pd.Series(figures, index=offsets, name='Roman Numerals')
        ser = ser[ser != ser.shift()]   # remove consecutive duplicates
        return snapTo(ser, snap_to, filler, output)

    def _m21ObjectsNoTies(self):
        """
        Remove tied notes in a given voice. Only the first note in a tied group 
        will be kept.

        :param voice: A music21 stream Voice object.
        :return: A list of music21 objects with ties removed.
        """
        if '_m21ObjectsNoTies' not in self._analyses:
            self._analyses['_m21ObjectsNoTies'] = self._parts(
                multi_index=True).map(removeTied).dropna(how='all')
        return self._analyses['_m21ObjectsNoTies']

    def _measures(self, compact=False):
        """
        Return a DataFrame of the measure starting points.

        :param compact: Boolean, default False. If True, the method will keep
            chords unified rather then expanding them into separate columns.
        :return: A DataFrame where each column corresponds to a part in the score,
            and each row index is the offset of a measure start. The values are 
            the measure numbers.
        """
        if ('_measures', compact) not in self._analyses:
            partCols = self._parts(compact=compact).columns
            partMeasures = []
            for i, part in enumerate(self._flatParts):
                meas = {m.offset: m.measureNumber for m in part.makeMeasures(
                ) if isinstance(m, m21.stream.Measure)}
                ser = [pd.Series(meas, dtype='Int16')]
                voiceCount = len(
                    [col for col in partCols if col.startswith(self.partNames[i])])
                partMeasures.extend(ser * voiceCount)
            df = pd.concat(partMeasures, axis=1)
            df.columns = partCols
            self._analyses[('_measures', compact)] = df
        return self._analyses[('_measures', compact)].copy()

    def _barlines(self):
        """
        Return a DataFrame of barlines specifying which barline type.

        Double barline, for example, can help detect section divisions, and the 
        final barline can help process the `highestTime` similar to music21.

        :return: A DataFrame where each column corresponds to a part in the score,
            and each row index is the offset of a barline. The values are the 
            barline types.
        """
        if "_barlines" not in self._analyses:
            partBarlines = [pd.Series({bar.offset: bar.type for bar in part.getElementsByClass(['Barline'])})
                            for i, part in enumerate(self._flatParts)]
            df = pd.concat(partBarlines, axis=1, sort=True)
            df.columns = self.partNames
            self._analyses["_barlines"] = df
        return self._analyses["_barlines"]

    def _keySignatures(self, kern=True):
        """
        Return a DataFrame of key signatures for each part in the score.

        :param kern: Boolean, default True. If True, the key signatures are 
            returned in the **kern format.
        :return: A DataFrame where each column corresponds to a part in the score,
            and each row index is the offset of a key signature. The values are 
            the key signatures.
        """
        if ('_keySignatures', kern) not in self._analyses:
            kSigs = []
            for i, part in enumerate(self._flatParts):
                kSigs.append(pd.Series({ky.offset: ky for ky in part.getElementsByClass(
                    ['KeySignature'])}, name=self.partNames[i]))
            df = pd.concat(kSigs, axis=1).sort_index(kind='mergesort')
            if kern:
                df = '*k[' + df.map(lambda ky: ''.join(
                    [_note.name for _note in ky.alteredPitches]).lower(), na_action='ignore') + ']'
            self._analyses[('_keySignatures', kern)] = df
        return self._analyses[('_keySignatures', kern)]

    def _timeSignatures(self, ratio=True):
        """
        Return a DataFrame of time signatures for each part in the score.

        :return: A DataFrame where each column corresponds to a part in the score,
            and each row index is the offset of a time signature. The values are 
            the time signatures in ratio string format.
        """
        if ('_timeSignatures', ratio) not in self._analyses:
            tsigs = []
            for i, part in enumerate(self._flatParts):
                if not ratio:
                    tsigs.append(pd.Series(
                        {ts.offset: ts for ts in part.getTimeSignatures()}, name=self.partNames[i]))
                else:
                    tsigs.append(pd.Series(
                        {ts.offset: ts.ratioString for ts in part.getTimeSignatures()}, name=self.partNames[i]))
            df = pd.concat(tsigs, axis=1).sort_index(kind='mergesort')
            self._analyses[('_timeSignatures', ratio)] = df
        return self._analyses[('_timeSignatures', ratio)]

    def durations(self, multi_index=False, df=None):
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

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            piece.durations()
        """
        if df is None:
            key = ('durations', multi_index)
            if key not in self._analyses:
                m21objs = self._m21ObjectsNoTies()
                res = m21objs.map(lambda nrc: nrc.quarterLength,
                                  na_action='ignore').astype(float).round(5)
                if not multi_index and isinstance(res.index, pd.MultiIndex):
                    res = res.droplevel(1)
                self._analyses[key] = res
            return self._analyses[key]
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
                    vals.append(self.score.highestTime - ndx[-1])
                sers.append(pd.Series(vals, part.index, dtype='float64'))
            res = pd.concat(sers, axis=1, sort=True)
            if not multi_index and isinstance(res.index, pd.MultiIndex):
                res = res.droplevel(1)
            res.columns = df.columns
            return res

    def midi_ticks_durations(self, i=1, df=None):
        """
        Replaces the placeholder ONSET_SEC and OFFSET_SEC columns with specific placements calculated by MIDI
        tick information. The method translates the music21 stream to MIDI and replaces the values accordingly

        See Also
        --------
        :meth:`notes`
            Return a DataFrame of the newly calculated ONSET_SEC and OFFSET_SEC times, and DURATION

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            piece.durations()
        """

        # Convert music21 stream to MIDI
        i = 1
        midi_file = []
        mf = m21.midi.translate.music21ObjectToMidiFile(self.score)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as temp_midi_file:
            mf.open(temp_midi_file.name, 'wb')
            mf.write()
            mf.close()
            midi_file = temp_midi_file.name
        mid = mido.MidiFile(midi_file)
        # Logic from durations_from_midi_ticks
        onsOffsList = []

        # Default PPQN and tempo values
        ppqn = 9600
        current_tempo = 500000

        # Check for tempo metadata in the MIDI file
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    current_tempo = msg.tempo
                    break  # No need to check further messages in the same track

        # Convert ticks per beat to seconds per tick
        seconds_per_tick = current_tempo / (1_000_000 * ppqn)

        # Check if index is out of range
        if i >= len(mid.tracks):
            print(
                f"Index {i} is out of range. Total number of tracks is {len(mid.tracks)}.")
            return df

        # for track in mid.tracks:
        cum_time = 0

        for msg in mid.tracks[i]:
            if msg.type == 'end_of_track':
                break  # Stop processing this track upon encountering end_of_track
            cum_time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                note = msg.note
                velocity = msg.velocity
                start_time = cum_time * seconds_per_tick
                onsOffsList.append([start_time, 0])

            if msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                for event in reversed(onsOffsList):
                    if event[1] == 0 and event[0] <= cum_time * seconds_per_tick:
                        end_time = cum_time * seconds_per_tick
                        event[1] = end_time
                        break
        # Replace values in DataFrame
        onsOffsList = truncate_and_scale_onsOffsList(
            onsOffsList, len(df.index))

        res = pd.DataFrame(
            onsOffsList, columns=['ONSET_SEC', 'OFFSET_SEC'], index=df.index)

        # Update the columns with new values from `res`
        df['ONSET_SEC'] = res['ONSET_SEC']
        df['OFFSET_SEC'] = res['OFFSET_SEC']
        df['DURATION'] = (res['ONSET_SEC'] - res['OFFSET_SEC']) * -1

        return df

    def contextualize(self, df, offsets=True, measures=True, beats=True):
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
            meas = self._measures().iloc[:, 0]
            meas.name = 'Measure'
            toConcat.append(meas)
            cols.append('Measure')
        if beats:
            bts = self._beats().apply(
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

    def _beats(self):
        """
        Return a DataFrame of beat numbers for each part in the score.

        :return: A DataFrame where each column corresponds to a part in the score,
            and each row index is the offset of a beat. The values are the beat
            numbers.
        """
        if '_beats' not in self._analyses:
            df = self._parts(compact=True).map(
                lambda obj: obj.beat, na_action='ignore')
            self._analyses['_beats'] = df
        return self._analyses['_beats']

    def midiPitches(self, multi_index=False):
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

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            piece.midiPitches()
        """
        key = ('midiPitches', multi_index)
        if key not in self._analyses:
            midiPitches = self._m21ObjectsNoTies().map(
                lambda nr: -1 if nr.isRest else nr.pitch.midi, na_action='ignore')
            if not multi_index and isinstance(midiPitches.index, pd.MultiIndex):
                midiPitches = midiPitches.droplevel(1)
            self._analyses[key] = midiPitches
        return self._analyses[key]

    def notes(self, combine_rests=True, combine_unisons=False):
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

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            piece.notes()
        """
        if 'notes' not in self._analyses:
            df = self._m21ObjectsNoTies().map(noteRestHelper, na_action='ignore')
            self._analyses['notes'] = df
        ret = self._analyses['notes'].copy()
        if combine_rests:
            ret = ret.apply(combineRests)
        if combine_unisons:
            ret = ret.apply(combineUnisons)
        if isinstance(ret.index, pd.MultiIndex):
            ret = ret.droplevel(1)
        return ret

    def kernNotes(self):
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

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            piece.kernNotes()
        """
        if 'kernNotes' not in self._analyses:
            self._analyses['kernNotes'] = self._parts(
                True, True).map(kernNRCHelper, na_action='ignore')
        return self._analyses['kernNotes']

    def nmats(self, json_path=None, include_cdata=False):
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

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/Mozart_K179_seg.krn')
            piece.nmats()
        """
        if not json_path:   # user must pass a json_path if they want the cdata to be included
            include_cdata = False
        key = ('nmats', json_path, include_cdata)
        if key not in self._analyses:
            nmats = {}
            included = {}
            dur = self.durations(multi_index=True)
            mp = self.midiPitches(multi_index=True)
            ms = self._measures()
            ids = self.xmlIDs()
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
            for i, partName in enumerate(self._parts().columns):
                meas = ms.iloc[:, i]
                midi = mp.iloc[:, i].dropna()
                onsetBeat = pd.Series(midi.index.get_level_values(
                    0), index=midi.index, dtype='float64')
                durBeat = dur.iloc[:, i].dropna()
                part = pd.Series(partName, midi.index, dtype='string')
                xmlID = ids.iloc[:, i].dropna()
                if self.fileExtension == 'csv':
                    csv_data = self._analyses['tony_csv']
                    csv_data.index = part.index
                    onsetSec = csv_data['ONSET_SEC']
                    offsetSec = csv_data['ONSET_SEC'] + csv_data['DURATION']
                else:
                    onsetSec = onsetBeat.copy()  # This is overwritten by midiTicks function
                    offsetSec = onsetBeat + durBeat  # This is overwritten by midiTicks function
                df = pd.concat([meas, onsetBeat, durBeat, part, midi,
                               onsetSec, offsetSec, xmlID], axis=1, sort=True)
                df.columns = ['MEASURE', 'ONSET', 'DURATION', 'PART',
                              'MIDI', 'ONSET_SEC', 'OFFSET_SEC', 'XML_ID']
                df['MEASURE'] = df['MEASURE'].ffill()
                df.dropna(how='all', inplace=True, subset=df.columns[1:5])
                df = df.set_index('XML_ID')

                # Remove rows where MIDI == -1.0
                df = df[df['MIDI'] != -1.0]

                df = self.midi_ticks_durations(i+1, df)
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
            self._analyses[('nmats', json_path, False)] = nmats
            if json_path:
                self._analyses[('nmats', json_path, True)] = included
        return self._analyses[key]

    def pianoRoll(self):
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

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            piece.pianoRoll()
        """
        if 'pianoRoll' not in self._analyses:
            mp = self.midiPitches()
            # remove non-last offset repeats and forward-fill
            mp = mp[~mp.index.duplicated(keep='last')].ffill()
            pianoRoll = pd.DataFrame(index=range(128), columns=mp.index.values)
            for offset in mp.index:
                for pitch in mp.loc[offset]:
                    if pitch >= 0:
                        pianoRoll.at[pitch, offset] = 1
            pianoRoll = pianoRoll.infer_objects(copy=False).fillna(0)
            self._analyses['pianoRoll'] = pianoRoll
        return self._analyses['pianoRoll']

    def sampled(self, bpm=60, obs=24):
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

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            piece.sampled()
        """
        key = ('sampled', bpm, obs)
        if key not in self._analyses:
            slices = 60/bpm * obs
            timepoints = pd.Index(
                [t/slices for t in range(0, int(self.score.highestTime * slices))])
            pr = self.pianoRoll().copy()
            pr.columns = [col if col in timepoints else timepoints.asof(
                col) for col in pr.columns]
            pr = pr.T
            pr = pr.iloc[~pr.index.duplicated(keep='last')]
            pr = pr.T
            sampled = pr.reindex(columns=timepoints, method='ffill')
            self._analyses[key] = sampled
        return self._analyses[key]

    def mask(self, winms=100, sample_rate=2000, num_harmonics=1, width=0,
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

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            piece.mask()
        """
        key = ('mask', winms, sample_rate, num_harmonics,
               width, bpm, aFreq, base_note, tuning_factor)
        if key not in self._analyses:
            width_semitone_factor = 2 ** ((width / 2) / 12)
            sampled = self.sampled(bpm, obs)
            num_rows = int(
                2 ** round(math.log(winms / 1000 * sample_rate) / math.log(2) - 1)) + 1
            mask = pd.DataFrame(index=range(
                num_rows), columns=sampled.columns).infer_objects(copy=False).fillna(0)
            fftlen = 2**round(math.log(winms / 1000 *
                              sample_rate) / math.log(2))

            for row in range(base_note, sampled.shape[0]):
                note = base_note + row
                # MIDI note to Hz: MIDI 69 = 440 Hz = A4
                freq = tuning_factor * \
                    (2 ** (note / 12)) * aFreq / (2 ** (69 / 12))
                if sampled.loc[row, :].sum() > 0:
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
                        sampled.iloc[row])[0]] = 1
            self._analyses[key] = mask
        return self._analyses[key]

    def jsonCDATA(self, json_path):
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

        Example
        -------
        .. code-block:: python

            piece = Score('./test_files/CloseToYou.mei.xml')
            piece.jsonCDATA(json_path='./test_files/CloseToYou.json')
        """
        key = ('jsonCDATA', json_path)
        if key not in self._analyses:
            nmats = self.nmats(json_path=json_path, include_cdata=True)
            cols = ['ONSET_SEC', *next(iter(nmats.values())).columns[7:]]
            post = {}
            for partName, df in nmats.items():
                res = df[cols].copy()
                res.rename(columns={'ONSET_SEC': 'absolute'}, inplace=True)
                post[partName] = res
            self._analyses[key] = post
        return self._analyses[key]

    def insertAudioAnalysis(self, output_path, data, mimetype='', target='', mei_tree=None):
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
            If this is not passed, then the original MEI file is used if the Score
            is an MEI file. Otherwise a new MEI file is created with .toMEI().
        :return: None but a new file is written

        See Also
        --------
        :meth:`nmats`
        :meth:`toKern`
        :meth:`toMEI`

        Example
        -------
        .. code-block:: python

            piece = Score('./test_files/CloseToYou.mei.xml')
            piece.insertAudioAnalysis(output_filename='newfile.mei.xml'
                data='./test_files/CloseToYou.json',
                mimetype='audio/aiff',
                target='Close to You vocals.wav')
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
            jsonCDATA = self.jsonCDATA(data)
        if mei_tree is None:
            if self._meiTree is None:
                self.toMEI()   # this will save the MEI tree to self._meiTree
            mei_tree = self._meiTree
        musicEl = mei_tree.find('.//music')
        musicEl.insert(0, performance)
        # if not output_path.endswith('.mei.xml'):
        #     output_path = output_path.split('.', 1)[0] + '.mei.xml'

        indentMEI(self._meiTree.getroot())
        # get header/xml descriptor from original file
        lines = []
        if self.path.endswith('.mei.xml') or self.path.endswith('.mei'):
            with open(self.path, 'r') as f:
                for line in f:
                    if '<mei ' in line:
                        break
                    lines.append(line)
            header = ''.join(lines)
        else:
            convert_attribs_to_str(self._meiTree.getroot())
            xml_string = ET.tostring(
                self._meiTree.getroot(), encoding='unicode')
            score_lines = xml_string.split('\n')
            for line in score_lines:
                if '<mei ' in line:
                    break
                lines.append(line)
        header = ''.join(lines)
        with open(f'{output_path}', 'w') as f:
            f.write(header)
            ET.ElementTree(self._meiTree.getroot()).write(
                f, encoding='unicode')

    def show(self, start=None, stop=None):
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

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
            piece.show(5, 10)
        """
        if isinstance(start, int) and isinstance(stop, int) and start > stop:
            start, stop = stop, start
        tk = self.toKern()
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
        if stop and stop + 1 < self._measures().iloc[:, 0].max():
            tk = tk[:tk.index(f'={stop + 1}')]
        encoded = base64.b64encode(tk.encode()).decode()
        if len(encoded) > 2000:
            print(f'''\nAt {len(encoded)} characters, this excerpt is too long to be passed in a url. Instead,\
            \n to see the whole score you can run .toKern("your_file_name"), then drag and drop\
            \nthat file to VHV: https://verovio.humdrum.org/''')
        else:
            print(f'https://verovio.humdrum.org/?t={encoded}')

    def toKern(self, path_name='', data='', lyrics=True, dynamics=True):
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

        Example
        -------
        .. code-block:: python

            # create a kern file from a different symbolic notation file
            piece = Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/K179.xml')
            piece.toKern()
        """
        key = ('toKern', data)
        if key not in self._analyses:
            me = self._measures().map(
                lambda cell: f'={cell}-' if cell == 0 else f'={cell}', na_action='ignore')
            events = self.kernNotes()
            isMI = isinstance(events.index, pd.MultiIndex)
            includeLyrics, includeDynamics = False, False
            if lyrics and not self.lyrics().empty:
                includeLyrics = True
                lyr = self.lyrics()
                if isMI:
                    lyr.index = pd.MultiIndex.from_arrays(
                        (lyr.index, [0]*len(lyr.index)))
            if dynamics and not self.dynamics().empty:
                includeDynamics = True
                dyn = self.dynamics()
                if isMI:
                    dyn.index = pd.MultiIndex.from_arrays(
                        (dyn.index, [0]*len(dyn.index)))
            _cols, firstTokens, partNumbers, staves, instruments, partNames, shortNames = [
            ], [], [], [], [], [], []
            # reverse column order because kern order is lowest staves on the left
            for i in range(len(events.columns), 0, -1):
                col = events.columns[i - 1]
                _cols.append(events[col])
                partNum = self.partNames.index(
                    col) + 1 if col in self.partNames else -1
                firstTokens.append('**kern')
                partNumbers.append(f'*part{partNum}')
                staves.append(f'*staff{partNum}')
                instruments.append('*Ivox')
                partNames.append(f'*I"{col}')
                shortNames.append(f"*I'{col[0]}")
                if includeLyrics and col in lyr.columns:
                    lyrCol = lyr[col]
                    lyrCol.name = 'Text_' + lyrCol.name
                    _cols.append(lyrCol)
                    firstTokens.append('**text')
                    partNumbers.append(f'*part{partNum}')
                    staves.append(f'*staff{partNum}')
                if includeDynamics and col in dyn.columns:
                    dynCol = dyn[col]
                    dynCol.name = 'Dynam_' + dynCol.name
                    _cols.append(dynCol)
                    firstTokens.append('**dynam')
                    partNumbers.append(f'*part{partNum}')
                    staves.append(f'*staff{partNum}')
            events = pd.concat(_cols, axis=1)
            ba = self._barlines()
            ba = ba[ba != 'regular'].dropna().replace(
                {'double': '||', 'final': '=='})
            ba.loc[self.score.highestTime, :] = '=='
            if data:
                cdata = fromJSON(data).reset_index(drop=True)
                cdata.index = events.index[:len(cdata)]
                firstTokens.extend([f'**{col}' for col in cdata.columns])
                addTieBreakers((events, cdata))
                events = pd.concat((events, cdata), axis=1)
            me = pd.concat([me.iloc[:, 0]] * len(events.columns), axis=1)
            ba = pd.concat([ba.iloc[:, 0]] * len(events.columns), axis=1)
            me.columns = events.columns
            ba.columns = events.columns
            ds = self._analyses['_divisiStarts']
            ds = ds.reindex(events.columns, axis=1).fillna('*')
            de = self._analyses['_divisiEnds']
            de = de.reindex(events.columns, axis=1).fillna('*')
            clefs = self._clefs()
            clefs = clefs.reindex(events.columns, axis=1).fillna('*')
            ts = '*M' + self._timeSignatures()
            ts = ts.reindex(events.columns, axis=1).fillna('*')
            ks = self._keySignatures()
            ks = ks.reindex(events.columns, axis=1).fillna('*')
            partTokens = pd.DataFrame([firstTokens, partNumbers, staves, instruments, partNames, shortNames, ['*-']*len(events.columns)],
                                      index=[-12, -11, -10, -9, -8, -7, int(self.score.highestTime + 1)]).fillna('*')
            partTokens.columns = events.columns
            to_concat = [partTokens, de, me, ds, clefs, ks, ts, events, ba]
            if isinstance(events.index, pd.MultiIndex):
                addTieBreakers(to_concat)
            body = pd.concat(to_concat).sort_index(kind='mergesort')
            if isinstance(body.index, pd.MultiIndex):
                body = body.droplevel(1)
            body = body.fillna('.')
            for colName in [col for col in body.columns if '__' in col]:
                divStarts = np.where(body.loc[:, colName] == '*^')[0]
                divEnds = np.where(body.loc[:, colName] == '*v')[0]
                colIndex = body.columns.get_loc(colName)
                for _ii, startRow in enumerate(divStarts):
                    if _ii == 0:  # delete everying in target cols up to first divisi
                        body.iloc[:startRow + 1, colIndex] = np.nan
                    else:  # delete everything from the last divisi consolidation to this new divisi
                        body.iloc[divEnds[_ii - 1] +
                                  1: startRow + 1, colIndex] = np.nan
                    # delete everything in target cols after final consolidation
                    if _ii + 1 == len(divStarts) and _ii < len(divEnds):
                        body.iloc[divEnds[_ii] + 1:, colIndex] = np.nan

            result = [kernHeader(self.metadata)]
            result.extend(body.apply(lambda row: '\t'.join(
                row.dropna().astype(str)), axis=1))
            result.extend((kernFooter(self.fileExtension),))
            result = '\n'.join(result)
            self._analyses[key] = result
        if not path_name:
            return self._analyses[key]
        else:
            if not path_name.endswith('.krn'):
                path_name += '.krn'
            with open(path_name, 'w') as f:
                f.write(self._analyses[key])

    def _meiStack(self):
        """
        Return a DataFrame stacked to be a multi-indexed series containing the
        score elements to be processed into the MEI format. This is used  for
        MEI output. Only used internally.

        :return: A Series of the score in MEI format

        See Also
        --------
        :meth:`toMEI`
        """
        if '_meiStack' not in self._analyses:
            # assign column names in format (partNumber, voiceNumer) with no splitting up of chords
            events = self._parts(compact=True, number=True).copy()
            clefs = self._m21Clefs().copy()
            ksigs = self._keySignatures(False).copy()
            tsigs = self._timeSignatures(ratio=False).copy()
            mi = pd.MultiIndex.from_tuples([(x, 1) for x in range(
                1, len(clefs.columns) + 1)], names=['Staff', 'Layer'])
            for i, staffInfo in enumerate((clefs, ksigs, tsigs)):
                if 0.0 in staffInfo.index:
                    staffInfo.drop(0.0, inplace=True)
                staffInfo.index = pd.MultiIndex.from_product(
                    [staffInfo.index, [i - 9]])
                staffInfo.columns = mi

            me = self._measures(compact=True)
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
            self._analyses['_meiStack'] = stack
        return self._analyses['_meiStack']

    def _coreMEIElements(self):
        root = ET.Element(
            'mei', {'xmlns': 'http://www.music-encoding.org/ns/mei', 'meiversion': '5.1-dev'})

        meiHead = ET.SubElement(root, 'meiHead')
        fileDesc = ET.SubElement(meiHead, 'fileDesc')
        titleStmt = ET.SubElement(fileDesc, 'titleStmt')
        title = ET.SubElement(titleStmt, 'title')
        title.text = self.metadata['title']
        composer = ET.SubElement(titleStmt, 'composer')
        composer.text = self.metadata['composer']
        pubStmt = ET.SubElement(fileDesc, 'pubStmt')
        unpub = ET.SubElement(pubStmt, 'unpub')
        unpub.text = f'This mei file was converted from a .{self.fileExtension} file by pyAMPACT'
        music = ET.SubElement(root, 'music')
        # insert performance element here
        body = ET.SubElement(music, 'body')
        mdiv = ET.SubElement(body, 'mdiv')
        score = ET.SubElement(mdiv, 'score')
        section = ET.SubElement(score, 'section')
        self.insertScoreDef(root)
        return root

    def toMEI(self, file_name='', indentation='\t', data='', start=None, stop=None, dfs=None, analysis_tag='annot'):
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

        Example
        -------
        .. code-block:: python

            # create an MEI file from a different symbolic notation file
            piece = Score('kerntest.krn')
            piece.toMEI(file_name='meiFile.mei.xml')
        """
        key = ('toMEI', data, start, stop)
        if isinstance(dfs, pd.DataFrame):
            dfs = {'analysis': dfs}
        if key not in self._analyses:
            root = self._coreMEIElements()
            section = root.find('.//section')
            stack = self._meiStack()
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
            self._analyses[key] = ET.ElementTree(root)
            if self._meiTree is None:
                self._meiTree = self._analyses[key]

        if dfs is None:
            ret = self._analyses[key]
        else:   # add analysis data
            ret = deepcopy(self._analyses[key])
            if any((start, stop)):
                for measure in ret.findall('.//measure'):
                    measure_number = int(measure.get('n'))
                    if (start and measure_number < start) or (stop and measure_number > stop):
                        measure.getparent().remove(measure)
            events = self._parts(compact=True, number=True)
            for ii, (tag, df) in enumerate(dfs.items()):
                _df = self.contextualize(
                    df, offsets=True, measures=True, beats=True)
                _df.columns = events.columns[:len(_df.columns)]
                if any((start, stop)):   # trim _df to start and stop
                    if start and stop:
                        _df = _df.loc[idx[:, start:stop, :]]
                        print('here')
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
