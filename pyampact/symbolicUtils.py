"""
symbolicUtils
==============


.. autosummary::
    :toctree: generated/

    _escape_cdata
    addMEINote
    addTieBreakers
    kernClefHelper
    combineRests
    combineUnisons
    fromJSON
    _id_gen
    indentMEI
    _kernChordHelper
    kernFooter
    kernHeader
    _kernNoteHelper
    kernNRCHelper
    noteRestHelper
    remove_namespaces
    removeTied
    snapTo
    truncate_and_scale_onsOffsList
    githubURLtoRaw

"""

import json
import numpy as np
import pandas as pd
import re
import requests
import xml.etree.ElementTree as ET
from fractions import Fraction

__all__ = [
    "_escape_cdata",
    "addMEINote",
    "addTieBreakers",
    "kernClefHelper",
    "combineRests",
    "combineUnisons",
    "fromJSON",
    "_id_gen",
    "indentMEI",
    "_kernChordHelper",
    "kernFooter",
    "kernHeader",
    "_kernNoteHelper",
    "kernNRCHelper",
    "noteRestHelper",
    "remove_namespaces",
    "removeTied",
    "snapTo",
    "truncate_and_scale_onsOffsList",
    "_duration2Kern",
    "duration2MEI",
    "function_pattern",
    "imported_scores",
    "tinyNotation_pattern",
    "volpiano_pattern",
    "meiDeclaration",
    "idGen",
    "githubURLtoRaw"

]


def _escape_cdata(text):
    """
    Escape certain characters in a CDATA string for XML serialization.

    This function checks if the input text is a CDATA string. If it is, the text is
    returned as is. If it's not, the function escapes the characters "&", "<", and
    ">" by replacing them with their corresponding XML entities ("&amp;", "&lt;",
    and "&gt;").

    This function is used to overwrite the default escape function in the xml.etree
    module. The default escape function does not escape characters in CDATA strings,
    which can cause XML serialization to fail.

    Parameters:
    text (str): The input string to be escaped.

    Returns:
    str: The escaped string, safe for XML serialization.

    Raises:
    TypeError: If the input is not a string.
    """
    try:
        if text.startswith(" <![CDATA[") and text.endswith("]]> "):
            return text
        if "&" in text:
            text = text.replace("&", "&amp;")
        if "<" in text:
            text = text.replace("<", "&lt;")
        if ">" in text:
            text = text.replace(">", "&gt;")
        return text
    except TypeError:
        raise TypeError("cannot serialize %r (type %s)" %
                        (text, type(text).__name__))


ET._escape_cdata = _escape_cdata

_duration2Kern = {  # keys get rounded to 5 decimal places
    56:        '000..',
    48:        '000.',
    32:        '000',
    28:        '00..',
    24:        '00.',
    16:        '00',
    14:        '0..',
    12:        '0.',
    8:         '0',
    7:         '1..',
    6:         '1.',
    4:         '1',
    3.5:       '2..',
    3:         '2.',
    2.66667:   '3%2',
    2:         '2',
    1.75:      '4..',
    1.5:       '4.',
    1.33333:   '3',
    1:         '4',
    .875:      '8..',
    .75:       '8.',
    .66667:    '6',
    .5:        '8',
    .4375:     '16..',
    .375:      '16.',
    .33333:    '12',
    .25:       '16',
    .21875:    '32..',
    .1875:     '32.',
    .16667:    '24',
    .125:      '32',
    .10938:    '64..',
    .09375:    '64.',
    .08333:    '48',
    .0625:     '64',
    .05469:    '128..',
    .04688:    '128.',
    .04167:    '96',
    .03125:    '128',
    .02734:    '256..',
    .02344:    '256.',
    .02083:    '192',
    .01563:    '256',
    .01367:    '512..',
    .01172:    '512.',
    .01042:    '384',
    .00781:    '512',
    .00684:    '1024.',
    .00586:    '1024.',
    .00582:    '768',
    .00391:    '1024',
    0:         '',
    '128th':   '128',    # grace note durations
    '64th':    '64',
    '32nd':    '32',
    '16th':    '16',
    'eighth':  '8',
    'quarter': '8'      # make quarter grace notes default to eighth notes too
}

duration2MEI = {
    'maxima':  'maxima',
    'longa':   'longa',
    'breve':   'breve',
    'whole':   '1',
    'half':    '2',
    'quarter': '4',
    'eighth':  '8',
    '16th':    '16',
    '32nd':    '32',
    '64th':    '64',
    '128th':   '128',
    '256th':   '256',
    '512th':   '512',
    '1024th':  '1024'
}

function_pattern = re.compile('[^TtPpDd]')
imported_scores = {}
tinyNotation_pattern = re.compile("^[-0-9a-zA-Zn _/'#:~.{}=]+$")
volpiano_pattern = re.compile(r'^\d--[a-zA-Z0-9\-\)\?]*$')

meiDeclaration = """<?xml version="1.0" encoding="UTF-8"?>
<?xml-model href="https://music-encoding.org/schema/dev/mei-all.rng" type="application/xml" schematypens="http://relaxng.org/ns/structure/1.0"?>
"""


def addMEINote(note, parent, syl=None):
    """
    Add a note element to an MEI parent element from a music21 note object.

    This function creates a new 'note' subelement under the given parent
    element, and sets its attributes based on the properties of the given note.
    It also handles grace notes, accidentals, lyrics, and dynamics if any of
    these are found on the note.

    Parameters:
    note (music21.note.Note): The note to add. The note's properties (octave,
        step, id, duration, pitch, lyric, expressions) are used to set the
        attributes of the new MEI element.
    parent (xml.etree.ElementTree.Element): The parent element to which the new
        'note' element will be added.
    syl (str, optional): A syllable to add to the note as a 'syl' element. If
        not provided, the note's lyric property is used.

    Returns:
    xml.etree.ElementTree.Element: The new 'note' element.
    """

    note_el = ET.SubElement(parent, 'note', {'oct': f'{note.octave}',
                                             'pname': f'{note.step.lower()}', 'xml:id': f'{note.id}', 'dots': f'{note.duration.dots}'})
    if note.duration.isGrace:
        note_el.set('grace', 'acc')
        note_el.set('dur', duration2MEI[note.duration.type])
    else:
        note_el.set('dur', duration2MEI[note.duration.type])
    alter = note.pitch.alter or 0
    if note.pitch.accidental and note.pitch.accidental.displayStatus:
        if alter > 0:
            note_el.set('accid', 's'*int(alter))
            note_el.set('accid.ges', 's'*int(alter))
        elif alter < 0:
            note_el.set('accid', 'f'*int(-alter))
            note_el.set('accid.ges', 'f'*int(-alter))
        else:
            note_el.set('accid', 'n')
            note_el.set('accid.ges', 'n')
    else:
        if alter > 0:
            note_el.set('accid.ges', 's'*int(alter))
        elif alter < 0:
            note_el.set('accid.ges', 'f'*int(-alter))
        else:
            note_el.set('accid.ges', 'n')
    if note.lyric:
        verse_el = ET.SubElement(
            note_el, 'verse', {'n': '1', 'xml:id': next(idGen)})
        syl_el = ET.SubElement(verse_el, 'syl', {'xml:id': next(idGen)})
        syl_el.text = note.lyric.strip().split('\n')[0]
    for exp in note.expressions:
        if 'Dynamic' in exp.classes:
            dyn_el = ET.SubElement(note_el, 'dynam', {'xml:id': next(idGen)})
            dyn_el.text = exp.value


def addTieBreakers(partList):
    """
    Add tie-breaker level to index. Changes parts in partList in place and 
    returns None. 

    This is particularly useful to disambiguate the order of events that 
    happen at the same offset, which is an issue most commonly encountered 
    with grace notes since they have no duration. This is needed in several 
    `Score` methods because you cannot append multiple pandas series (parts) 
    if they have non-unique indices. So this method is needed internally to 
    be able to use pd.concat to turn a list of series into a single dataframe 
    if any of those series has a repeated value in its index.

    :param partList: A list of pandas Series, each representing a part in 
        the score.
    :return: None
    """
    for part in partList:
        if isinstance(part.index, pd.MultiIndex):
            continue
        tieBreakers = []
        nexts = part.index.to_series().shift(-1)
        for ii in range(-1, -1 - len(part.index), -1):
            if part.index[ii] == nexts.iat[ii]:
                tieBreakers.append(tieBreakers[-1] - 1)
            else:
                tieBreakers.append(0)
        tieBreakers.reverse()
        part.index = pd.MultiIndex.from_arrays((part.index, tieBreakers))


def kernClefHelper(clef):
    """
    Parse a music21 clef object into the corresponding humdrum syntax token.

    :param clef: A music21 clef object.
    :return: A string representing the humdrum syntax token for the clef.
    """
    octaveChange = ''
    if clef.octaveChange > 0:
        octaveChange = '^' * clef.octaveChange
    elif clef.octaveChange < 0:
        octaveChange = 'v' * abs(clef.octaveChange)
    return f'*clef{clef.sign}{octaveChange}{clef.line}'


def combineRests(col):
    """
    Helper function for the `notes` method. 

    Combine consecutive rests in a given voice. Non-first consecutive rests 
    will be removed.

    :param col: A pandas Series representing a voice.
    :return: The same pandas Series with consecutive rests combined.
    """
    col = col.dropna()
    return col[(col != 'r') | ((col == 'r') & (col.shift(1) != 'r'))]


def combineUnisons(col):
    """
    Helper function for the `notes` method. 

    Combine consecutive unisons in a given voice. Non-first consecutive unisons 
    will be removed.

    :param col: A pandas Series representing a voice.
    :return: The same pandas Series with consecutive unisons combined.
    """
    col = col.dropna()
    return col[(col == 'r') | (col != col.shift(1))]


def githubURLtoRaw(string):
    """
    Convert a GitHub URL to a raw URL and return it. Otherwise return the string.
    """
    if string.startswith('https://github.com/'):
        return 'https://raw.githubusercontent.com/' + string[19:].replace('/blob/', '/', 1)
    return string


def fromJSON(json_path):
    """
    Load a JSON or dez file/url into a pandas DataFrame.

    The outermost keys of the JSON object are interpreted as the index values of 
    the DataFrame and should be in seconds with decimal places allowed. The 
    second-level keys become the columns of the DataFrame.

    :param json_path: Path to a JSON or dez file.
    :return: A pandas DataFrame representing the JSON data.

    See Also
    --------
    :meth:`jsonCDATA`
    :meth:`nmats`

    Example
    -------
    .. code-block:: python

        piece = Score('./test_files/CloseToYou.mei.xml')
        piece.fromJSON(json_path='./test_files/CloseToYou.json')
    """
    if json_path.startswith('https://') or json_path.startswith('http://'):
        json_path = githubURLtoRaw(json_path)
        response = requests.get(json_path)
        data = json.loads(response.text)
    else:
        with open(json_path) as json_data:
            data = json.load(json_data)

    if ((isinstance(json_path, str) and json_path.lower().endswith('.dez'))
            or (hasattr(json_path, 'name') and json_path.name.lower().endswith('.dez'))):
        df = pd.DataFrame.from_records(data['labels'])
        if 'start' in df.columns:
            df['start'] = df['start'].fillna(0.0)
    else:   # .json file
        df = pd.DataFrame(data).T
        df.index = df.index.astype(str)
    return df


def _id_gen(start=1):
    """
    Generate a unique ID for each instance of the Score class.

    This function generates a unique ID for each instance of the Score class 
    by incrementing a counter starting from the provided start value. The ID 
    is in the format 'pyAMPACT-{start}'. This isn't meant to be used directly
    so see the example below for usage.

    :param start: An integer representing the starting value for the ID 
        counter. Default is 1.
    :yield: A string representing the unique ID.

    See Also
    --------
    :meth:`insertAudioAnalysis`
    :meth:`xmlIDs`

    Example
    --------
    .. code-block:: python

        newID = next(idGen)
    """
    while True:
        yield f'pyAMPACT-{start}'
        start += 1


idGen = _id_gen()


def indentMEI(elem, indentation='\t', _level=0):
    """
    Indent an MEI (Music Encoding Initiative) XML element and its children.

    This function recursively indents an XML element and its children for
    pretty printing. The indentation level is increased for each level of
    depth in the XML tree.

    Parameters:
    elem (xml.etree.ElementTree.Element): The XML element to indent.
    indentation (str, optional): The indentation string to use. Default is a
        tab character. Use a ' ' (space) for maximally compact output.
    _level (int, optional): The initial indentation level. This parameter is used
        internally in recursive calls but should not be set by the user.

    Returns:
    None. The function modifies the XML element in place.
    """
    i = f'\n{_level*indentation}'
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = f'{i}{indentation}'
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indentMEI(elem, indentation, _level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if _level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def _kernChordHelper(_chord):
    """
    Parse a music21 chord object into a kern chord token.

    This method uses the `_kernNoteHelper` method to convert each note in the 
    chord into a kern note token. The tokens are then joined together with 
    spaces to form the kern chord token.

    :param _chord: A music21 chord object to be converted into a kern chord token.
    :return: A string representing the kern chord token.
    """
    return ' '.join([_kernNoteHelper(note) for note in _chord.notes])


def kernFooter(fileExtension):
    """
    Return a string of the kern format footer global comments.

    The footer includes the translation date and other relevant metadata.

    :return: A string representing the kern format footer.
    """
    from datetime import datetime
    return f"""!!!RDF**kern: %=rational rhythm
!!!RDF**kern: l=long note in original notation
!!!RDF**kern: i=editorial accidental
!!!ONB: Translated from a {fileExtension} file on {datetime.today().strftime("%Y-%m-%d")} via pyAMPACT
!!!title: @{{OTL}}"""


def kernHeader(metadata):
    """
    Return a string of the kern format header global comments.

    The header includes the composer and title metadata.

    :return: A string representing the kern format header.
    """
    return f'!!!COM: {metadata["composer"]}\n!!!OTL: {metadata["title"]}'


def _kernNoteHelper(_note):
    """
    Parse a music21 note object into a kern note token.

    This method handles the conversion of various musical notations such as 
    ties, slurs, beams, durations, octaves, accidentals, longas, and grace 
    notes into the kern format.

    :param _note: A music21 note object to be converted into a kern note token.
    :return: A string representing the kern note token.
    """
    # TODO: this doesn't seem to be detecting longas in scores. Does m21 just not detect longas in kern files? Test with mei, midi, and xml
    startBracket, endBracket, beaming = '', '', ''
    if hasattr(_note, 'tie') and _note.tie is not None:
        if _note.tie.type == 'start':
            startBracket += '['
        elif _note.tie.type == 'continue':
            endBracket += '_'
        elif _note.tie.type == 'stop':
            endBracket += ']'

    spanners = _note.getSpannerSites()
    for spanner in spanners:
        if 'Slur' in spanner.classes:
            if spanner.isFirst(_note):
                startBracket = '(' + startBracket
            elif spanner.isLast(_note):
                endBracket += ')'

    beams = _note.beams.beamsList
    for beam in beams:
        if beam.type == 'start':
            beaming += 'L'
        elif beam.type == 'stop':
            beaming += 'J'

    _oct = _note.octave
    if _oct > 3:
        letter = _note.step.lower() * (_oct - 3)
    else:
        letter = _note.step * (4 - _oct)
    acc = _note.pitch.accidental
    acc = acc.modifier if acc is not None else ''
    longa = 'l' if _note.duration.type == 'longa' else ''
    if _note.duration.isGrace:
        dur = _duration2Kern.get(_note.duration.type, '')
        grace = 'q' if _note.duration.slash else 'qq'
    else:
        grace = ''
        dur = _duration2Kern[round(float(_note.quarterLength), 5)]
    # TODO: make this sensitive to notehead and practical duration
    grace = 'q' if _note.duration.isGrace else ''
    fermata = ''
    for exp in _note.expressions:
        if exp.name == 'fermata':
            fermata = ';'
    return f'{startBracket}{dur}{letter}{acc}{longa}{grace}{fermata}{beaming}{endBracket}'


def kernNRCHelper(nrc):
    """
    Convert a music21 note, rest, or chord object to its corresponding kern token.

    This method uses the `_kernNoteHelper` and `_kernChordHelper` methods to 
    convert note and chord objects, respectively. Rest objects are converted 
    directly in this method.

    :param nrc: A music21 note, rest, or chord object to be converted into a 
        kern token.
    :return: A string representing the kern token.
    """
    if nrc.isNote:
        return _kernNoteHelper(nrc)
    elif nrc.isRest:
        return f'{_duration2Kern.get(round(float(nrc.quarterLength), 5))}r'
    else:
        return ' '.join([_kernNoteHelper(note) for note in nrc.notes])


def noteRestHelper(nr):
    """
    Helper function for the `notes` method. 

    If the note/rest object `nr` is a rest, return 'r'. Otherwise, return the 
    note's name with octave.

    :param nr: A note/rest object.
    :return: 'r' if `nr` is a rest, otherwise the note's name with octave.
    """
    if nr.isRest:
        return 'r'
    return nr.nameWithOctave


def remove_namespaces(doc):
    """
    Indent an MEI (Music Encoding Initiative) element for better readability.

    This function recursively indents an MEI element and its children, improving 
    the readability of the MEI XML structure. It modifies the input element in-place.

    :param elem: An xml.etree.ElementTree.Element representing the MEI element.
    :param level: An integer representing the current indentation level. Default is 0.
    :return: None
    """
    root = doc.getroot()
    namespace = ''
    if '}' in root.tag:
        namespace = root.tag[1:root.tag.index('}')]
    for elem in doc.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag[elem.tag.index('}') + 1:]
    if namespace:
        root.set('xmlns', namespace)


def removeTied(noteOrRest):
    """
    Helper function for the `_m21ObjectsNoTies` method. 

    Remove tied notes in a given note or rest. Only the first note in a tied 
    group will be kept.

    :param noteOrRest: A music21 note or rest object.
    :return: np.nan if the note is tied and not the first in the group, 
        otherwise the original note or rest.
    """
    if hasattr(noteOrRest, 'tie') and noteOrRest.tie is not None and noteOrRest.tie.type != 'start':
        return np.nan
    return noteOrRest


def snapTo(data, snap_to=None, filler='forward', output='array'):
    """"
    Takes a `harm`, `keys`, `functions`, `chords`, or `cdata` as `data` and
    the `snap_to` and `filler` parameters as described in the former three's 
    doc strings.

    The passed data is returned in the shape of the snap_to dataframe's columns,
    and any filling operations are applied. The output will be in the form of a 
    1D numpy array unless `output` is changed, in which case a series will be 
    returned for harm, keys, functions, and chords data, and a dataframe for 
    cdata data.

    :param data: Can be `harm`, `keys`, `functions`, `chords`, or `cdata`.
    :param snap_to: Described in the docstrings of `harm`, `keys`, and 
        `functions`.
    :param filler: Described in the docstrings of `harm`, `keys`, and 
        `functions`.
    :param output: If changed, a series will be returned for `harm`, `keys`, 
        `functions`, and `chords` data, and a dataframe for `cdata` data. Default 
        is a 1D numpy array.
    :return: The passed data in the shape of the `snap_to` dataframe's columns 
        with any filling operations applied.
    """
    if isinstance(data, list):
        if len(data) == 1:
            _data = data[0].copy()
        else:
            _data = pd.concat(data, axis=1)
    else:
        _data = data.copy()
    if snap_to is not None:
        if not _data.index.is_unique:
            _data = _data[~_data.index.duplicated(keep='last')]
        _data = _data.reindex(snap_to.columns)
    if filler != '.':
        _data.replace('.', np.nan, inplace=True)
    if isinstance(filler, str):
        filler = filler.lower()
        if filler == 'forward':
            _data = _data.infer_objects(copy=False).ffill()
        else:
            if filler in ('nan', 'drop'):
                _data.fillna(np.nan, inplace=True)
            else:
                _data.fillna(filler, inplace=True)
    if filler == 'drop':
        _data.dropna(inplace=True)
    if output == 'array':
        return _data.values
    else:
        return _data


def truncate_and_scale_onsOffsList(onsOffsList, target_length):
    """
    Scale and truncate onsOffsList to match the target_length, adjusting values proportionally.

    Parameters:
    onsOffsList (list of lists): List containing [ONSET_SEC, OFFSET_SEC] pairs.
    target_length (int): Desired length of the output list.

    Returns:
    list of lists: Scaled and truncated onsOffsList.
    """
    current_length = len(onsOffsList)

    if current_length == target_length:
        return onsOffsList

    elif current_length > target_length:
        # Scale down
        scale_factor = current_length / target_length
        truncated_list = []
        for i in range(target_length):
            start_index = int(i * scale_factor)
            end_index = int((i + 1) * scale_factor)

            # Calculate mean values for the range
            start_time = np.mean(
                [onsOffsList[start_index][0], onsOffsList[end_index - 1][0]])
            end_time = np.mean(
                [onsOffsList[start_index][1], onsOffsList[end_index - 1][1]])

            truncated_list.append([start_time, end_time])

        return truncated_list

    else:
        # Scale up by interpolation
        scaling_factor = target_length / current_length
        new_onsOffsList = []
        for i in range(target_length):
            original_index = i / scaling_factor
            lower_index = int(np.floor(original_index))
            upper_index = int(np.ceil(original_index))

            if lower_index == upper_index:
                new_onsOffsList.append(onsOffsList[lower_index])
            else:
                upper_index = upper_index - 1
                lower_value = onsOffsList[lower_index]
                upper_value = onsOffsList[upper_index]
                fraction = original_index - lower_index

                new_onsOffsList.append([
                    lower_value[0] + fraction *
                    (upper_value[0] - lower_value[0]),
                    lower_value[1] + fraction *
                    (upper_value[1] - lower_value[1])
                ])

        return new_onsOffsList
