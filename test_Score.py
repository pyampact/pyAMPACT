import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import pyampact
import pandas as pd
from unittest.mock import patch
from io import StringIO

def check_sampled_piano_roll(score_path, truth_path):
  piece = pyampact.Score(score_path)
  pianoRoll = piece.piano_roll()
  sampled = piece.sampled()
  samp = sampled.copy()
  sampled.columns = range(sampled.columns.size)  # reset column names to be enumerated
  groundTruth = pd.read_csv(truth_path, header=None)
  groundTruth.index = list(range(1, 128))
  slice1 = sampled.loc[1:127, :]
  slice2 = groundTruth.loc[1:127, :]
  assert(slice1.equals(slice2))

def check_lyrics(score_path, shape, first, last):
  piece = pyampact.Score(score_path)
  lyrics = piece.lyrics()
  assert(type(lyrics) == pd.DataFrame)
  assert(lyrics.shape == shape)
  assert(lyrics.iloc[0].at[lyrics.iloc[0].first_valid_index()] == first)
  assert(lyrics.iloc[-1].at[lyrics.iloc[-1].last_valid_index()] == last)

def check_harm_spine(score, control, filler='forward', output='array'):
  test = score.harm(filler=filler, output=output)
  if output == 'array':
    assert all(test == control)
  else:  # data are pandas series
    assert test.equals(control)

def check_harm_keys(score, control, filler='forward', output='array'):
  test = score.keys(filler=filler, output=output)
  if output == 'array':
    assert all(test == control)
  else:  # data are pandas series
    assert test.equals(control)

def check_function_spine(score, control, filler='forward', output='array'):
  test = score.functions(filler=filler, output=output)
  if output == 'array':
    assert all(test == control)
  else:  # data are pandas series
    assert test.equals(control)

# check creation of Score objects from various types of symbolic notation files
def test_local_import():
  assert isinstance(pyampact.Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/monophonic6notes.mid'), pyampact.Score)
  assert isinstance(pyampact.Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/polyphonic3voices1note.mid'), pyampact.Score)
  assert isinstance(pyampact.Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/polyphonic4voices1note.mei'), pyampact.Score)

def test_remote_import():
  assert isinstance(pyampact.Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn'), pyampact.Score)

def test_volpiano_import_1():
  piece = pyampact.Score('1---gkjH7--klk-jkjh-ghg---jh---kl--k---jh--jk--hj---ghjh--hg---g--g--g--g7---hjh-ghjh---gf--gh--hkjklkjh--hjh-ghg---3---f--g77---3')
  assert isinstance(piece, pyampact.Score)
  assert not piece.notes().empty
  
def test_volpiano_import_2():
  piece = pyampact.Score('1--c--d---f--d---ed--c--d---f---g--h--j---hgf--g--h---')
  assert isinstance(piece, pyampact.Score)
  assert not piece.notes().empty

def test_tinyNotation_import_1():
  piece = pyampact.Score('4/4 c4 c# c c## cn c- c-- c_Lyric c1')
  assert isinstance(piece, pyampact.Score)
  assert not piece.notes().empty

def test_tinyNotation_import_2():
  piece = pyampact.Score('tinyNotation: 4/4 c4 trip{c8 d e} trip{f4 g a} b-1')
  assert isinstance(piece, pyampact.Score)
  assert not piece.notes().empty

def test_lyrics():
  check_lyrics('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/Busnoys_In_hydraulis.krn', (438, 4), 'In', 'cum.')

def test_spine_data():
  piece = pyampact.Score('https://github.com/pyampact/pyAMPACTtutorials/blob/main/test_files/M025_00_01a_a.krn')
  harm_series = pd.Series(['V', 'I', 'I', 'I', 'I', 'I', 'I', 'V', 'V', 'V', 'V', 'V', 'V7b', 'V7b', 'I', 'I', 'I', 'I',
      'V', 'V', 'V7b', 'V7b', 'I', 'I', 'I', 'iib', 'iib', 'iib', 'iib', 'V', 'V', 'I', 'I', 'I'],
    [0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0, 13.0,
      14.0, 15.0, 16.0, 16.5, 17.0, 18.0, 19.0, 19.5, 20.0, 20.5, 21.0, 22.0, 23.0])
  check_harm_spine(piece, harm_series.values)
  check_harm_spine(piece, harm_series, output='series')
  check_harm_keys(piece, pd.Series(['D'], [0.0]), filler='drop', output='series')
  check_function_spine(piece, pd.Series(['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T',
      'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'P', 'P', 'P', 'P', 'D', 'D', 'T', 'T', 'T'], harm_series.index), output='series')

def test_show():
  '''\tMake sure the url generation works in a very simple case. Only check the first 100 characters
  because pyAMPACT adds the date of generation of the kern score in the footer metadata, so the full
  string is a little different every day.'''
  kernString = """**kern
1c;
==
*-"""
  piece = pyampact.Score(kernString)
  with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
    piece.show()
  expected_output = "https://verovio.humdrum.org/?t=ISEhQ09NOiBDb21wb3NlciBub3QgZm91bmQKISEhT1RMOiBUaXRsZSBub3QgZm91bmQKK"
  assert mock_stdout.getvalue()[:100] == expected_output

