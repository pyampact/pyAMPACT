.. pyAMPACT documentation master file, created by
   sphinx-quickstart on Mon Dec 11 12:13:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


**************
pyAMPACT
**************
`pyAMPACT` (Python-based Automatic Music Performance Analysis and Comparison
Toolkit) is a python package thatlinks symbolic and audio music representations
to facilitate score-informed estimation of performance data in audio as well as
general linking of symbolic and audio music representations with a variety of
annotations. pyAMPACT  can read a range of symbolic formats and can output
note-linked audio descriptors/performance data into MEI and Humdrum kern files.
The audio analysis uses score alignment to calculate time-frequency regions of
importance for each note in the symbolic representation from which to estimate
a range of parameters. These include tuning-, dynamics-, and timbre-related
performance descriptors, while timing-related information is available from
the score alignment. Beyond performance data estimation, pyAMPACT also
facilitates multi-modal investigations through its robust infrastructure for
linking symbolic representations and annotations to audio.



Welcome to pyAMPACT's documentation!
====================================

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   install


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   alignment
   alignmentUtils
   dataCompilation
   performance
   symbolic
   symbolicUtils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
