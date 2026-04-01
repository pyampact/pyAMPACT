#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Version info"""

import sys
import importlib

short_version = "0.0"
version = "0.0.6"

def __get_mod_version(modname):
    try:
        if modname in sys.modules:
            mod = sys.modules[modname]
        else:
            mod = importlib.import_module(modname)
        try:
            return mod.__version__
        except AttributeError:
            return "installed, no version number available"

    except ImportError:
        return None


def show_versions() -> None:
    """Return the version information for all pyampact dependencies."""
    core_deps = [
        'music21==9.1.0',
        'pandas==2.2.2',
        'pyarrow>=15.0.0',
        'numpy>=1.26.4,<2.0',
        'requests==2.31.0',
        'scipy>=1.11.4',
        'librosa>=0.10.1',
        'praat-parselmouth==0.4.3',
        'pywavelets>=1.4.1',
    ]

    print("INSTALLED VERSIONS")
    print("------------------")
    print(f"python: {sys.version}\n")
    print(f"pyampact: {version}\n")
    for dep in core_deps:
        print("{}: {}".format(dep, __get_mod_version(dep)))
    print("")
    # for dep in extra_deps:
    #     print("{}: {}".format(dep, __get_mod_version(dep)))
