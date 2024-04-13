from .version import version as __version__

from .alignment import *
from .alignmentUtils import *
from .dataCompilation import *
from .performance import *
from .symbolic import Score
from .symbolicUtils import *

__all__ = alignment.__all__ + alignmentUtils.__all__ + dataCompilation.__all__ + performance.__all__ + ["Score"] + symbolicUtils.__all__