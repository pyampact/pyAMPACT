from .version import version as __version__

from . import alignment, alignmentUtils, dataCompilation, performance
from . import speechDescriptors, speechDescriptorsUtils
from . import symbolic, symbolicUtils

from .alignment import *
from .alignmentUtils import *
from .dataCompilation import *
from .performance import *
from .speechDescriptors import *
from .speechDescriptorsUtils import *
from .symbolic import *
from .symbolicUtils import *

__all__ = (
    alignment.__all__
    + alignmentUtils.__all__
    + dataCompilation.__all__
    + performance.__all__
    + speechDescriptors.__all__
    + speechDescriptorsUtils.__all__
    + symbolic.__all__
    + symbolicUtils.__all__
)

