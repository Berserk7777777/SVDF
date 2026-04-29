try:
    from .animalkingdom import AnimalKingdomActionRecognitionDataset
except ImportError:
    pass

try:
    from .charades import CharadesDataset
except ImportError:
    pass

from .ucf import UCFDataset
from .hmdb import HMDBDataset
from .ssv2 import SSV2Dataset, SSV2FeatureDataset