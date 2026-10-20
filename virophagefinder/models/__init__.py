"""Model architectures for VirophageFinder."""

from .protein_encoder import ProteinEncoder
from .nucleotide_encoder import NucleotideEncoder
from .graph_encoder import GeneGraphEncoder
from .fusion import GatedFusion
from .classifier import ClassificationHead
from .virophagefinder import VirophageFinderModel

__all__ = [
    "ProteinEncoder",
    "NucleotideEncoder",
    "GeneGraphEncoder",
    "GatedFusion",
    "ClassificationHead",
    "VirophageFinderModel",
]
