"""
Neural Network Training Progress Visualization

:author: Silvie Nemcova (xnemco06@stud.fit.vutbr.cz)
:year: 2021
"""
from .data_loader import *
from .examine1D import *
from .examine2D import *
from .examine_surface import *
from .net import *
from .paths import *
from .plot import *
from .prelim import *
import logging

logger = logging.getLogger("vis_net")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    filename="vis_net.log")

__version__ = "1.1.3"
