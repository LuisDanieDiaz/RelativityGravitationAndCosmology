# my_relativity/__init__.py

# Importar funciones/clases específicas de tus submódulos
from . import plots
from . import newton
from . import relativity

# import *
__all__ = ['plots', 'newton']

# Package information
__version__ = '0.1.0'
__author__ = 'Luis Daniel Díaz'