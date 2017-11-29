"""
=======
CausticMass
=======
Python tools to calculate Caustic Masses of Galaxy clusters given 
spectroscopic redshift data

Github: https://github.com/akremin/CausticMass

Notes: Currently uses an NFW fit as the caustic surface*
CausticMass.py contains 3 classes/objects each with a list of attributes and functions
"""


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from . import caustic
from . import causticsurface
from . import datastructs #data_structures_depr as datastructs

Caustic = caustic.Caustic
CausticSurface = causticsurface.CausticSurface
MassInfo = datastructs.MassInfo
CausticFitResults = datastructs.CausticFitResults
ClusterData = datastructs.ClusterData






        


        




            

        
