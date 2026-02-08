"""
Geometry Techniques Package

This package contains geometry technique implementations split across multiple modules:
- circles: Circle-related techniques (power of point, circumradius, inradius, etc.)
- triangles: Triangle-related techniques (Heron's formula, Stewart's theorem, etc.)
- polygons: Polygon-related techniques (quadrilaterals, convex hulls, trapezoids, etc.)
- coordinates: Coordinate geometry techniques (distance, lines, vectors, etc.)
- transformations: Geometric transformations (homothety, reflection, spiral similarity, etc.)
- misc: Miscellaneous techniques (lattice points, 3D geometry, projective geometry, etc.)

All technique classes are decorated with @register_technique for automatic registration.
"""

# Re-export common utilities
from ._common import (
    MethodBlock, MethodResult, register_technique,
    ensure_tuple, line_intersection,
)

# Import all circle techniques
from .circles import *
from .circles_2 import *

# Import all triangle techniques
from .triangles import *
from .triangles_2 import *
from .triangles_3 import *
from .triangles_4 import *

# Import all polygon techniques
from .polygons import *
from .polygons_2 import *

# Import all coordinate geometry techniques
from .coordinates import *
from .coordinates_2 import *
from .coordinates_3 import *

# Import all transformation techniques
from .transformations import *

# Import all miscellaneous techniques
from .misc import *
from .misc_2 import *
from .misc_3 import *
from .misc_4 import *

# Import ported geometry/optimization techniques
from .ported_geo import *
from .ported_geo_2 import *
