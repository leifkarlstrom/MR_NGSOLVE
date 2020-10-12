from netgen.geom2d import SplineGeometry


def region_outside_rectangular_cavity(A, B, D, L, hcavity):
    """
    This routine creates a spline geometry (which can later be meshed) for
    the region outside a rectangular cavity, with these parameters:

    (0, B+D)
    +--------------------------------------------+  (L, B+D)
    |                                            |
    |                                            |
    |                                            |
   (0,B)                                         |
    +--------+ (A, B)                            |
             |                                   |
             |                                   |
   (0,0)     |                                   |
    +        |                                   |
             |                                   |
             |                                   |
             |                                   |
    +--------+                                   |
   (0,-B)                                        |
    |                                            |
    |                                            |
    |                                            |
    |                                            |
    |                                            |
    |                                            |
    |                                            |
    +--------------------------------------------+ (L, -L)
    (0, -L)


    Input "hcavity" specifies meshsize near the inner rectangular cavity.

    """

    htop = L
    geo = SplineGeometry()
    Ctop, Ctrg, Cbrg, Cbot = [geo.AppendPoint(x, y)
                              for (x, y) in [(0, B),
                                             (A, B),
                                             (A, -B),
                                             (0, -B)]]
    Boxtop, Boxtrg, Boxbrg, Boxbot = [geo.AppendPoint(x, y)
                                      for (x, y) in [(0, B+D),
                                                     (L, B+D),
                                                     (L, -L),
                                                     (0, -L)]]

    # go counter clockwise to get outward unit normal right
    geo.Append(['line', Cbot, Boxbot],
               bc='axis', leftdomain=1, rightdomain=0)
    geo.Append(['line', Boxbot, Boxbrg],
               bc='bot', leftdomain=1, rightdomain=0)
    geo.Append(['line', Boxbrg, Boxtrg],
               bc='rgt', leftdomain=1, rightdomain=0)
    geo.Append(['line', Boxtrg, Boxtop],
               bc='top', leftdomain=1, rightdomain=0, maxh=htop)
    geo.Append(['line', Boxtop, Ctop],
               bc='axis', leftdomain=1, rightdomain=0)
    geo.Append(['line', Ctop, Ctrg],
               bc='cavity', leftdomain=1, rightdomain=0, maxh=hcavity)
    geo.Append(['line', Ctrg, Cbrg],
               bc='cavity', leftdomain=1, rightdomain=0, maxh=hcavity)
    geo.Append(['line', Cbrg, Cbot],
               bc='cavity', leftdomain=1, rightdomain=0, maxh=hcavity)

    return geo


def region_outside_cavity(A, B, D, Lr, hcavity, Lz=None, htop=None):
    """
    This routine creates a spline geometry (which can later be meshed) for
    the region outside the magma cavity, with these parameters:

    - Cavity is ellipsoidal of r-semimajor axis length A, z-semiminor axis B.
    - Terrain is at distance D from the center of the ellipse.
    - The ellipse is centered at the origin of the coordinate system.
    - The enclosing box has vertices (0, B+D), (Lr, B+D), (0, -Lz), (Lr, -Lz).
    - Maximal element length near ellipse (when meshing) is not more
      than hcavity.

    (0, B+D)
    +--------------------------------------------+  (Lr, B+D)
    |                                            |
    |                                            |
    |                                            |
   (0,B)                                         |
    +...                                         |
         ...                                     |
           ..                                    |
   (0,0)     .                                   |
    +        + (A,0)                             |
             .                                   |
           ..                                    |
        ...                                      |
    +...                                         |
   (0,-B)                                        |
    |                                            |
    |                                            |
    |                                            |
    |                                            |
    |                                            |
    |                                            |
    |                                            |
    +--------------------------------------------+ (Lr, -Lz)
    (0, -Lz)


    """

    if Lz is None:
        Lz = Lr

    if htop is None:
        htop = Lr
    geo = SplineGeometry()
    ellipsetop, ellipsectlu, ellipsergt = [geo.AppendPoint(x, y)
                                           for (x, y) in [(0, B),
                                                          (A, B),
                                                          (A, 0)]]
    ellipsebot, ellipsectld = [geo.AppendPoint(x, y)
                               for (x, y) in [(0, -B), (A, -B)]]
    boxtoplft, boxtoprgt, boxbotlft, boxbotrgt = [geo.AppendPoint(x, y)
                                                  for (x, y) in [(0, B+D),
                                                                 (Lr, B+D),
                                                                 (0, -Lz),
                                                                 (Lr, -Lz)]]
    # go counter clockwise to get outward unit normal right
    geo.Append(['line', ellipsebot, boxbotlft],
               bc='axis', leftdomain=1, rightdomain=0)
    geo.Append(['line', boxbotlft, boxbotrgt],
               bc='bot', leftdomain=1, rightdomain=0)
    geo.Append(['line', boxbotrgt, boxtoprgt],
               bc='rgt', leftdomain=1, rightdomain=0)
    geo.Append(['line', boxtoprgt, boxtoplft],
               bc='top', leftdomain=1, rightdomain=0, maxh=htop)
    geo.Append(['line', boxtoplft, ellipsetop],
               bc='axis', leftdomain=1, rightdomain=0)
    geo.Append(["spline3", ellipsetop, ellipsectlu, ellipsergt],
               bc='cavity', leftdomain=1, rightdomain=0, maxh=hcavity)
    geo.Append(["spline3", ellipsergt, ellipsectld, ellipsebot],
               bc='cavity', leftdomain=1, rightdomain=0, maxh=hcavity)

    return geo
