from axisymviscelas import AxisymViscElas, r, z
import ngsolve as ng
import ngsolve.webgui
from ngsolve import CoefficientFunction as CF

ng.ngsglobals.msg_level = 1

# cavity and  material parameters
A = 3       # cavity semi-major axis
B = 2       # cavity semi-minor axis
D = 10      # cavity depth beneath Earth's surface
L = 15      # domain length along r-axis
μ = 0.5     # μ and λ are Lamé parameters
λ = 4.0
τ = 0.2     # relaxation time
p = 5       # order of FESpace

ave = AxisymViscElas(mu=μ, lam=λ, tau=τ, A=A, B=B, D=D, L=L, p=p,
                     hcavity=1,  hglobal=4,
                     tractionBCparts='cavity|top|rgt',
                     kinematicBCparts='axis|bot')

# time
t = ng.Parameter(0.0)

# time-varying force on top boundary: consider a suddent uniform top
# loading starting around some specific time into the simulation:
Q = -0.1 * ng.exp(-100 * (t-0.5)**2)

# Give stress boundary conditions as a 2x2 (r, z) tensor s such that s * n
# equals your desired loading force
toptraction = CF((0, 0,
                  0, Q), dims=(2, 2))

# sinusoidal pressure to be imposed on cavity boundary
α = 0.1        # amplitude
ω = 20         # angular frequency
P = α * (1 + ng.sin(ω * t))
ρ = ng.sqrt(r*r + z*z)
sinφ = r / ρ
cosφ = z / ρ
n = ng.specialcf.normal(ave.mesh.dim)
sn = (P * sinφ, P * cosφ)
cavitytraction = CF((sn[0]*n[0], sn[0]*n[1],
                     sn[1]*n[0], sn[1]*n[1]), dims=(2, 2))

# Specify traction bc on various named boundaries
traction = {'cavity': cavitytraction, 'top': toptraction}

# initial data
u0 = ng.GridFunction(ave.U)
c0 = ng.GridFunction(ave.S)
c0.vec[:] = 0
u0.vec[:] = 0

# simulate
cu, uht, cht, sht = ave.solve2(tfin=1, nsteps=200, u0=u0, c0=c0, t=t,
                               tractionBC=traction, draw=True)
