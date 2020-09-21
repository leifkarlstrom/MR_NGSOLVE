from axisymviscelas import AxisymViscElas, r, z
import ngsolve as ng
from ngsolve import CoefficientFunction as CF
ng.ngsglobals.msg_level = 1

"""
Model a pressurized spherical cavity in a viscoelastic medium.
Pressure along the cavity wall is sinusoidal and of the form
    P(t) = α sin(ωt)
for amplitude α and angular frequency ω.
"""

# cavity and  material parameters
A = 4       # cavity semi-major axis
B = 4       # cavity semi-minor axis
D = 10      # cavity depth beneath Earth's surface
L = 15      # domain length along r-axis
μ = 0.5     # μ and λ are lamé parameters
λ = 4.0
τ = 1       # relaxation time
p = 3       # order of FESpace


ave = AxisymViscElas(mu=μ, lam=λ, tau=τ, A=A, B=B, D=D, L=L, p=p,
                     hcavity=0.5,  hglobal=4,
                     tractionBCparts='cavity',
                     kinematicBCparts='axis|top|rgt|bot')

# Time as a parameter whose value can be set later
t = ng.Parameter(0.0)

# sinusoidal pressure to be imposed on cavity boundary
α = 0.5        # amplitude
ω = 10          # angular frequency
P = α * (1 + ng.sin(ω * t))

# starting from the cavity boundary conditions written in spherical (ρ, φ, θ)
# we know that the cavity b.c must satisfy
#   σρρ = -P
#   σρθ = σρφ = 0
#
# then, by a change of coordinates to cylindrical (r, z, θ) we have
#   sinφ  cosφ 0| -P        -P sinφ
#   cosφ -sinφ 0|  0     =  -P cosα
#    0     0   1|  0            0
ρ = ng.sqrt(r*r + z*z)
sinφ = r / ρ
cosφ = z / ρ

# Pressure on the cavity
n = ng.specialcf.normal(ave.mesh.dim)
sn = (P * sinφ, P * cosφ)
σBC = CF((sn[0]*n[0], sn[0]*n[1],
          sn[1]*n[0], sn[1]*n[1]), dims=(2, 2))

# boundary data for remote and depth boundaries
uBC = CF((0, 0))

# initial data (on displacement and viscous stress)
u0 = ng.GridFunction(ave.U)
c0 = ng.GridFunction(ave.S)

u0.components[0].Set(0)
u0.components[1].Set(0)

crr, crz, czz, cθθ = c0.components
crr.Set(0)
crz.Set(0)
czz.Set(0)
cθθ.Set(0)


# simulate
cu = ave.solve2(tfin=1, nsteps=100, u0=u0, c0=c0, t=t, tractionBC=σBC,
                draw=True)
