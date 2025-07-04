from magmaxisym import AxisymViscElas, r, z
import ngsolve as ng
from ngsolve import CoefficientFunction as CF
import numpy as np
ng.ngsglobals.msg_level = 1

"""
Model a pressurized spherical cavity in a viscoelastic medium.
Pressure along the cavity wall is sinusoidal and of the form
    P(t) = P₀ sin(ωt)
for amplitude P₀ and angular frequency ω.
"""

# cavity and  material parameters
A = 4             # cavity semi-major axis
B = 4             # cavity semi-minor axis
D = 10            # cavity depth beneath Earth's surface
L = 15            # domain length along r-axis
μ = 0.5           # μ and λ are lamé parameters
λ = 4.0
η = 0.5           # viscosity
τ = η / μ         # relaxation time
p = 3             # order of FESpace

# sinusoidal pressure parameters
P0 = 0.1          # amplitude
ω = 10            # angular frequency
K = 4*2*np.pi     # some scaling of the frequency for determining max run time;
# t∈[0, ωK], then t̃∈[0, K]

# scale geometric parameters by cavity radius A
Ã = A / A
B̃ = B / A
D̃ = D / A
L̃ = L / A

# scale material coefficients μ, λ
ε0 = 1             # scaling of strain s.t ε = ε0̃ε
μ̃ = (ε0 / P0) * μ
λ̃ = (ε0 / P0) * λ

# scale characteristic relaxation by pressure frequency
De = τ * ω         # Deborah number

# number of steps required for stability of Forward Euler
N = int(K / (2 * De)) + 1

ave = AxisymViscElas(mu=μ̃, lam=λ̃, tau=De, A=Ã, B=B̃, D=D̃, Lr=L̃, p=p,
                     hcavity=0.5,  hglobal=4,
                     tractionBCparts='cavity|top',
                     kinematicBCparts='axis|rgt|bot',
                     refine=1, curvedegree=2)

# Time as a parameter whose value can be set later
t = ng.Parameter(0.0)

# cavity pressure
P = (ng.sin(t))


# starting from the cavity boundary conditions written in spherical (ρ, φ, θ)
# we know that the cavity b.c must satisfy
#   σρρ = -P
#   σρθ = σρφ = 0
#
# then, by a change of coordinates to cylindrical (r, z, θ) we have
#   sinφ  cosφ 0| -P        -P sinφ
#   cosφ -sinφ 0|  0     =  -P cosφ
#    0     0   1|  0            0
ρ = ng.sqrt(r*r + z*z)
sinφ = r / ρ
cosφ = z / ρ

# Pressure on the cavity
n = ng.specialcf.normal(ave.mesh.dim)
sn = (P * sinφ, P * cosφ)
cavitytraction = CF((sn[0]*n[0], sn[0]*n[1],
                     sn[1]*n[0], sn[1]*n[1]), dims=(2, 2))

toptraction = CF((0, 0,
                  0, 0), dims=(2, 2))

σBC = {'cavity': cavitytraction, 'top': toptraction}
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
cu = ave.solve2(tfin=K, u0=u0, c0=c0, nsteps=200, t=t, tractionBC=σBC,
                draw=True)
