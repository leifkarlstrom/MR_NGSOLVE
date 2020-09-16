from axisymviscelas import AxisymViscElas, r, z, rinv, ε
import ngsolve as ng
from ngsolve import CoefficientFunction as CF
ng.ngsglobals.msg_level = 1

"""
Model a pressurized sphere in a viscoelastic half-space. Pressure along the
cavity wall is sinusoidal and of the form
    P(t) = αsin(ωt)
for amplitude α and angular frequency ω.
"""

# cavity and  material parameters
A = 4       # cavity smi-major axis
B = 4       # cavity semi-minor axis
D = 5       # cavity depth beneath Earth's surface
L = 10      # domain length along r-axis

μ = 0.5     # μ and λ are lamé parameters
λ = 4.0
τ = 1.0     # viscosity?

p = 3       # order of FESpace
refine = 0  # mesh refinement level

ave = AxisymViscElas(mu=μ, lam=λ, tau=τ, A=A, B=B, D=D, L=L, p=p, refine=refine,
                         hcavity=4, hglobal=4, curvedegree=0,
                         tractionBCparts='cavity', kinematicBCparts='axis|top|rgt|bot')

# Time as a parameter whose value can be set later
t = ng.Parameter(0.0)

# sinusoidal pressure to be imposed on cavity boundary
α = 1.0         # amplitude
ω = 1.0         # angular frequancy
P = α * ng.sin(ω * t)

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

σBC = CF( (-P*sinφ,  0,
                 0, -P * cosφ), dims=(2,2))

# boundary data for remote and depth boundaries
uBC = CF((0, 0))

# initial data (on displacement and viscous stress)
u0 = ng.GridFunction(ave.U)
c0 = ng.GridFunction(ave.S)

u0.components[0].Set( 0 )
u0.components[1].Set( 0 )

crr, crz, czz, cθθ = c0.components
crr.Set( 0 )
crz.Set( 0 )
czz.Set( 0 )
cθθ.Set( 0 )

# itty-bitty time interval so temporal error does not creep in when
# observing the spatial error
T = 1e-7

cu = ave.solve2(tfin=T, nsteps=1, u0=u0, c0=c0, t=t,tractionBC=σBC,
                    kinematicBC=uBC)

cₕ = cu.components[0]
uₕ = cu.components[1]

uᵣ = uₕ.components[0]
u𝑧 = uₕ.components[1]

# plot the approximate displacements
t.Set(T)
ng.Draw(uᵣ)
ng.Draw(u𝑧)
