from magmaxisym import AxisymViscElas, r, z
import ngsolve as ng
from ngsolve import CoefficientFunction as CF
ng.ngsglobals.msg_level = 1

"""
Model a pressurized spherical cavity in a temperature
dependent viscoelastic medium. Pressure along the cavity wall is
sinusoidal and of the form   P(t) = α sin(ωt).
"""

# Cavity and  material parameters

A = 2       # cavity semi-major axis
B = 2       # cavity semi-minor axis
D = 10      # cavity depth beneath Earth's surface
L = 15      # domain length along r-axis
μ = 0.5     # μ and λ are lamé parameters
λ = 4.0
p = 4       # order of FESpace


# Initialize AxisymViscElas object without material parameters

ave = AxisymViscElas(mu=None, lam=λ, tau=None, A=A, B=B, D=D, Lr=L, p=p,
                     hcavity=1,  hglobal=4,
                     tractionBCparts='cavity',
                     kinematicBCparts='axis|top|rgt|bot')

# Solve the temperature problem

T = ave.temperature({'cavity': 700,
                     'rgt': ((B+D-z)*200 + 20*(z+L))/(L+B+D),
                     'bot': 200,
                     'top': 20})
ng.Draw(T)

# Set temperature dependent material properties μ, λ, tau

tau = 10 * ng.exp(50*((1/T) - (1.0/20)))
ng.Draw(tau, ave.mesh, 'tau')
taumin, taumax = ave.estimatebdryminmax(tau)
print('Note tau min & max: ', taumin, taumax)
ave.setmaterials(μ, λ, tau)

# Time as a parameter whose value can be set later

t = ng.Parameter(0.0)

# Sinusoidal pressure to be imposed on cavity boundary

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

# Boundary data for remote and depth boundaries

uBC = CF((0, 0))

# Initial data (on displacement and viscous stress)
u0 = ng.GridFunction(ave.U)
c0 = ng.GridFunction(ave.S)

u0.components[0].Set(0)
u0.components[1].Set(0)

crr, crz, czz, cθθ = c0.components
crr.Set(0)
crz.Set(0)
czz.Set(0)
cθθ.Set(0)


# Simulate
cu, uht, cht, sht, ts = ave.solve2(tfin=1, u0=u0, c0=c0, nsteps=100, t=t,
                                   tractionBC=σBC, draw=True)
