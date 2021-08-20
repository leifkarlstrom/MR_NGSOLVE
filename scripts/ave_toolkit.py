from magmaxisym import AxisymViscElas, r, z
import ngsolve as ng
import numpy as np
from ngsolve import CoefficientFunction as CF
from ngsolve.special_functions import erf


class AveToolkit(AxisymViscElas):
    """Class implementing facilities to aid in parameter studies."""

    def __init__(self, mu=0.5, lam=4.0, eta=1.0, tau=2.0, om=1e-7, A=2000,
                 B=2000, D=5000, Lr=20000, Lz=None, p=4, refine=1, hcavity=4,
                 hglobal=4, curvedegree=2, tractionBCparts='cavity|top',
                 kinematicBCparts='axis|rgt|bot'):
        """Doc."""
        self.length_scale = A
        self.time_scale = om
        self.stress_scale = 1  # 10e6
        self.T = None

        # scale geometric parameters
        A = A / self.length_scale
        B = B / self.length_scale
        D = D / self.length_scale
        Lr = Lr / self.length_scale

        if Lz is not None:
            Lz = Lz / self.length_scale

        # scale material parameters
        mu = CF(mu / self.stress_scale)
        lam = CF(lam / self.stress_scale)

        # scale time parameter
        tau = CF(tau * self.time_scale)

        super().__init__(mu=mu, lam=lam, eta=eta, tau=tau, om=om, A=A, B=B,
                         D=D, Lr=Lr, Lz=Lz, p=p, refine=refine,
                         hcavity=hcavity, hglobal=hglobal,
                         curvedegree=curvedegree,
                         tractionBCparts=tractionBCparts,
                         kinematicBCparts=kinematicBCparts)

        self.K = 2 * np.pi * 5

        # Time as a parameter whose value can be set later
        self.t = ng.Parameter(0.0)

        # cavity pressure
        P = (ng.sin(self.t))

        # starting from the cavity boundary conditions written in spherical
        # (ρ, φ, θ) we know that the cavity b.c must satisfy
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
        n = ng.specialcf.normal(self.mesh.dim)
        sn = (P * sinφ, P * cosφ)
        cavitytraction = CF((sn[0]*n[0], sn[0]*n[1],
                             sn[1]*n[0], sn[1]*n[1]), dims=(2, 2))

        toptraction = CF((0, 0,
                          0, 0), dims=(2, 2))

        self.σBC = {'cavity': cavitytraction, 'top': toptraction}
        # boundary data for remote and depth boundaries
        self.uBC = CF((0, 0))

        # initial data (on displacement and viscous stress)
        self.u0 = ng.GridFunction(self.U)
        self.c0 = ng.GridFunction(self.S)

        self.u0.components[0].Set(0)
        self.u0.components[1].Set(0)

        crr, crz, czz, cθθ = self.c0.components
        crr.Set(0)
        crz.Set(0)
        czz.Set(0)
        cθθ.Set(0)

    def update_class(self):
        P = self.geometryparams
        super().__init__(mu=self.mu, lam=self.lam, eta=self.eta, tau=self.tau,
                         om=self.om, A=P['A'], B=P['B'], D=P['D'], Lr=P['Lr'],
                         Lz=P['Lz'], p=4, refine=1, hcavity=P['hcavity'],
                         hglobal=P['hglobal'],
                         curvedegree=P['curvedegree'],
                         tractionBCparts='cavity|top',
                         kinematicBCparts='axis|rgt|bot')

    def scale_geo_params(self):
        """Nondimensionalize the system parameters."""
        P = self.geometryparams
        # scale geometric parameters by cavity radius A
        P['A'] = P['A'] / self.length_scale
        P['B'] = P['B'] / self.length_scale
        P['D'] = P['D'] / self.length_scale
        P['Lr'] = P['Lr'] / self.length_scale

        if P['Lz'] is not None:
            P['Lz'] = P['Lz'] / self.length_scale

    def geotherm_bvp(self, Tₗ=1000, Tₛ=25, grad=20):
        """Solve an equilibrium geotherm boundary value problem.

        INPUTS:
            Tₗ: liquidus temperature of magma (ᵒC)
            Tₛ: temperature at Earth's surface (ᵒC)
            ΔT: rate of temperature increase with resepect to depth (ᵒC/km)
        """
        # equilibrium geotherm boundary conditions
        params = self.geometryparams
        cavity_depth = params['D'] + params['B']

        # scale ΔT to account for the domain being scaled
        ΔT = grad * (self.length_scale / 1000)

        lateral_bd = Tₛ - (z - cavity_depth)*ΔT

        # get temperatures as the solution to a steady state diffusion problem
        self.T = self.temperature(temperatureBC={'cavity': Tₗ,
                                  'bot|top|rgt': lateral_bd}, kappa=2)

    def temperature_dependence(self, geotherm_grad=20, threshold=600):
        """Set temperature dependent material parameters for the system."""

        self.geotherm_bvp(grad=geotherm_grad)

        # set the temp. threshold for elastic behavior; anything colder than
        # this temp will get the same viscosity
        T2 = ng.IfPos(threshold-self.T, threshold, self.T)

        # parameters for temperature-dependent viscosity
        B = 8.31        # molar gas constant
        G = 141e3       # activation energy for creep
        A = 4.25e7      # material dependent constant for viscosity law

        η = A * ng.exp(G / (B * T2))  # Arhennius Law for viscosity

        # parameters for fitting temperature-dependent Young's modulus
        a = 1.85e10
        b = -3.5e6
        c = 4.3e9
        Tc = 924
        s = 120

        Eₘₐₓ = 4e10
        νₘₐₓ = 0.49
        νₘᵢₙ = 0.25

        E = a * (1 - (erf(self.T - Tc).real / s)) + b * self.T + c

        ν = (1 - E / Eₘₐₓ) * (νₘₐₓ - νₘᵢₙ) + νₘᵢₙ

        # determine temperature-dependent Lame parameters λ, μ from
        # Young's modulus E and Poisson's ratio ν
        λ = CF((1/self.stress_scale) * E * ν / ((1 + ν) * (1 - 2*ν)))
        μ = CF((1/self.stress_scale) * E / (2 * (1 + ν)))
        τ = CF(η / (μ * self.stress_scale))
        De = CF(τ * self.time_scale)
        self.setmaterials(μ, λ, De)

    def run_simulation(self):
        cu, uht, cht, sht, ts = self.solve2(tfin=self.K, u0=self.u0,
                                            c0=self.c0, t=self.t,
                                            kinematicBC=self.uBC,
                                            tractionBC=self.σBC, draw=True)
        return cu, uht, cht, sht, ts

    def set_domain_geometry(self, a=None, b=None, d=None, lr=None, lz=None):
        """Manually set parameters for the domain geometry.

        INPUTS:
            a:  horizontal axis of the reservoir ellipse.
            b:  vertical axis of the reservoir ellipse.
            d:  distance from Earth's surface to top of the reservoir.
            lr: maximum domain distance in radial direction.
            lz: maximum domain depth.
        """
        P = self.geometryparams
        if a is None:
            P['A'] = P['A'] * self.length_scale
        else:
            P['A'] = a

        if b is None:
            P['B'] = P['B'] * self.length_scale
        else:
            P['B'] = b

        if d is None:
            P['D'] = P['D'] * self.length_scale
        else:
            P['D'] = d

        if lr is None:
            P['Lr'] = P['Lr'] * self.length_scale
        else:
            P['Lr'] = lr

        if lz is None and P['Lz'] is not None:
            P['Lz'] = P['Lz'] * self.length_scale
        else:
            P['Lz'] = lz

        self.length_scale = P['A']
        self.scale_geo_params()
        self.update_class()
        self.geotherm_bvp()

    def draw_parameters(self):
        ng.Draw(self.eta, self.mesh, 'viscosity')
        ng.Draw(self.mu, self.mesh, 'shear_modulus')
        ng.Draw(self.T, self.mesh, 'temperature')
