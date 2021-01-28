from .cavitygeometry import region_outside_cavity
import ngsolve as ng
from ngsolve import dx, ds, grad, BND, InnerProduct
from ngsolve import CoefficientFunction as CF
from ngsolve.internal import visoptions
from ngsolve import SetVisualization
import time


# Things often used in this scope ######################################

r = ng.x
z = ng.y
drdz = dx(bonus_intorder=1)  # increase order anticipating r-factor
threshold = 1.e-15  # Use thresholding to avoid division by zero in 1/r
rinv = 1.0 / ng.IfPos(r - threshold, r, threshold)


# Tensor and vector shorthands #########################################

def fip(a, b):
    """ Symmetric matrices of the form
            c = [c_rr  c_rz    0]
                [c_rz  c_zz    0]
                [0     0    c_θθ]
    are represented by the 4-vector (c_rr, c_rz, c_zz, c_θθ). This method
    returns the Frobenius inner product of symmetric matrices a, b
    both represented as 4-vectors. """

    arr, arz, azz, aθθ = a
    brr, brz, bzz, bθθ = b
    return arr*brr + 2*arz*brz + azz*bzz + aθθ*bθθ


def ip(a, b):
    """ Return the vector inner product, not Frobenius product
    (more convenient when using SolveM). """

    arr, arz, azz, aθθ = a
    brr, brz, bzz, bθθ = b
    return arr*brr + arz*brz + azz*bzz + aθθ*bθθ


def mat3x3(s):
    """ Return the matrix representation of a 4-vector. """

    srr, srz, szz, sθθ = s
    return CF((srr,   srz,      0,
               srz,   szz,      0,
               0,        0,   sθθ),  dims=(3, 3))


def mat2x2(s):
    """ Return matrix representation without hoop component. """

    srr, srz, szz, sθθ = s
    return CF((srr,   srz,
               srz,   szz),  dims=(2, 2))


def dev(s):
    """ Return deviatoric  dev(s) = s - (tr s/3) I. """

    srr, srz, szz, sθθ = s
    t = (srr + szz + sθθ)/3
    return CF((srr-t, srz, szz-t, sθθ-t))


def εrz(u):
    """ Return 2x2 linearized strain in the rz plane. """

    ur, uz = u
    drur, dzur = grad(ur)
    druz, dzuz = grad(uz)
    return CF((drur,          (druz+dzur)/2,
               (druz+dzur)/2,         dzuz),   dims=(2, 2))


def ε(u):
    """ 3D linearized strain:  ε(ur, uz) = [εrz(ur, uz)   0]
                                           [0          ur/r].
    This only works when "u" is a CF expressed in terms of r and z. """

    ur, uz = u
    drur, dzur = grad(ur)
    druz, dzuz = grad(uz)
    return CF((drur, (druz+dzur)/2, dzuz, ur * rinv))


def rε(u):
    """ Return r * ε(u), simplified. """

    ur, uz = u
    drur, dzur = grad(ur)
    druz, dzuz = grad(uz)
    return CF((r * drur, r * (druz+dzur)/2, r * dzuz, ur))


def divrz(u):
    """ Return 2D divergence in the rz plane (not actual 3D div). """

    ur, uz = u
    drur, dzur = grad(ur)
    druz, dzuz = grad(uz)
    return CF(drur + dzuz)


# Class for Axisymmetric Viscoelastic Maxwell Model ####################


class AxisymViscElas:

    """Class implementing axisymmetric Maxwell viscoelastic model of
    a region outside a magma cavity.
    """

    def __init__(self, mu=0.5, lam=4.0, tau=1, A=4, B=4, D=5, Lr=10, Lz=None,
                 hcavity=0.5, hglobal=1, p=2,  tractionBCparts='',
                 kinematicBCparts='axis|cavity|top|rgt|bot',
                 refine=0, curvedegree=2):
        """
        INPUTS:
           mu, lam, tau: Lame elastic parameters (mu, lam) and
        viscoelastic relaxation time (tau = eta/mu, where eta=viscosity).

           A, B, D, Lr, Lz: Magma cavity is ellipsoidal of r-semimajor
        axis length A, z-semiminor axis B. Terrain is at distance D from
        the center of the ellipse, which is taken as the origin. The
        enclosing box has vertices (0, B+D), (Lr, B+D), (0, -Lz), (Lr, -Lz).

           hcavity, hglobal, refine: maximal grid spacing near cavity (hcavity)
        and over the whole mesh (hglobal), for initial mesh construction.
        Initial mesh is refined (refine) many times before computations
        unless refine=0.

           p: finite element degree

           tractionBCparts, kinematicBCparts: strings made from named
        boundaries where boundary conditions are to be imposed.
        """

        self.geometryparams = {'A': A, 'B': B, 'D': D, 'Lr': Lr, 'Lz': Lz,
                               'hcavity': hcavity, 'hglobal': hglobal,
                               'curvedegree': curvedegree}
        self.geometry = region_outside_cavity(A, B, D, Lr, hcavity, Lz)
        ngmesh = self.geometry.GenerateMesh(maxh=hglobal)
        for i in range(refine):
            print('  Refining mesh')
            ngmesh.Refine()
        self.mesh = ng.Mesh(ngmesh)
        self.mesh.ngmesh.SetGeometry(self.geometry)
        if curvedegree > 0:
            self.mesh.Curve(curvedegree)
        ng.Draw(self.mesh)
        self.p = p

        all = self.mesh.Boundaries(tractionBCparts) + \
            self.mesh.Boundaries(kinematicBCparts)
        if sum(~all.Mask()) != 0:
            raise ValueError('All boundaries must be included in ' +
                             'tractionBCparts union kinematicBCparts')
        self.σbdry = tractionBCparts
        self.ubdry = kinematicBCparts
        self.ubdry_noaxis = self.subtract_axis(self.ubdry)

        if mu is not None and lam is not None and tau is not None:
            self.mu = CF(mu)
            self.lam = CF(lam)
            self.tau = CF(tau)
            self.matready = True
            self.initFEfacilities()
        else:
            self.matready = False

    def setmaterials(self, mu, lam, tau):
        """Input material parameters as (possibly spatially varying)
        ngsolve coefficient functions or grid functions.        """

        self.matready = True
        self.mu = CF(mu)
        self.lam = CF(lam)
        self.tau = CF(tau)
        self.initFEfacilities()

    def initFEfacilities(self):
        """ some reusable initializations """

        # Displacement space for u = (ur, uz)
        Vr = ng.H1(self.mesh, order=self.p, dirichlet=self.ubdry)
        Vz = ng.H1(self.mesh, order=self.p, dirichlet=self.ubdry_noaxis)
        self.U = ng.FESpace([Vr, Vz])

        # Space for c = (c_rr, c_rz, c_zz, c_θθ)
        L = ng.L2(self.mesh, order=self.p)
        #                    c_rr  c_rz  c_zz  c_θθ
        self.S = ng.FESpace([L,     L,    L,    L])

        self.SU = ng.FESpace([self.S, self.U])

        print('  Setting up static elastic system')
        u, v = self.U.TnT()
        a = ng.BilinearForm(self.U, symmetric=True)
        a += fip(self.Ce(rε(u)), ε(v)) * drdz

        with ng.TaskManager():
            a.Assemble()
            self.a = a
            self.ainv = a.mat.Inverse(freedofs=self.U.FreeDofs())

    def subtract_axis(self, regexp):
        return regexp.replace('|axis|', ''). \
            replace('|axis', '').replace('axis|', '')

    # Material tensor manipulations #####################################

    def Ce(self, s):
        """ Elastic stiffness tensor Ce applied to s:
        Ce(s) = 2μ dev(s) + K(tr s) I, which using K = λ + (2μ/3) becomes
        Ce(s) = 2μ s + λ(tr s) I.
        """
        srr, srz, szz, sθθ = s
        t = srr + szz + sθθ
        return CF((2 * self.mu * srr + self.lam * t,
                   2 * self.mu * srz,
                   2 * self.mu * szz + self.lam * t,
                   2 * self.mu * sθθ + self.lam * t))

    def CeAv(self, s):
        """ Return Ce(Av(s)) in simplified form: Ce(Av(s)) = (dev s) / τ.
        """
        return dev(s) / self.tau

    # Computational routines ############################################

    def temperature(self, temperatureBC, kappa=1):
        """
        Solve for a steady temperature distribution with boundary conditions
        on each named boundary given in temperatureBC, e.g.,
        temperatureBC = {'cavity': 700, 'bot': 200, 'top': 20}.
        No-flux boundary condition is imposed on boundaries not appearing
        as a key in this dict. Diffusion coefficient should be input in kappa.
        """

        self.Tbdry = self.subtract_axis('|'.join(temperatureBC.keys()))
        V = ng.H1(self.mesh, order=self.p, dirichlet=self.Tbdry)
        u, v = V.TnT()
        a = ng.BilinearForm(V, symmetric=True)
        a += kappa * grad(u) * grad(v) * r * drdz
        a.Assemble()

        T = ng.GridFunction(V, name='temperature')
        T.Set(self.mesh.BoundaryCF(temperatureBC), ng.BND)
        f = T.vec.CreateVector()
        f.data = -a.mat * T.vec
        T.vec.data += a.mat.Inverse(V.FreeDofs()) * f
        self.T = T
        return T

    def primalsolve(self, F=None, kinematicBC=None, tractionBC=None):
        """ Solve the standard axisymmetric displacement formulation
        for the linear (not viscoelastic) elastic problem with input
        load (vector F) and boundary data (input in kinematicBC and/or
        tractionBC in the respective boundary parts of this object).
        """

        if len(self.σbdry) and tractionBC is None:
            raise ValueError('Traction BC must be input since ' +
                             'tractionBCparts is not empty.')

        u = ng.GridFunction(self.U)
        self.setkinematicbc(u, kinematicBC)
        f = ng.LinearForm(self.U)
        vr, vz = self.U.TestFunction()
        v = CF((vr, vz))
        n = ng.specialcf.normal(self.mesh.dim)

        if F is not None:
            f += InnerProduct(F, v) * r * drdz
        if tractionBC is not None:
            dγ = ds(skeleton=True, bonus_intorder=1,
                    definedon=self.mesh.Boundaries(self.σbdry))
            f += (tractionBC * n) * v * r * dγ

        with ng.TaskManager():
            f.Assemble()
        return self.staticsolve(f.vec, u)

    def staticsolve(self, f, u):
        with ng.TaskManager():
            f.data -= self.a.mat * u.vec
            u.vec.data += self.ainv * f
        return u

    def setkinematicbc(self, u, kinematicBC=None):
        with ng.TaskManager():
            if kinematicBC is not None:
                u.components[0].Set(kinematicBC[0], BND)
                u.components[1].Set(kinematicBC[1], BND)
            else:
                u.vec[:] = 0.0

    def solve2(self, tfin, nsteps, u0, c0,
               t=None, F=None, kinematicBC=None, tractionBC=None, G=None,
               draw=False, skip=1):
        """
        This function numerically solves for
               u(r, z, t) = uviscous + uelastic, and
               c(r, z, t) =  Ce ε(uviscous)
        satisfying
            c' = Ce Av (Ce ε(u) - c) + G(t),
            div(Ce ε(u)) = F(t) + div(c),
        starting from time 0, with initial iterates "u0", "c0", proceeding
        until final time "tfin" is reached.  This is a reformulation of
        the equations of the Maxwell viscoelastic model for the total
        (elastic+viscous) displacement u.

        All material parameters used here are the ones incorporated in the
        functions CeAv(..) and Ce(..).

        Boundary conditions imposed are
          u = kinematicBC         on kinematicBCparts,
          σn = tractionBC * n     on tractionBCparts,
        where kinematicBCparts and tractionBCparts are previously
        set by class constructor. Input "kinematicBC" should be a 2-vector
        CoefficientFunction  and  input "tractionBC" should be a 2x2 matrix
        CoefficientFunction  (even though only its product with outward unit
        normal n will be used). Note that the true viscoelastic stress σ is
        related to total displacement u by   Ce ε(u) = σ + c. The input
        boundary data tractionBC * n is expected to equal the
        viscoelastic force  σn = (Ce ε(u) - c) n. (Wrong results may be
        obtained if Ce ε(u) n is given in tractionBC instead.)

        The boundary and load data (F, kinematicBC, tractionBC, G)  are
        allowed to depend on an input time parameter "t" (and if "t" is
        not given, they are assumed to not depend on time).
        F should be given as a 2-vector and G should be given as
        a 4-vector to represent the 3x3 matrix of the form of c.

        OUTPUTS: cu, uht, cht, sht

         - "cu"  is a composite grid function containing both
        c = Ce ε(uviscous) and displacement u at the last time step

         - "uht, cht, sht" contains the time history of the
        displacement u,  c = Ce ε(uviscous), and s = Ce ε(u),
        throughout the simulation, but skipping every "skip" timesteps.

        - "ts" contains a list of time values corresponding to each grid
        function in the uht, cht, sht time series. (Length of "ts" may be
        shorter than number of timesteps depending on "skip".)

        Note that s = Ce ε(u) is not the true viscoelastic stress
        σ = Ce ε(uelastic). Clearly, s and σ are related by s = σ + c.
        """

        if not self.matready:
            raise Exception('Missing material parameters.')

        dt = tfin / nsteps
        if t is None:
            t = ng.Parameter(0.0)
        t.Set(0.0)
        ts = [0]

        # Make the form  (Ce Av (Ce ε(u) - c) + G,  s)ᵣ
        #    IMPORTANT:  The c system uses 4-vectors and requires
        #    the use of ip(., .), not fip(., .)
        c, u = self.SU.TrialFunction()
        s = self.S.TestFunction()
        cupdate = ip(self.CeAv(self.Ce(rε(u))), s)
        cupdate -= ip(self.CeAv(c), s) * r
        if G is not None:
            cupdate += ip(G, s) * r
        cupdate.Compile()
        b = ng.BilinearForm(trialspace=self.SU, testspace=self.S,
                            nonassemble=True)
        b += cupdate * drdz

        # Make the form d(u, v) = -(F, v)ᵣ + (c, ε(v))ᵣ
        #    IMPORTANT:  The u system requires Frobenius inner products
        #    so we use fip(.,.), not 4-vector inner product ip(.,.)
        u, v = self.U.TnT()
        vr, vz = v
        v = tuple(v)
        c = self.S.TrialFunction()
        n = ng.specialcf.normal(self.mesh.dim)

        d = ng.BilinearForm(trialspace=self.S, testspace=self.U,
                            nonassemble=True)
        dvol = fip(c, rε(v))
        if F is not None:
            dvol += -InnerProduct(F, v) * r
        dvol.Compile()
        d += dvol * drdz

        # Make traction BC source term for u update
        if tractionBC is not None:
            if len(self.σbdry) == 0:
                raise ValueError('Constructor should know where ' +
                                 'to put traction.')
            if isinstance(tractionBC, CF):
                frc = tractionBC * n
            else:
                if isinstance(tractionBC, dict):
                    f0 = CF((0, 0,
                             0, 0), dims=(2, 2))
                    frc = self.mesh.BoundaryCF(tractionBC, default=f0) * n
                else:
                    raise ValueError('Give tractionBC as a CF or a dict')
            frc.Compile()

            dγ = ds(bonus_intorder=1,
                    definedon=self.mesh.Boundaries(self.σbdry))
            σnv = ng.LinearForm(self.U)
            sbdr = InnerProduct(frc, v) * r
            sbdr.Compile()
            σnv += sbdr * dγ

        cu = ng.GridFunction(self.SU)
        w = u0.vec.CreateVector()
        s = c0.vec.CreateVector()

        cu.components[0].vec.data = c0.vec
        cu.components[1].vec.data = u0.vec
        c = cu.components[0]
        u = cu.components[1]

        # Output fields
        V = ng.VectorH1(self.mesh, order=self.p)
        L = ng.L2(self.mesh, order=self.p, dim=4)
        uht = ng.GridFunction(V, name='u', multidim=0)
        sht = ng.GridFunction(L, name='s', multidim=0)
        cht = ng.GridFunction(L, name='c', multidim=0)
        uh = ng.GridFunction(V, name='u')
        σh = ng.GridFunction(L, name='s')
        ch = ng.GridFunction(L, name='c')

        uh.Set(CF((u.components[0], u.components[1])))
        σh.Set(self.Ce(ε(u.components)))
        crr, crz, czz, cθθ = c.components
        ch.Set(CF((crr, crz, czz, cθθ)))
        uht.AddMultiDimComponent(uh.vec)
        cht.AddMultiDimComponent(ch.vec)
        sht.AddMultiDimComponent(σh.vec)

        if draw:
            tr = ng.GridFunction(ng.L2(self.mesh, order=self.p),
                                 name='trace(s)')
            tr.Set(σh[0] + σh[2] + σh[3])
            ng.Draw(uh)
            ng.Draw(tr)
            visoptions.vecfunction = 'u'
            SetVisualization(deformation=True)

        with ng.TaskManager():

            for i in range(nsteps):

                t.Set((i+1)*dt)
                print('  Timestep %3d  to reach time %g' % (i+1, t.Get()),
                      end='\r')

                # Replace c by c + dt (Ce Av (Ce ε(u) - c) + G)
                b.Apply(cu.vec, s)
                self.S.SolveM(rho=r, vec=s)
                c.vec.data += dt * s

                # Put u = solution of (Ce ε(u), ε(v))ᵣ = (c, ε(v))ᵣ - (F, v)ᵣ
                d.Apply(c.vec, w)
                if tractionBC is not None:
                    σnv.Assemble()
                    w += σnv.vec
                self.setkinematicbc(u, kinematicBC)
                self.staticsolve(w, u)

                # Store for output and visualize
                if (i+1) % skip == 0:
                    ts.append(t.Get())
                    uh.Set(CF((u.components[0], u.components[1])))
                    σh.Set(self.Ce(ε(u.components)))
                    crr, crz, czz, cθθ = c.components
                    ch.Set(CF((crr, crz, czz, cθθ)))

                    uht.AddMultiDimComponent(uh.vec)
                    cht.AddMultiDimComponent(ch.vec)
                    sht.AddMultiDimComponent(σh.vec)

                    if draw:
                        tr.Set(σh[0] + σh[2] + σh[3])
                        ng.Redraw(blocking=True)

        print('\nSimulation done.')
        return cu, uht, cht, sht, ts

    # Convenience facilities ############################################

    def reanim(self, ut, st, sfun,
               sleep=0.1, maxim=None, minim=None,
               probe=None):
        """
        Animate a previously computed time series of solutions.
        Displacements u and s = Ce ε(u) for all time are input in "ut" and
        "st", respectively.

        To display a scalar function of stress, define your function
        elsewhere and pass it as input argument "sfun".

        To use time-independent color scalings (relevent to your sfun
        values), give "maxim, minim".

        To get a list of values of your "sfun" at some points in the domain,
        give the r, z coordinates as a list in input "probe" like
        probe = [(r0, z0), (r1, z1), ...]. Then, a list "vals"
        will be output, whose i-th element is a list of values
        of sfun at the probe points.
        """

        u = ng.GridFunction(ut.space, name='u')
        s = ng.GridFunction(st.space, name='s')
        u.vec.data = ut.vecs[0]
        s.vec.data = st.vecs[0]
        sf = sfun(mat3x3(s))

        ng.Draw(u)
        ng.Draw(sf, self.mesh, 'sfun')
        visoptions.vecfunction = 'u'
        SetVisualization(deformation=True, max=maxim, min=minim)

        vals = []
        if probe is not None:
            mips = [self.mesh(x, y) for x, y in probe]

        for i in range(len(ut.vecs)):
            u.vec.data = ut.vecs[i]
            s.vec.data = st.vecs[i]
            if probe is not None:
                vals.append([sf(mip) for mip in mips])
            time.sleep(sleep)  # slow down if your cpu is too fast
            ng.Redraw()

        return vals

    def __getstate__(self):
        """Remove unpicklable attributes"""
        state = self.__dict__.copy()
        del state['a']
        del state['ainv']
        del state['S']
        del state['U']
        del state['SU']
        return state

    def __setstate__(self, state):
        """Restore unpickled attributes"""
        self.__dict__.update(state)
        print('  Rebuilding AxisymViscElas object')
        self.initFEfacilities()
        self.mesh.ngmesh.SetGeometry(self.geometry)
        if self.geometryparams['curvedegree'] > 0:
            self.mesh.Curve(self.geometryparams['curvedegree'])

    def estimatebdryminmax(self, f):
        """Give quick estimators for min & max of f on the domain boundary"""

        V1 = ng.H1(self.mesh, order=1, dirichlet='axis|cavity|top|rgt|bot')
        f1 = ng.GridFunction(V1)
        f1.Set(f, ng.BND)
        fv = f1.vec.FV().NumPy()[~V1.FreeDofs()]
        return min(fv), max(fv)
