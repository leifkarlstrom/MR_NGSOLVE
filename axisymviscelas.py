from cavitygeometry import region_outside_cavity
import ngsolve as ng
from ngsolve import dx, ds, grad, BND, InnerProduct
from ngsolve import CoefficientFunction as CF
from ngsolve.internal import visoptions
from ngsolve import SetVisualization


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
    if not isinstance(u, CF):
        raise ValueError('Function ε(u) only works when u ' +
                         'is a CoefficientFunction! type(u)=%s' % type(u))
    drur = ur.Diff(r)  # Diff works for CF (grad works for test/trial fn)
    dzur = ur.Diff(z)
    druz = uz.Diff(r)
    dzuz = uz.Diff(z)
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

    def __init__(self, mu=0.5, lam=4.0, tau=1, A=4, B=4, D=5, L=10,
                 hcavity=0.5, hglobal=1, p=2,  tractionBCparts='',
                 kinematicBCparts='axis|cavity|top|rgt|bot',
                 refine=0, curvedegree=2):

        self.geometry = region_outside_cavity(A, B, D, L, hcavity)
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

        self.mu = mu
        self.lam = lam
        self.tau = tau

        all = self.mesh.Boundaries(tractionBCparts) + \
            self.mesh.Boundaries(kinematicBCparts)
        if sum(~all.Mask()) != 0:
            raise ValueError('All boundaries must be included in ' +
                             'tractionBCparts union kinematicBCparts')
        self.σbdry = tractionBCparts
        self.ubdry = kinematicBCparts
        self.ubdry_noaxis = self.subtract_axis(self.ubdry)

        # Displacement space for u = (ur, uz)
        Vr = ng.H1(self.mesh, order=p, dirichlet=self.ubdry)
        Vz = ng.H1(self.mesh, order=p, dirichlet=self.ubdry_noaxis)
        self.U = ng.FESpace([Vr, Vz])

        # Space for c = (c_rr, c_rz, c_zz, c_θθ)
        L = ng.L2(self.mesh, order=p)
        #                    c_rr  c_rz  c_zz  c_θθ
        self.S = ng.FESpace([L,     L,    L,    L])

        self.SU = ng.FESpace([self.S, self.U])

        print('  Setting up static elastic system')
        u, v = self.U.TnT()
        ur = u[0]
        vr = v[0]
        a = ng.BilinearForm(self.U, symmetric=True)
        a += 2 * self.mu * InnerProduct(εrz(u), εrz(v)) * r * drdz
        a += 2 * self.mu * rinv * ur * vr * drdz
        a += self.lam * (r * divrz(u) + ur) * (divrz(v) + rinv * vr) * drdz

        with ng.TaskManager():
            a.Assemble()
            self.a = a
            self.ainv = a.mat.Inverse(freedofs=self.U.FreeDofs())

    def subtract_axis(self, regexp):
        return regexp.replace('|axis|', ''). \
            replace('|axis', '').replace('axis|', '')

    # Material tensor manipulations #####################################

    def Av(self, s):
        """ Viscous compliance tensor Av applied to s:
        Av(s) = (dev s) / (2μτ)
        """
        return dev(s) / (2 * self.mu * self.tau)

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

    def primalsolve(self, F=None, kinematicBC=None, tractionBC=None):
        """ Solve the standard axisymmetric displacement formulation
        for the linear (not viscoelastic) elastic problem with input
        load (vector F) and boundary data (inpu tin kinematicBC and/or
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
        This function numerically solves for c(r, z, t) and u(r, z, t)
        satisfying
            c' = Ce Av (Ce ε(u) - c) + G(t),
            div(Ce ε(u)) = F(t) + div(c),
        starting from time 0, with initial iterates "u0", "c0", proceeding
        until final time "tfin" is reached. Here u denotes the total
        (elastic+viscous) displacement, c = Ce ε(uviscous), and the
        above equations are a reformulation of the equations of
        the Maxwell viscoelastic model.  Boundary conditions imposed are

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

        OUTPUTS: cu, uht, cht, σht
         - cu  is a composite grid function containing both c and u
        at the last time step,
         - uht, cht, σht contains the time history of displacement, c,
        and stress, throughout the simulation, but skipping every "skip"
        timesteps.

        """

        dt = tfin / nsteps
        if t is None:
            t = ng.Parameter(0.0)
        t.Set(0.0)

        # Make the form  (Ce Av (Ce ε(u) - c) + G,  s)ᵣ
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

        # Make the form  (-F - div c,  v)ᵣ after integrating by parts
        u, v = self.U.TnT()
        vr, vz = v
        v = tuple(v)
        c = self.S.TrialFunction()
        n = ng.specialcf.normal(self.mesh.dim)

        d = ng.BilinearForm(trialspace=self.S, testspace=self.U,
                            nonassemble=True)
        dvol = ip(c, rε(v))
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
            sbdr = frc * v * r
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
        uht = ng.GridFunction(V, name='displacement', multidim=0)
        σht = ng.GridFunction(L, name='stress', multidim=0)
        cht = ng.GridFunction(L, name='c', multidim=0)
        uh = ng.GridFunction(V, name='displacement')
        σh = ng.GridFunction(L, name='stress')
        ch = ng.GridFunction(L, name='c')

        urz = CF((u.components[0], u.components[1]))
        uh.Set(urz)
        σh.Set(self.Ce(ε(urz)))
        crr, crz, czz, cθθ = c.components
        ch.Set(CF((crr, crz, czz, cθθ)))
        uht.AddMultiDimComponent(uh.vec)
        cht.AddMultiDimComponent(ch.vec)
        σht.AddMultiDimComponent(σh.vec)

        if draw:
            tr = ng.GridFunction(ng.L2(self.mesh, order=self.p),
                                 name='trace(stress)')
            tr.Set(σh[0] + σh[2] + σh[3])
            ng.Draw(uh)
            ng.Draw(tr)
            visoptions.vecfunction = 'displacement'
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

                # Replace u by numerical solution of div(Ce ε(u)) = f + div(c)
                d.Apply(c.vec, w)
                if tractionBC is not None:
                    σnv.Assemble()
                    w += σnv.vec
                self.setkinematicbc(u, kinematicBC)
                self.staticsolve(w, u)

                # Store for output and visualize
                if (i+1) % skip == 0:
                    urz = CF((u.components[0], u.components[1]))
                    uh.Set(urz)
                    σh.Set(self.Ce(ε(urz)))
                    crr, crz, czz, cθθ = c.components
                    ch.Set(CF((crr, crz, czz, cθθ)))

                    uht.AddMultiDimComponent(uh.vec)
                    cht.AddMultiDimComponent(ch.vec)
                    σht.AddMultiDimComponent(σh.vec)

                    if draw:
                        tr.Set(σh[0] + σh[2] + σh[3])
                        ng.Redraw()

        print('\nSimulation done.')
        return cu, uht, cht, σht
