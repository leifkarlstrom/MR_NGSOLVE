from cavitygeometry import region_outside_cavity
import ngsolve as ng
from ngsolve import dx, ds, grad, BND, InnerProduct, sqrt
from ngsolve import CoefficientFunction as CF


# Things often used in this scope ######################################

r = ng.x
z = ng.y
drdz = dx(bonus_intorder=1)  # increase order anticipating r-factor
threshold = 1.e-15  # Use thresholding to avoid division by zero in 1/r
rinv = 1.0 / ng.IfPos(r - threshold, r, threshold)


# Tensor and vector shorthands #########################################

def ip(a, b):
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


def mat3x3(s):
    """ Return the matrix representation of a 4-vector. """

    srr, srz, szz, sθθ = s
    return CF((srr,   srz,      0,
               srz,   szz,      0,
               0,        0,   sθθ),  dims=(3, 3))


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
                                           [0          ur/r].  """
    ur, uz = u
    drur, dzur = grad(ur)
    druz, dzuz = grad(uz)
    return mat3x3((drur, (druz+dzur)/2, dzuz, ur * rinv))


def rε(u):
    """ Return r * ε(u), simplified. """

    ur, uz = u
    drur, dzur = grad(ur)
    druz, dzuz = grad(uz)
    return CF((r * drur, r * (druz+dzur)/2, r * dzuz, ur))


def divrz(u):
    """ Return 2D divergence in the rz plane(not actual 3D div). """

    ur, uz = u
    drur, dzur = grad(ur)
    druz, dzuz = grad(uz)
    return CF(drur + dzuz)


class AxisymViscElas:

    """Class implementing axisymmetric Maxwell viscoelastic model of
    a region outside a magma cavity.
    """

    def __init__(self, mu=0.5, lam=4.0, tau=1, A=4, B=4, D=5, L=10,
                 hcavity=0.5, hglobal=1, p=2,  tractionBCparts='',
                 kinematicBCparts='axis|cavity|top|rgt|bot'):

        self.geometry = region_outside_cavity(A, B, D, L, hcavity)
        self.mesh = ng.Mesh(self.geometry.GenerateMesh(maxh=hglobal))
        self.mesh.Curve(2)
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

        # Space of viscous stresses c = (c_rr, c_rz, c_zz, c_θθ)
        L = ng.L2(self.mesh, order=p)
        #                    c_rr  c_rz  c_zz  c_θθ
        self.S = ng.FESpace([L,     L,    L,    L])

        self.SU = ng.FESpace([self.S, self.U])

        print('  Setting up static solve system')
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
        return 2 * self.mu * s + self.lam * t * CF((1, 0, 1, 1))

    def CeAv(self, s):
        """ Return Ce(Av(s)), in simplified form: Ce(Av(s)) = (dev s) / τ.
        """
        return dev(s) / self.tau
