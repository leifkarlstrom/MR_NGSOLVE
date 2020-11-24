from axisymviscelas import AxisymViscElas, r, z, rinv, ε
import ngsolve as ng
from ngsolve import CoefficientFunction as CF
import pickle
import os

ng.ngsglobals.msg_level = 1


def compute_manufactured_soln(p=3, refine=0):
    """
    This function implements a manufactured solution as initial and boundary
    data for the magma cavity problem in viscoelastic halfspace. The problem
    is governed by equlibirum
        ∇⋅(Cₑε(u)) = f + ∇⋅(Cₑγ)
    and a strain evolution given by a modified Maxwell viscoelastic material
        C' = CₑAᵥσ.
    where Cₑ is the elastic stiffness tensor, γ the viscous strain tensor,
    σ the stress tensor and ε the strain tensor.

    The chosen manufactured solution is based on the analytic solution to
    an elastic fullspace problem in a cylindrical coordinate system (r, z, θ)
        uₑ = Pa³r / 4μ(r²+z²)^(3/2)
             Pa³z / 4μ(r²+z²)^(3/2)
    where P is the pressure along the cavity wall, a is the cavity radius and
    μ the shear modulus. Exact displacement u and viscous stress C=Cₑγ are
    chosen to be
        u = (2 - e⁻ᵗ)uₑ
        C = (1 - e⁻ᵗ)Cₑε(uₑ).

    This choice of u and C satisfy the equilibrium equations with body force
    f = 0 but do not satisfy the Maxwell material equation. To account for
    this,  a source term G, given by
        G = e⁻ᵗCₑε(uₑ) - CₑAᵥCₑε(uₑ),
    is provided as a source term to the modified maxwell material equation.
    That is, our viscous stress evolution is given by
        C' = CₑAᵥσ + G.

    The output returns the L2 error for displacements and viscous stress
    approximations.
    """
    ave = AxisymViscElas(p=p, refine=refine, hcavity=4,
                         hglobal=4, curvedegree=0, tractionBCparts='cavity',
                         kinematicBCparts='axis|top|rgt|bot',)

    def stress(ε):
        σrr, σrz, σzz, σθθ = ave.Ce(ε)
        return CF((σrr, σrz,
                   σrz, σzz), dims=(2, 2))

    # Time as a parameter whose value can be set later
    t = ng.Parameter(0.0)

    # Exact solution
    P0 = 10
    A = 4
    mu = 0.5

    α = P0 * A * A * A / (4 * mu * (r**2 + z**2)**(3/2))
    uₑ = CF(α * (r, z))

    uexact = CF((2 - ng.exp(-t)) * α * (r, z))
    cexact = (1 - ng.exp(-t))*ave.Ce(ε(uₑ))

    # b.c are imposed on Cₑε(u) = Cₑε(uₑ) + Cₑγ
    tractions = stress(ε(uₑ)) + cexact

    def tmp_source(u):
        """
        Computes the temporal source term G = e⁻ᵗCₑε(u) - CₑAᵥCₑε(u)
        """
        ur, uz = u

        drur = ur.Diff(r)
        dzur = ur.Diff(z)
        druz = uz.Diff(r)
        dzuz = uz.Diff(z)

        # trace of the strain ( tr(ε(u)) )
        tr = drur + dzuz + ur * rinv

        K1 = 2 * ave.mu * (ng.exp(-t) - (1 / ave.tau))
        K2 = (ave.lam * ng.exp(-t) + 1/3) * tr

        return CF((K1 * drur + K2,
                   K1 * (druz+dzur)/2,
                   K1 * dzuz + K2,
                   K1 * rinv * ur + K2))

    G = tmp_source(uₑ)

    # Time-varying boundary condition using the parameter t
    uBC = uexact

    # Initial data
    u0 = ng.GridFunction(ave.U)
    c0 = ng.GridFunction(ave.S)
    u0.components[0].Set(α * r)
    u0.components[1].Set(α * z)
    crr, crz, czz, cθθ = c0.components
    crr.Set(0)
    crz.Set(0)
    czz.Set(0)
    cθθ.Set(0)

    # Time step and solve up to time T
    T = 1e-7
    cu, uht, cht, sht, ts = ave.solve2(tfin=T, nsteps=1, u0=u0, c0=c0, t=t,
                                       tractionBC=tractions, kinematicBC=uBC, G=G)

    # save

    fname = './outputs/ratedata_4(3).pickle'
    print('Saving to ', fname)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open('outputs/ratedata/ratedata_4(3).pickle', 'wb') as f:
        pickle.dump([cu, uht, cht, sht, ave], f)

    c = cu.components[0]  # extract c and u components from output
    u = cu.components[1]

    # Compare with the exact soltuion
    t.Set(T)
    errc = CF(c.components) - cexact
    erru = CF(u.components) - uexact
    ec = ng.sqrt(ng.Integrate(ng.InnerProduct(errc, errc), ave.mesh))
    eu = ng.sqrt(ng.Integrate(ng.InnerProduct(erru, erru), ave.mesh))

    return ec, eu

compute_manufactured_soln(p=4, refine=3)
"""
def rates(func, deg, nref):

    errs = []

    for n in range(nref):
        print('\nREFINEMENT LEVEL ', n)
        ec, eu = func(p=deg, refine=n)

        errs += [[ec, eu]]

    print('\nL² CONVERGENCE RATES:')
    print('=====Case p=%1d' % deg + '='*33)
    print(' h    ||c-ch||   c-rate   ||u-uh||   u-rate ')
    print('-'*46)
    rateu = []
    ratec = []

    for i in range(nref):
        rateu.append('  *  ')
        ratec.append('  *  ')

    for i in range(1, nref):
        e = errs[i]
        eprior = errs[i-1]

        if abs(e[0]) > 1.e-11:
            ratec[i] = format(ng.log(eprior[0]/e[0]) / ng.log(2), '+5.2f')
        else:
            ratec[i] = '  *  '

        if abs(e[1]) > 1.e-11:
            rateu[i] = format(ng.log(eprior[1]/e[1]) / ng.log(2), '+5.2f')
        else:
            rateu[i] = '  *  '
    for i in range(nref):
        print(' h/%-3d %8.2e   %s   %8.2e   %s  ' %
              (2**(i+1),
               errs[i][0], ratec[i],
               errs[i][1], rateu[i]))


rates(compute_manufactured_soln, 3, 4)
"""
