from magmaxisym import AxisymViscElas, r, z
import ngsolve as ng
from ngsolve import CoefficientFunction as CF
ng.ngsglobals.msg_level = 1


ave = AxisymViscElas(p=2,  tractionBCparts='cavity',
                     kinematicBCparts='axis|top|rgt|bot')


def test_primal():
    """
    Check that displacement r eᵣ is recovered by primalsolve() method.
    """

    uexact = CF((r, 0))
    traction = CF((2*(ave.mu + ave.lam), 0,
                   0,                    2*ave.lam), dims=(2, 2))

    u = ave.primalsolve(kinematicBC=uexact, tractionBC=traction)
    ng.Draw(u.components[0])
    Δu = CF((uexact[0] - u.components[0], uexact[1] - u.components[1]))
    normΔu = ng.sqrt(ng.Integrate(Δu*Δu, ave.mesh))
    print('Error in u:', normΔu)
    success = normΔu < 1.e-12
    assert success, 'Exact solution r unrecovered by the primal method!'


def test_cupdate():
    """ Check that when u = (rz, -r) and c = identity,  one iteration
    of the c update  c = c + dt Ce Av (Ce ε(u) - c) gives the expected result
    with dt = 1.  Note that CeAv(c) = 0 since  dev(c)=0. Also hand calculation
    gives Ce ε(u) = (9*z, (r-1)/2, 8*z, 9*z).  So the result should be
    c + Ce Av Ce ε(u) = I + CeAv((9*z, (r-1)/2, 8*z, 9*z)). This is what's
    checked here.
    """

    c0 = ng.GridFunction(ave.S)
    u0 = ng.GridFunction(ave.U)

    u0.components[0].Set(r * z)
    u0.components[1].Set(-r)
    c0.vec[:] = 0

    crr, crz, czz, cθθ = c0.components
    crr.Set(1)
    crz.Set(0)
    czz.Set(1)
    cθθ.Set(1)

    cu, _, _, _, _ = ave.solve2(1, 1, u0, c0)

    cexact = CF((1, 0, 1, 1)) + ave.CeAv((9*z, (r-1)/2, 8*z, 9*z))

    err = cexact - CF(tuple(cu.components[0].components))
    e = ng.Integrate(ng.InnerProduct(err, err), ave.mesh)
    print('Error =', e)
    success = e < 1.e-4  # can't expect better error due to curving
    assert success, ''


def test_solve2():
    """
    Check that the timestepping recovers the following manufactured solution:

      u = r * exp( (2μ/3-1) t/τ) * eᵣ        (zero z-component)

      c = [1  0  0] * exp( (2μ/3-1) t/τ)
          [0 -2  0]
          [0  0  1]

    This u and c satisfies the system

      c' = Ce Av (Ce ε(u) - c),
      div(Ce ε(u)) = f + div(c),

    with f = 0 and a time-varying kinematic boundary condition.

    Since div(c) = 0 for this example, the numerical u-solution only depends
    on the boundary condition.    When kinematic bc is given, we may
    therefore expect machine 0 error for u.
    The numerical approximation of c on the other hand will have larger
    errors due to the time stepping not recovering the exponential exactly.

    When traction bc is given, the traction source term interacts with values
    of c on the boudary so u-error will depend on c-error, and hence neither
    error can be expected to machine 0.
    """

    # Time as a parameter whose value can be set later
    t = ng.Parameter(0.0)

    # Exact solution
    ct = ng.exp((2 * ave.mu / 3 - 1) * t / ave.tau)
    cexact = CF((ct, 0, -2*ct, ct))
    uexact = CF((ct * r, 0))
    traction = ct * CF((2*(ave.mu+ave.lam) - 1,             0,
                        0,                      2*ave.lam + 2),
                       dims=(2, 2))

    # Time-varying boundary condition using the parameter t
    uBC = uexact

    # Initial data
    u0 = ng.GridFunction(ave.U)
    c0 = ng.GridFunction(ave.S)
    u0.components[0].Set(r)
    u0.components[1].Set(0)
    crr, crz, czz, cθθ = c0.components
    crr.Set(1)
    crz.Set(0)
    czz.Set(-2)
    cθθ.Set(1)

    # Time step and solve up to time T
    T = 0.1
    cu, _, _, _, _ = ave.solve2(tfin=T, nsteps=10, u0=u0, c0=c0, t=t,
                                kinematicBC=uBC, tractionBC=traction)
    c = cu.components[0]  # extract c and u components from output
    u = cu.components[1]

    # Compare with the exact soltuion
    t.Set(T)
    errc = CF(c.components) - cexact
    erru = CF(u.components) - uexact
    ec = ng.sqrt(ng.Integrate(ng.InnerProduct(errc, errc), ave.mesh))
    eu = ng.sqrt(ng.Integrate(ng.InnerProduct(erru, erru), ave.mesh))
    print('Error in c = ', ec)
    print('Error in u = ', eu)
    success = eu < 1e-2 and ec < 1e-2
    assert success, 'Timestepping by solve2(..) did not yield expected error'


test_primal()
test_cupdate()
test_solve2()
