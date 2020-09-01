from axisymviscelas import AxisymViscElas, r, z
import ngsolve as ng
from ngsolve import CoefficientFunction as CF
ng.ngsglobals.msg_level = 1


ave = AxisymViscElas(p=2, refine=0, hcavity=2, hglobal=2, curvedegree=0)


threshold = 1.e-15  # Use thresholding to avoid division by zero in 1/r
rinv = 1.0 / ng.IfPos(r - threshold, r, threshold)
def ε(u):
    """
    Compute strain (for a coefficient function) from input displacement
    u = ur * er + uz * ez, where er and ez are the unit vectors in r and z
    directions.
    """
    ur, uz = u

    drur = ur.Diff(r)
    dzur = ur.Diff(z)
    druz = uz.Diff(r)
    dzuz = uz.Diff(z)
    return CF((drur, (druz+dzur)/2, dzuz, ur * rinv))


def test_primal():
    """
    Check that displacement r eᵣ is recovered by primalsolve() method.
    """

    uexact = CF((r, 0))

    u = ave.primalsolve(kinematicBC=uexact)
    ng.Draw(u.components[0])
    Δu = CF((uexact[0] - u.components[0], uexact[1] - u.components[1]))
    normΔu = ng.sqrt(ng.Integrate(Δu*Δu, ave.mesh))
    print('Error in u:', normΔu)
    success = normΔu < 1.e-13
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

    cu = ave.solve2(1, 1, u0, c0)

    cexact = CF((1, 0, 1, 1)) + ave.CeAv((9*z, (r-1)/2, 8*z, 9*z))

    err = cexact - CF(tuple(cu.components[0].components))
    e = ng.Integrate(ng.InnerProduct(err, err), ave.mesh)
    print('Error =', e)
    success = e < 1.e-8  # can't expect better error due to curving
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

    Since div(c) = 0, the numerical u-solution in this case only depends
    on the boundary condition. Hence we expect machine 0 error for u.
    The numerical approximation of c on the other hand will have larger
    errors due to the time stepping not recovering the exponential exactly.
    """

    # Time as a parameter whose value can be set later
    t = ng.Parameter(0.0)

    # Exact solution
    ct = ng.exp((2 * ave.mu / 3 - 1) * t / ave.tau)
    cexact = CF((ct, 0, -2*ct, ct))
    uexact = CF((ct * r, 0))

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
    cu = ave.solve2(tfin=T, nsteps=10, u0=u0, c0=c0, t=t, kinematicBC=uBC)
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
    success = eu < 1e-13 and ec < 1e-2
    assert success, 'Timestepping by solve2(..) did not yield expected error'

def test_manufactured_soln():
    #ng.Mesh(ave.mesh.ngmesh.Refine())
    #ave.mesh.Curve(2)
    # Time as a parameter whose value can be set later
    t = ng.Parameter(0.0)

    # Exact solution
    P0 = 10
    A = 4
    mu = 0.5

    α = P0 * A * A * A / (4 * mu * (r**2 + z**2)**(3/2))
    uₑ = CF(α * (r , z ))

    uexact = CF( (2 - ng.exp(-t)) * α * (r, z))
    cexact = (1 - ng.exp(-t))*ave.Ce(ε(uₑ))

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

        return CF( (K1* drur + K2,
                    K1 * (druz+dzur)/2 ,
                    K1* dzuz + K2,
                    K1* rinv * ur + K2 ))

    #G = ng.exp(-t) * ave.Ce(ε(uₑ)) - ave.CeAv(ave.Ce(ε(uexact))) + ave.CeAv(cexact)
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
    T = 0.001
    cu = ave.solve2(tfin=T, nsteps=10, u0=u0, c0=c0, t=t, kinematicBC=uBC, G=G)
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
    #success = eu < 1e-13 and ec < 1e-2
    #assert success, 'Timestepping by solve2(..) did not yield expected error'

#test_primal()
#test_cupdate()
#test_solve2()
test_manufactured_soln()
