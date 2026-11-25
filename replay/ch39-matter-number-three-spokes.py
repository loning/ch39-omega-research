"""零裸质量的 75 维物质在三条辐边上携带共同粒子数相位。"""
from fractions import Fraction
from functools import cache
from pathlib import Path
from runpy import run_path

from flint import acb, acb_mat, arb, arb_poly, ctx, fmpq, fmpq_poly, fmpz_mat

crossing = run_path(str(Path(__file__).with_name("ch39-unit-seventy-five-crossing.py")))
smooth, soft = crossing["smooth"], crossing["soft"]



def verify_vacuum(matrices, pole, orbit, weight):
    for q, matrix in matrices.items():
        def polynomial(phase):
            m = acb_mat(matrix)
            for j in orbit:
                m[pole,pole] += weight-1
                m[j,j] += weight-1
                m[pole,j] = -weight*phase
                m[j,pole] = m[pole,j].conjugate()
            coefficients = (-m).charpoly().coeffs()
            assert all(z.imag.contains(0) for z in coefficients)
            return arb_poly([z.real for z in coefficients])
        plus, minus, side = (polynomial(z) for z in (1, -1, acb(0,1)))
        constant = (plus+minus)*arb(".5")
        cosine = (plus-minus)*arb(".5")
        sine = side-constant
        # A phase-dependent permutation uses an oriented cycle on >=3
        # vertices; s^17, s^18 and s^19 are exactly phase independent.
        for poly in (cosine, sine):
            assert all(poly[k].contains(0) for k in (17,18,19))
        cosine = arb_poly(cosine.coeffs()[:17])
        sine = arb_poly(sine.coeffs()[:17])
        if q == 0:
            assert len(cosine.coeffs()) == 17 and all(t < 0 for t in cosine.coeffs())
        else:
            margin = -constant*cosine-(cosine**2+sine**2)
            assert len(margin.coeffs()) == 36 and all(t > 0 for t in margin.coeffs())
    print("Strict free determinant angle monotonicity, for all positive Euclidean squared momenta")

def verify():
    ctx.prec = 224
    h = acb.zeta_zero(11431).imag
    base, neutral, _ = soft["vector_pencil"](h)
    pole, orbit = soft["pole"], soft["star"]["orbit"]
    assert len(orbit) == 3
    matter = crossing["matter"]
    signed = {q: base+matter["nonspokes"](q,h) for q in (-10,-5,5,10)}
    ordinary = fmpz_mat([[int(v.real.unique_fmpz()) for v in row]
                         for row in (base+neutral).tolist()])
    p_plus = fmpq_poly(ordinary.charpoly())
    for j in orbit:
        assert ordinary[pole,j] == ordinary[j,pole] == -1
        ordinary[pole,j] = ordinary[j,pole] = 1
    p_minus = fmpq_poly(ordinary.charpoly())
    change = arb(23)/25*arb(2)**-20/(h*h+soft["point"]["study"]["offset"])
    verify_vacuum({0:base+neutral, 5:signed[5], 10:signed[10]}, pole, orbit, 1-change)
    weyl = arb(0).union(4*change)
    geo = soft["geo"]
    r2 = (geo["theta"](804)/geo["theta"](12))**2
    vector_mass = crossing["shell_spectra"]((0,5))
    v0, v5 = (sum(smooth["sheet"](v,r2,True) for v in vector_mass[q]) for q in (0,5))
    mz, em, weak = arb("91.1876"), arb("127.930"), arb(".23129")
    initial = (3*(1-weak)*em/5, weak*em, 1/arb(".1180"))
    pi, beta = arb.pi(), smooth["beta"]
    background = [v+7*(n*v0+k*v5)/(4*pi) for v,n,k in zip(initial,(0,2,3),(5,3,2))]
    weights = crossing["index_table"]()

    @cache
    def eigen(cosine):
        c = fmpq(cosine.numerator, cosine.denominator)
        phase = acb(arb(c), (1-arb(c)**2).sqrt())
        # Every determinant monomial meets the pole row and column at most
        # once each. Reality and conjugation leave an affine function of cos.
        polynomial = ((1+c)*p_plus+(1-c)*p_minus)/2
        roots = polynomial.complex_roots()
        assert all(z.imag.contains(0) for z,k in roots)
        result = {0:[z.real-weyl for z,k in roots for _ in range(k)]}
        for q, matrix in signed.items():
            m = acb_mat(matrix)
            for j in orbit:
                m[pole,j], m[j,pole] = -phase, -phase.conjugate()
            result[q] = [v-weyl for v in smooth["masses"](m)]
        assert all(len(row) == 19 for row in result.values())
        return result

    def intercept(cosine, allowance=0):
        def read(row):
            # Weyl errors can be inserted into the mass before taking sheet.
            assert all(v-allowance > arb(".0001") for v in row)
            return sum(smooth["sheet"](v+arb(0,allowance), r2, False) for v in row)
        d = {q: read(row) for q,row in eigen(cosine).items()}
        return [v-sum(row[i]*(d[q]+d[-q]) for q,row in weights.items())/(3*pi)
                for i,v in enumerate(background)]

    def projection(a):
        return (beta[1]-beta[2])*(a[0]-a[2])-(beta[0]-beta[2])*(a[1]-a[2])

    a, b = Fraction(9,10), Fraction(23,25)
    sign = projection(intercept(a))
    assert sign*projection(intercept(b)) < 0
    for _ in range(30):
        mid = (a+b)/2
        side = projection(intercept(mid))
        assert not side.contains(0)
        if side*sign > 0:
            a = mid
        else:
            b = mid
    # Use rational endpoints and a centered operator norm enclosure for all
    # intermediate cosines. The twisted off-diagonal star has norm sqrt(3).
    a -= Fraction(1,10**9)
    b += Fraction(1,10**9)
    center = (a+b)/2
    def ball(v):
        return arb(fmpq(v.numerator,v.denominator))
    motion = arb(3).sqrt()*ball(b-a)/(2*(1-ball(b)**2).sqrt())
    whole = intercept(center, motion.upper())
    L = 2*pi*(whole[0]-whole[2])/(beta[0]-beta[2])
    logs = L+arb(0,arb("1e-5"))
    ref2 = r2*(-2*logs).exp()
    err = 19*ref2/(8*arb(".0001"))*(35/(4*pi)+50/(3*pi))
    error = arb(0,err.upper())
    assert projection([v+error for v in intercept(a)])*projection([v+error for v in intercept(b)]) < 0
    for t,sgn in ((logs.lower(),1),(logs.upper(),-1)):
        assert sgn*(whole[0]-whole[2]-(beta[0]-beta[2])*t/(2*pi)+2*error) > 0
    inverse = whole[2]-beta[2]*logs/(2*pi)+error
    angle = ball(a).union(ball(b)).acos()
    value = mz*logs.exp()
    assert inverse > 17 and arb("1e16") < value < arb("2e16")
    sites = geo["sites"]
    distances = {sum((u-v)**2 for u,v in zip(x,y)) for i,x in enumerate(sites) for y in sites[:i]}
    assert len(distances) == 30
    radius_sq = sum(u*u for u in sites[0])
    assert all(sum(u*u for u in x) == radius_sq for x in sites)
    window_ref2 = r2*(mz/arb("1e13"))**2
    window_err = arb(0,(19*window_ref2/(8*arb(".0001"))*(35/(4*pi)+50/(3*pi))).upper())
    gaps = []
    for d in sorted(distances):
        direct = intercept(1-Fraction(d,2*radius_sq))
        p = projection([v+window_err for v in direct])
        # The positive coefficients of this zero-sum projection total bY-bC.
        gaps.append(abs(p).lower()/(beta[0]-beta[2]))
    assert min(gaps) > arb(".05222")
    at_pi = projection([v+window_err for v in intercept(Fraction(-1))])
    vacuum_gap = abs(at_pi)/(beta[0]-beta[2])
    assert vacuum_gap > arb("2.51")
    print("Spread floor at the free vacuum", vacuum_gap.lower().str(11))
    assert arb(".43546463") < angle < arb(".43546465")
    assert arb("1.3676e16") < value < arb("1.3678e16")
    assert arb("17.1595") < inverse < arb("17.1597")
    print("Three-spoke matter phase (rad), scale GeV, common inverse", angle.str(12),value.str(12),inverse.str(10))
    print("Global spread floor for all native single pair angles", min(gaps).lower().str(11))


if __name__ == "__main__":
    verify()
