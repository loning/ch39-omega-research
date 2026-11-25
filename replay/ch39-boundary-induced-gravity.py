"""三代手征交汇谱的一圈诱导牛顿常数上界与真空超迹矩。"""
from pathlib import Path
from runpy import run_path

from flint import acb, arb, ctx

matched = run_path(str(Path(__file__).with_name("ch39-boundary-baryon-triality.py")))
boundary = matched["boundary"]
model = boundary["model"]


def gravity_bound(vectors, scale):
    # Positive curvature weights: 45 light Weyl, 240 heavy Dirac, four
    # spectator Dirac components, six spectator Weyl and at most 77 real scalars.
    positive = arb(45)/6 + arb(240+4)/3 + arb(6)/6 + arb(77)/6
    ceiling = positive-8  # twelve massless gauge vectors after ghosts
    assert (positive-arb(308)/3).contains(0)
    def slope(x):
        return ceiling-6*sum((-a/x).exp() for a in vectors)
    lo,hi = arb(1)/100,arb(100)
    assert slope(lo)>0>slope(hi)
    for _ in range(38):
        mid = (lo+hi)/2
        if slope(mid)>0:
            lo = mid
        else:
            assert slope(mid)<0
            hi = mid
    x = lo.union(hi)
    # E2 is decreasing on the positive axis, endpoint evaluation avoids
    # integrating with an unnecessarily wide ball.
    terms = [((a/x).upper().expint(2)).lower() for a in vectors]
    upper = (x*(ceiling-6*sum(terms))/(2*arb.pi())).upper()
    fraction = upper*(scale/arb("1.22089e19"))**2
    print("诱导牛顿倒数上包络极大点的截止/交汇尺度、上界/交汇尺度平方、观测份额上界",
          x.sqrt().str(11),upper.str(11),fraction.upper().str(11))
    return fraction


def vacuum_moments(fields):
    # The listed sector excludes the X vector and radial Higgs. At the
    # electroweak-symmetric origin, all ordinary light masses are set to zero.
    assert sum(count for count,_ in fields)==-330
    first,second = [-sum(count*a**p for count,a in fields) for p in (1,2)]
    assert arb("367.67") < first**2/second < arb("367.68")
    # Free bosonic polarizations include the three massive X and one radial mode.
    assert 330*second < first**2
    bosons = 332
    # Add one singlet Weyl and two additional real scalars. Let the
    # bosons all have squared mass b, and solve B b^2 = second + 2 x^2.
    coefficient = 2*bosons-4
    x = (4*first+(16*first**2+4*coefficient*(first**2-bosons*second)).sqrt())/(2*coefficient)
    b = (first+2*x)/bosons
    assert (bosons*b-2*x-first).contains(0)
    assert (bosons*b**2-2*x**2-second).contains(0)
    base_log = sum(count*a**2*a.log() for count,a in fields if not a.is_zero())
    remainder = base_log + bosons*b*b*b.log()-2*x*x*x.log()
    assert -7307 < remainder < -7305
    print("抵消三个超迹的新增 Weyl 与共同玻色质量/交汇尺度，有限对数超迹/交汇尺度四次",
          x.sqrt().str(12), b.sqrt().str(12), remainder.str(12))
    # With one distinguished real scalar, solve both polynomial mass sums
    # first. A sign change in the remaining log moment gives an exact root.
    def two_masses(t):
        mean = (first+2*t)/bosons
        variance = (second+2*t*t)/bosons-mean*mean
        shift = (variance/(bosons-1)).sqrt()
        return mean-shift, mean+(bosons-1)*shift
    def log_moment(t):
        a,c = two_masses(t)
        assert a>0 and c>0
        return base_log+(bosons-1)*a*a*a.log()+c*c*c.log()-2*t*t*t.log()
    lo,hi = arb(57),arb(1000)
    assert log_moment(lo)<0<log_moment(hi)
    for _ in range(34):
        mid = (lo+hi)/2
        if log_moment(mid)<0:
            lo = mid
        else:
            assert log_moment(mid)>0
            hi = mid
    t = lo.union(hi)
    a,c = two_masses(t)
    print("同时消去剩余对数矩时的三种质量/交汇尺度（Weyl、共同玻色、单独标量）",
          t.sqrt().str(12),a.sqrt().str(12),c.sqrt().str(12))
    third = sum(count*z**3 for count,z in fields)+(bosons-1)*a**3+c**3-2*t**3
    print("该解的六次质量超迹/交汇尺度六次",third.str(12))
    return third, [(bosons-1,a),(1,c),(-2,t)]


def finite_cutoff(scale, fields, third, added):
    planck = arb("1.22089e19")
    # High-cutoff signed curvature weight after the additional 328 scalars
    # and one Weyl. At any finite cutoff the loss per component is <= 1-E2(zmax).
    net = (-arb(83)+329)/6
    assert net == 41
    # Original positive weights < 103 and all negatives have magnitude 116.5,
    # while the extra neutral scalars and Weyl add 329/6.
    total_abs_bound = arb(280)
    low, high = arb(".391"), arb(".392")
    ratio = low.union(high)
    all_fields = fields+added
    zmax = max(a.upper() for _,a in all_fields)*(scale/(low*planck))**2
    assert zmax < arb("2e-9")
    loss = total_abs_bound*(1-zmax.upper().expint(2))
    # The root is in this interval, and its positive slope above the lower
    # endpoint follows from the same estimate with exp(-z) instead of E2.
    assert low**2*(net+loss)/(2*arb.pi()) < 1 < high**2*(net-loss)/(2*arb.pi())
    assert net-total_abs_bound*(1-(-zmax).exp()) > 0
    # Subtract the four exact zero moments analytically before evaluating
    # the finite-cutoff remainder. For z<=1 the next term bounds the tail.
    quartic_abs = sum(abs(count)*a**4 for count,a in all_fields)
    prefactor = (scale/planck)**6/(32*arb.pi()**2*ratio**2)
    remainder = quartic_abs*(scale/(low*planck))**2/48
    rho = -prefactor*(third/6+arb(0,remainder.upper()))
    curvature = 8*arb.pi()*rho
    assert curvature < arb("-1e-31") and curvature > arb("-3e-31")
    print("校准牛顿倒数的截止/普朗克质量、有限截止的真空能密度与曲率（普朗克单位）",
          ratio.str(8),rho.str(8),curvature.str(8))


def verify():
    ctx.prec = 224
    scale, mass = matched["verify"]()
    height = acb.zeta_zero(11431).imag
    b = boundary["boundary_sites"]()
    pole = [boundary["soft"]["pole"]]
    radius = boundary["geo"]["theta"](804)/boundary["geo"]["theta"](12)
    u = boundary["arc"]["first_arc"]()
    vectors = [a*u/radius**2 for a in model["shell"]["shell_spectra"]((0,))[0]]
    fraction = gravity_bound(vectors, scale)
    assert fraction < arb("1e-10")
    fields = [(24-90+4,arb(0)),(-28,(mass/scale)**2)]
    fields.extend((36,a) for a in vectors)
    fields.extend((4,a/(u*radius**2)) for a in boundary["high_squares"](boundary["laplacian"](3,height),pole))
    for charge, dimension in zip(model["charges"], (6,3,3,2,1)):
        levels = boundary["high_squares"](boundary["laplacian"](charge,height),b)
        multiplier = 1/u if charge in (1,3) else u
        fields.extend((-4*dimension,a*multiplier/radius**2) for a in levels)
    third, added = vacuum_moments(fields)
    finite_cutoff(scale, fields, third, added)


if __name__ == "__main__":
    verify()
