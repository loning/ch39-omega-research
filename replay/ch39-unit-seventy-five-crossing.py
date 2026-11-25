"""一份七十五维重物质在同一十九点连接及三辐边修正下的平滑三线交点。"""
from collections import defaultdict
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from runpy import run_path

from flint import acb, arb, ctx, fmpz_mat

smooth = run_path(str(Path(__file__).with_name("ch39-smooth-two-stiffness-crossing.py")))
matter, soft = smooth["matter"], smooth["soft"]
beta = smooth["beta"]


def index_table():
    # End(wedge^2 5) = 1+24+75, End(5) = 1+24.
    rows = defaultdict(lambda: [Fraction(0) for _ in range(3)])
    for degree, sign in ((2, 1), (1, -1)):
        weights = list(combinations(range(5), degree))
        for left in weights:
            for right in weights:
                v = [int(i in left)-int(i in right) for i in range(5)]
                q = -2*sum(v[:3])+3*sum(v[3:])
                terms = (Fraction(q*q, 60), Fraction((v[3]-v[4])**2, 4),
                         Fraction((v[0]-v[1])**2, 4))
                for j,t in enumerate(terms):
                    rows[abs(q)][j] += sign*t
    assert dict(rows) == {0:[0,16,12], 5:[15,9,12], 10:[10,0,1]}
    return {q:[int(t) for t in row] for q,row in rows.items()}


def shell_spectra():
    height = acb.zeta_zero(11431).imag
    base, neutral, heavy = soft["vector_pencil"](height)
    matrix = base+neutral
    integer = fmpz_mat([[int(t.real.unique_fmpz()) for t in row] for row in matrix.tolist()])
    neutral_mass = [z.real for z,k in integer.charpoly().complex_roots() if z!=0 for _ in range(k)]
    # All three radial edges have unit transport for every charge. The norm
    # of the rank-three stiffness removal equals four times its per-edge weight.
    amount = arb(23)/25*arb(2)**-20/(height**2+soft["point"]["study"]["offset"])
    shift = arb(0).union(4*amount)
    values = {0:neutral_mass, 5:smooth["masses"](base+heavy),
              10:smooth["masses"](base+matter["nonspokes"](10,height))}
    result = {q:[v-shift for v in masses] for q,masses in values.items()}
    assert all(v>arb(".01") for row in result.values() for v in row)
    return result


def verify():
    ctx.prec = 224
    indices = index_table()
    mass = shell_spectra()
    geo = soft["geo"]
    radius = geo["theta"](804)/geo["theta"](12)
    mz, em, weak = arb("91.1876"), arb("127.930"), arb(".23129")
    initial = [3*(1-weak)*em/5, em*weak, 1/arb(".1180")]
    pi = arb.pi()
    def intercept(x, factor=1):
        q2 = (radius*factor)**2
        vectors = [sum(smooth["sheet"](v,q2,True) for v in mass[q]) for q in (0,5)]
        def dirac(q, m2):
            levels = mass[q] if q else [arb(0), *mass[0]]
            return sum(smooth["sheet"](v+m2,q2,False) for v in levels)
        exposures = {q:dirac(q,x.upper()).union(dirac(q,x.lower())) for q in indices}
        return [a+7*(n*vectors[0]+b*vectors[1])/(4*pi)
                -2*sum(indices[q][i]*v for q,v in exposures.items())/(3*pi)
                for i,(a,n,b) in enumerate(zip(initial,(0,2,3),(5,3,2)))]
    def differences(values):
        return values[0]-values[2], values[1]-values[2]
    def projection(values):
        first, second = differences(values)
        return (beta[1]-beta[2])*first-(beta[0]-beta[2])*second

    # Locate the mass from the component independent of the light running.
    lo, hi = arb(".01"), arb(".1")
    assert projection(intercept(lo))*projection(intercept(hi)) < 0
    positive_left = projection(intercept(lo)) > 0
    for _ in range(36):
        mid = (lo+hi)/2
        sign = projection(intercept(mid))
        assert sign > 0 or sign < 0
        if (sign > 0) == positive_left:
            lo = mid
        else:
            hi = mid
    box = lo.union(hi)+arb(0,arb("1e-9"))
    L = 2*pi*differences(intercept(lo.union(hi)))[0]/(beta[0]-beta[2])
    logs = L+arb(0,arb("1e-5"))
    assert box > arb(".01")
    reference = radius*(-logs).exp()
    error = 19*reference**2/(8*arb(".01"))*(35/(4*pi)+50/(3*pi))
    def readings(x, scale, factor=1):
        return [a-b*(scale+arb(factor).log())/(2*pi)+arb(0,error.upper())
                for a,b in zip(intercept(x,factor),beta)]
    assert projection(readings(box.lower(),logs.mid()))*projection(readings(box.upper(),logs.mid())) < 0
    assert differences(readings(box,logs.lower()))[0] > 0
    assert differences(readings(box,logs.upper()))[0] < 0
    common = readings(box,logs)[2]
    assert arb("1.3680e16") < mz*logs.exp() < arb("1.3682e16")
    assert arb(".02251147") < box < arb(".02251149")
    assert arb("17.2031") < common < arb("17.2033")
    print("One Dirac 75, common shell, smooth matching scale GeV, mass squared, inverse",
          (mz*logs.exp()).str(13),box.str(12),common.str(12))
    later = readings(box, logs, 2)
    assert all(v < -arb("6.85") for v in later)
    print("At twice the matching momentum", *[v.str(10) for v in later])


if __name__ == "__main__":
    verify()
