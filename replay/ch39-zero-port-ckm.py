"""在规范均值与四项味窗下求端口 6 的无 11431 混合的最小读数偏差。"""
from pathlib import Path
from runpy import run_path

from flint import acb, arb, arb_mat, arb_poly, ctx, fmpq, fmpq_poly

root = Path(__file__).parent
single = run_path(str(root/"ch39-zero-single-port.py"))
sharp = run_path(str(root/"ch39-sharp-gauge-flavor-variance.py"))


def port_cost():
    q, *nums = single["polynomials"]()
    num = nums[6]
    off = sharp["study"]["gauge"]["OFFSET"]
    offset = fmpq(off.numerator, off.denominator)
    k = offset-fmpq(1, 4)
    den = fmpq_poly([1, -2*k])
    slope = num.derivative()*q-num*q.derivative()
    curvature = den*(slope.derivative()*q-2*slope*q.derivative())-4*k*slope*q
    q, p, slope, curvature = [arb_poly(poly) for poly in (q, num, slope, curvature)]
    domain = arb(-1).union(0)/200
    assert q(domain) > 0 and curvature(domain) < 0 and acb.zeta_zero(1).imag > 10
    scale = acb.zeta_zero(11431).imag**2+arb(offset)
    def data(x):
        g = (1+x)/scale
        s = -g/(2*(1-arb(k)*g))
        factor = 1-2*arb(k)*s
        value = p(s)/q(s)
        first = -factor**2*slope(s)/(2*scale*q(s)**2)
        return value, first, s, factor
    center, tangent, s0, d0 = data(0)
    strength = -d0**3*curvature(s0)/(8*scale**2*q(s0)**3)
    def cost(x):
        value, first, _, _ = data(x)
        return (center+tangent*x-value)/strength, (tangent-first)/strength
    return cost, strength


def verify():
    ctx.prec = 224
    cost, strength = port_cost()
    support = sharp["support"]
    rows = [sharp["row"](acb.zeta_zero(n)) for n in support]
    weights, ref = sharp["attaining_distribution"](rows, [], [])
    targets = [cost(x)[0] for x, obs in rows]
    minimum = sum(w*y for w,y in zip(weights, targets))
    duals = []
    sharp["study"]["torus_bound"]()
    for u in (sharp["lo"], sharp["hi"]):
        tests = sharp["phase_tests"](u, ref)
        matrix = arb_mat([[1, x, *tests(obs)] for x, obs in rows])
        c = matrix.solve(arb_mat([[y] for y in targets]))
        assert matrix.det() > 0 and c[2,0] > 0 and c[3,0] > 0
        assert c[0,0].overlaps(minimum)
        def linear(obs, c=c, tests=tests):
            a, b = tests(obs)
            return c[2,0]*a+c[3,0]*b
        upper = sharp["compare_torus"](c[3,0],linear)
        for n in (8999, 15001):
            x, _ = sharp["row"](acb.zeta_zero(n))
            value, derivative = cost(x)
            assert (derivative > c[1,0] if n == 8999 else derivative < c[1,0])
            assert value-c[1,0]*x > minimum+upper
        duals.append((c[1,0],linear))
    print("Four-point feasible response gap", (minimum*strength).str(23),flush=True)
    slack = arb(1)
    for start in range(9000,15001,256):
        count = min(256,15001-start)
        zeros = acb.zeta_zeros(start,count)
        assert len(zeros) == count
        for n, zero in enumerate(zeros,start):
            if n == 11431 or n in support:
                continue
            x, obs = sharp["row"](zero)
            value, _ = cost(x)
            for slope, linear in duals:
                margin = value-minimum-slope*x-linear(obs)
                assert margin > 0
                slack = min(slack,margin.lower())
    print("Certified minimum normalized strict slack",slack.str(12),flush=True)


if __name__ == "__main__":
    verify()
