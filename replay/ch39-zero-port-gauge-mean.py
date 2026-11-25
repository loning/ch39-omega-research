"""规范均值固定时，以一个实对角端口检验全谱非相干混合。"""
from pathlib import Path
from runpy import run_path

from flint import acb, arb, ctx, fmpq, fmpq_poly

root = Path(__file__).parent
single = run_path(str(root/"ch39-zero-single-port.py"))
gauge = run_path(str(root/"ch39-high-zero-planck-window.py"))


def verify():
    ctx.prec = 260
    q, *nums = single["polynomials"]()
    offset = fmpq(gauge["OFFSET"].numerator, gauge["OFFSET"].denominator)
    k = offset-fmpq(1, 4)
    den = fmpq_poly([1, -2*k])
    domain = arb(-1).union(arb(0))/200
    assert acb.zeta_zero(1).imag > 10
    def evaluate(poly, x):
        out = arb(0)
        for coefficient in reversed(poly.coeffs()):
            out = out*x+arb(coefficient)
        return out
    assert evaluate(q, domain) > 0 and evaluate(den, domain) > 0
    heights = [acb.zeta_zero(n).imag for n in (11430, 11431, 11432)]
    xs = [-1/(2*(h*h+arb(1)/4)) for h in heights]
    gl, gy, gr = [1/(h*h+arb(offset)) for h in heights]
    weight = (gy-gr)/(gl-gr)
    assert 0 < weight < 1
    gaps = []
    for num in nums:
        slope = num.derivative()*q-num*q.derivative()
        curvature = den*(slope.derivative()*q-2*slope*q.derivative())-4*k*slope*q
        assert 0 not in evaluate(curvature, domain)
        l, y, r = [evaluate(num,x)/evaluate(q,x) for x in xs]
        gaps.append(abs(r-y+weight*(l-r)))
    best = max(range(19), key=lambda i: float(gaps[i]))
    tied = [j for j in range(19) if nums[j] == nums[best]]
    assert gaps[best] > 0 and all(gaps[i] < gaps[best] for i in range(19) if i not in tied)
    print("Gauge-matched neighbor pair, left weight", weight.str(22),
          "best diagonal ports", tied, "sharp gap", gaps[best].str(22))


if __name__ == "__main__":
    verify()
