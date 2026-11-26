"""共同引力截止下，三枚规范零点至少有两个隐单态质量须向相反方向改变。"""
from itertools import pairwise
from pathlib import Path
from runpy import run_path

from flint import acb, arb, ctx


fixed = run_path(str(Path(__file__).with_name("ch39-chiral-fixed-cutoff-singlets.py")))
heat = fixed["E"]
cutoff_floor = arb(18) / 100


def bound_cutoff_and_slope(rows):
    # 隐谱逐项 E₂ ≤ 2E₃。两项目标要求 12π/x ≤ 6ΣαE₂ − 2ΣgE₃。
    positive_bound = 6 * sum(a for _, a, _ in rows if a > 0)
    positive_bound += sum(-g for g, _, _ in rows if g < 0)
    first_floor = 12 * arb.pi() / positive_bound
    maximum = max(z.upper() for _, _, z in rows) / first_floor
    upper = sum(6 * a * (1 if a > 0 else heat(2, maximum)) for _, a, _ in rows)
    upper += sum(-2 * g * (arb(1) / 2 if g < 0 else heat(3, maximum)) for g, _, _ in rows)
    assert 12 * arb.pi() / upper > cutoff_floor
    # C=ΛG 与 K=G⁻¹/M_P² 的斜率组合消去全部隐单态。
    maximum = max(z.upper() for _, _, z in rows) / cutoff_floor
    weights = [6 * a - g for g, a, _ in rows]
    response = sum(b * ((-maximum).exp() if b > 0 else 1) for b in weights)
    assert response > 206
    slope = cutoff_floor**2 * response / (2 * arb.pi())
    assert slope > arb("1.06")
    assert slope / (1 + 3 * cutoff_floor + 2 * arb("1e-122")) > arb(".68")


def compare(first, second):
    # 按同一荷与谱序配对。下列差值均按第二枚减第一枚。
    maximum = max(z.upper() for _, _, z in first + second)
    signed_mass = newton_mass = log_mass = arb(0)
    signed_variation = newton_variation = response_variation = arb(0)
    for (g, a, z), (h, b, w) in zip(first, second, strict=True):
        assert g == h and a.contains(b) and b.contains(a)
        if z.is_zero() and w.is_zero():
            continue
        assert z > 0 and w > 0
        signed_mass += g * (w - z)
        signed_variation += abs(g * (w - z))
        newton_mass += a * (w - z)
        newton_variation += abs(a * (w - z))
        response_variation += abs((6 * a - g) * (w - z))
        log_mass += a * (w * (w.log() - 1) - z * (z.log() - 1))

    # 下界覆盖 x ≥ 0.18 的整个半轴。第二个主项对 log(x) 的斜率为 −newton_mass。
    vacuum = signed_mass - (1 - heat(2, maximum / cutoff_floor)) * signed_variation
    assert newton_mass < 0
    newton = log_mass + (arb.const_euler() - cutoff_floor.log()) * newton_mass
    newton -= maximum / cutoff_floor * newton_variation
    response = signed_mass - 6 * newton_mass
    response -= (1 - (-maximum / cutoff_floor).exp()) * response_variation
    return vacuum, newton / (2 * arb.pi()), response / (4 * arb.pi())


def second_moment(t):
    """已知 E₃(u)=t，包围 E₂(u)。此处剩余两场的 u 均在 [0,4]。"""
    lo, hi = arb(0), arb(4)
    assert heat(3, hi) < t < heat(3, lo)
    for _ in range(60):
        mid = (lo + hi) / 2
        value = heat(3, mid)
        if value > t:
            lo = mid
        elif value < t:
            hi = mid
        else:
            break
    return heat(2, lo.union(hi))


def two_masses(rows):
    q = arb(3) / 7
    s, t = fixed["required"](rows, q)
    # 309 份在所有背景取同一质量 M=Q/32，余下两份分担两个矩。
    s -= 309 * heat(3, arb(1) / 1024)
    t -= 309 * heat(2, arb(1) / 1024)
    assert arb(1) / 2 < s < 1
    low = 2 * second_moment(s / 2)
    high = 1 + second_moment(s - arb(1) / 2)
    assert low < t < high
    return s, t, low, high


def verify():
    chiral = fixed["chiral"]
    model, *_ = chiral["source"]["mean"]["inputs"]()
    ctx.prec = 192
    read = chiral["source"]["spectral_supplier"](model)
    spectra = []
    for number in (12207, 11661, 11934):
        source = read(acb.zeta_zero(number).imag)
        bands = {q: chiral["ordered"]([h for h, charge in source if charge == q])
                 for q in (0, *model["charges"])}
        rows = fixed["minimal"]["fields"](bands)
        bound_cutoff_and_slope(rows)
        print(number, "剩余真空矩、牛顿矩与牛顿矩两端",
              *(v.str(10) for v in two_masses(rows)))
        spectra.append((number, rows))
    for (i, left), (j, right) in pairwise(spectra):
        vacuum, newton, response = compare(left, right)
        assert vacuum > arb("9.0e-8") and newton > arb("1.47e-8") and response > arb("1.08e-8")
        print(i, j, "xΔS、可见牛顿倒数差、消去单态的斜率组合差下界",
              *(v.lower().str(9) for v in (vacuum, newton, response)))


if __name__ == "__main__":
    verify()
