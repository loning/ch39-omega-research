"""真实零点在三代规范窗口内以正下密度逼近指定正曲率。"""
from pathlib import Path
from runpy import run_path

from flint import acb, arb, ctx


root = Path(__file__).parent
vacuum = run_path(str(root / "ch39-chiral-graded-vacuum.py"))
gauge = run_path(str(root / "ch39-chiral-clock-unit-readout.py"))
chiral = gauge["chiral"]


def verify():
    model, _, _, initial, _ = chiral["source"]["mean"]["inputs"]()
    ctx.prec = 224
    height = acb.zeta_zero(11934).imag
    read = chiral["source"]["spectral_supplier"](model)
    geometry = chiral["source"]["geometry"]
    anchor = geometry["phases"](height)
    second = arb("-.026")
    edges = [(i, j) for i, j, d, _ in geometry["edges"] if d in (724, 804)]
    assert len(edges) == 12
    assert max(sum(a == i for a, b in edges) for i, j in edges) <= 2
    assert max(sum(b == j for a, b in edges) for i, j in edges) <= 4
    assert {i for i, j in edges}.isdisjoint(j for i, j in edges)
    assert 4 * arb(23) / 25 * arb(2)**-20 / 10000**2 < arb("1e-12")

    def layers(first):
        clocks = {724: anchor[724] * acb(0, first).exp(), 804: anchor[804] * acb(0, second).exp()}
        rows = read(height, clocks=clocks)
        return {q: chiral["ordered"]([v for v, charge in rows if charge == q])
                for q in (0, *model["charges"])}
    center = arb("-.07926918")
    width = arb("1e-7")
    middle = layers(center)
    thick = {}
    for q, levels in middle.items():
        error = 34 * (arb(8).sqrt() * q * width + arb("2e-12")) + arb("1e-10")
        thick[q] = [h + arb(0, error) for h in levels]
    assert all(thick[q][2] < 9 < thick[q][3] for q in model["charges"])
    mu_gut, _, alpha = gauge["crossing_enclosure"](model, initial, thick, False, 724, "29.04", ".1184")
    assert arb(".117") < alpha < arb(".119")

    planck = arb("1.22089e19")
    target = arb("1e-122")
    coefficient = (9 * arb(3).log() - 12 * arb(2).log()) / 2
    def mass(bands):
        heavy = 32 * arb("2.435e18") / 724**2 * bands[2][15]
        return arb("246.22")**2 / heavy
    mu = mass(thick)
    balanced = vacuum["paired_spectrum"](vacuum["minimal"]["fields"](thick))
    rows = balanced + vacuum["low_sector"](4, (mu / planck)**2)
    assert vacuum["newton"](rows, arb(".03")) < 1 < vacuum["newton"](rows, arb(".033"))
    loss = (mu / planck)**2 / (arb(".03") * coefficient)
    def ratio(first):
        return (mass(layers(first)) / planck)**4 * coefficient / (arb.pi() * target) * (1-arb(0).union(loss))
    left, right = ratio(center - width), ratio(center + width)
    print("相位段端点的曲率目标比值", left.str(14), right.str(14))
    assert left > 1 > right
    narrow = arb("1e-8")
    h = middle[2][15] + arb(0, 34 * (arb(8).sqrt() * 2 * narrow + arb("2e-12")) + arb("1e-10"))
    light = arb("246.22")**2 / (32 * arb("2.435e18") / 724**2 * h)
    ppm = (light / planck)**4 * coefficient / (arb.pi() * target) * (1 - arb(0).union(loss))
    assert abs(ppm - 1) < arb("1e-6")
    print("双钟小方块的曲率比值界、零点下密度", ppm.str(10), ((narrow / arb.pi())**2).str(10))
    print("统一能标、强耦合与四层轻质量 meV", mu_gut.str(8), alpha.str(8), (mu * 10**12).str(9))


if __name__ == "__main__":
    verify()
