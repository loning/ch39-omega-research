"""两种原生钟弦标定三代边界的质量单位，再从两圈电弱交点反求强耦合。"""
from pathlib import Path
from runpy import run_path

from flint import acb, arb, arb_mat, ctx

ROOT = Path(__file__).parent
two = run_path(str(ROOT / "ch39-chiral-center-two-loop.py"))
sm, chiral = two["sm"], two["chiral"]


def solve(model, initial, layers, full, denominator, seed_L, seed_alpha):
    unit = arb("2.435e18")/denominator**2
    start = (unit/model["mz"]).log()
    table = [(start+l,w,B,f,s) for l,w,B,f,s in two["thresholds"](model,layers,full)]
    L = arb(seed_L)+arb(0,".014")
    inverse = 1/arb(seed_alpha)+arb(0,".04")
    pi = arb.pi()
    assert all(0<l for l,*_ in table)
    even = [*layers[0],*(h for q in (2,4,6) for h in layers[q][3:])]
    assert all(32*h*unit>model["mz"]*L.exp() for h in even)

    def image(L,inverse,steps):
        # 以确定开启的阈值建立仿射主项，可能在盒内穿越的阈值保留正部。
        definite = [r for r in table if r[0]<L]
        possible = [r for r in table if not (r[0]>L)]
        b,B = sm["BETA"],sm["B"]
        segments = []
        last = arb(0)
        for i,(l,w,extra,_,_) in enumerate(possible):
            end = l.min(L)
            dt = (end-last).max(0)
            if dt.upper()>0:
                segments.append((dt,b,B,steps if i==0 else max(48,steps//128)))
            last = end
            b = [x+y for x,y in zip(b,w)]
            B = [[x+y for x,y in zip(r,s)] for r,s in zip(B,extra)]
        dt = (L-last).max(0)
        if dt.upper()>0:
            segments.append((dt,b,B,max(48,steps//128)))
        corr,_ = two["flow"]([*initial[:2],inverse],segments,sm["D"],sm["C"])
        rhs, mat = [],[]
        for i in range(3):
            b0 = sm["BETA"][i]+sum(r[1][i] for r in definite)
            a = initial[i] if i!=2 else 0
            val = 2*pi*(a+corr[i])+sum(r[1][i]*r[0] for r in definite)
            val -= sum(w[i]*(L-l).max(0) for l,w,_,_,_ in possible[len(definite):])
            rhs.append([val])
            mat.append([b0,2*pi,-2*pi*int(i==2)])
        return arb_mat(mat).solve(arb_mat(rhs)).entries()
    result = image(L,inverse,8192)
    assert L.contains_interior(result[0]) and inverse.contains_interior(result[2]),(result,L,inverse)
    for steps in (8192,16384,32768):
        L,inverse = L.intersection(result[0]),inverse.intersection(result[2])
        result = image(L,inverse,steps)
    L,inverse = L.intersection(result[0]),inverse.intersection(result[2])
    alpha = 1/inverse
    mu = model["mz"]*L.exp()
    print("十九标量" if full else "十六标量",denominator,"交点 / GeV 与反推强耦合",
          *[(v.lower().str(10),v.upper().str(10)) for v in (mu,alpha)])
    if not full and denominator==724:
        assert arb(".1183")<alpha<arb(".1185")
    else:
        assert alpha<arb(".117") or alpha>arb(".119")
    return alpha


def verify():
    model,_,_,initial,_ = chiral["source"]["mean"]["inputs"]()
    ctx.prec = 192
    read = chiral["source"]["spectral_supplier"](model)
    rows = read(acb.zeta_zero(11661).imag)
    layers = {q: chiral["ordered"]([v for v,z in rows if z==q]) for q in (0,*model["charges"])}
    assert all(layers[q][2]<9<layers[q][3] for q in model["charges"])
    geometry = chiral["source"]["geometry"]
    clocks = geometry["phases"](acb.zeta_zero(11661).imag)
    assert set(clocks)=={724,804}
    for d in clocks:
        assert sum(k==d for _,_,k,_ in geometry["edges"])==6
    for full,n,L,alpha in ((True,724,"28.987",".1225"),(True,804,"28.703",".11655"),
                           (False,724,"29.040",".11838"),(False,804,"28.763",".11301")):
        solve(model,initial,layers,full,n,L,alpha)


if __name__=="__main__":
    verify()
