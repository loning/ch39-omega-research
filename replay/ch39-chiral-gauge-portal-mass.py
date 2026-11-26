"""两枚三代规范交点上，电弱规范源产生 Higgs 二次项的双对数系数。"""
from pathlib import Path
from runpy import run_path

from flint import acb, arb, ctx, fmpq_mpoly_ctx

ROOT = Path(__file__).parent
fixed = run_path(str(ROOT / "ch39-chiral-clock-unit-readout.py"))
chiral = fixed["chiral"]


def gauge_trace():
    # 两个同超荷双重态，x=H†H、z=S†S、u=|H†S|²，|总 Pauli 向量|²=(x-z)²+4u。
    ring = fmpq_mpoly_ctx.get(("a","b","x","z","u","v","w"))
    a,b,x,z,u,v,w = ring.gens()
    # a,b 分别为 g2²、gY²。规范质量平方矩阵的三维块为 a(x+z)I/2。
    tr4 = (3*a**2+b**2)*(x+z)**2/4 + a*b*((x-z)**2+4*u)/2
    # 扣除两个各自背景，把 3/2 Tr M_V^4 转为 16π² beta_V。
    mixed = (tr4-(3*a**2+b**2+2*a*b)*(x**2+z**2)/4)*3/2
    lam3 = (3*a**2+b**2-2*a*b)*3/4
    lam4 = 3*a*b
    assert mixed == lam3*x*z+lam4*u
    assert 2*lam3+lam4 == (9*a**2+3*b**2)/2
    # v=M²、w=λ3、b 在此恒等式中代替 λ4。标量 Hessian 的两对简并根。
    trace_s = 2*(v+w*x)**2+2*(v+(w+b)*x)**2
    assert (trace_s-4*v**2-2*x**2*(w**2+(w+b)**2))/2 == 2*v*x*(2*w+b)


def verify():
    gauge_trace()
    model,_,_,initial,_ = chiral["source"]["mean"]["inputs"]()
    ctx.prec = 192
    read = chiral["source"]["spectral_supplier"](model)
    for n in (11661,11934):
        rows = read(acb.zeta_zero(n).imag)
        layers = {q:chiral["ordered"]([h for h,z in rows if z==q]) for q in (0,*model["charges"])}
        assert all(layers[q][2]<9<layers[q][3] for q in model["charges"])
        mu,common,_ = fixed["crossing_enclosure"](model,initial,layers,False,724,"29.040",".1184")
        unit = arb("2.435e18")/724**2
        masses = [unit*h for h in layers[3][3:]]
        active = [m for m in masses if m<mu]
        assert all(m<mu or mu<m for m in masses)
        assert len(active)==14
        g2 = 4*arb.pi()/common
        gy = 3*g2/5
        source = (9*g2**2+3*gy**2)/2
        loop = 16*arb.pi()**2
        curv = 2*source*sum(m*m for m in active)/loop**2
        generated = source*sum(m**2*(mu/m).log()**2 for m in active)/loop**2
        ratio = generated/(arb("125.25")**2/2)
        print(n,"共同逆耦合、门户源、二次项曲率、逐层双对数贡献/GeV²、与电弱参数之比",
              *[(v.lower().str(9),v.upper().str(9)) for v in (common,source,curv,generated,ratio)])
        assert source>arb(".48") and ratio>arb("1e19")


if __name__=="__main__":
    verify()
