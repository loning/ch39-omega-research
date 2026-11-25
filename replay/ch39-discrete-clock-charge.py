"""循环群电荷对时钟算符的次数约束，以及九态环的热权与规范匹配。"""
from fractions import Fraction
from pathlib import Path
from runpy import run_path

from flint import acb, arb, ctx
from sympy import Matrix, Rational, eye

root = Path(__file__).parent
absolute = run_path(str(root/"ch39-fermion-absolute-matching.py"))
chain = run_path(str(root/"ch39-neutral-fermion-phase-register.py"))["addition_chain"]


def verify():
    ctx.prec = 180
    p, repetitions = 23226, 7
    k = p//repetitions
    charges, weights = chain(k)
    assert charges[-1] == 3318 and len(charges)-1 == 18 and 6*max(charges)<p
    total_error_weight = repetitions*sum(weights[1:])
    assert total_error_weight == 23219
    # 七份单态跳，分别在两个极点连续传三步和四步，两条跨点边闭环。
    states = [('A',j*k) for j in range(4)]+[('B',j*k%p) for j in range(3,8)]
    assert len(set(states)) == len(states) == 9
    hops = []
    for i, (site,q) in enumerate(states):
        other,charge = states[(i+1)%len(states)]
        if site==other:
            assert (charge-q)%p == k
            hops.append(Rational(1,2))
        else:
            assert charge==q
            hops.append(Rational(1,5000))
    mass = eye(9)
    for i,r in enumerate(hops):
        mass[(i+1)%9,i] = r
    # 由 det(tI+H) 的常数、一次和二次项读出两个逆质量矩。
    coefficients = (-(mass.T*mass)).charpoly().all_coeffs()
    rational = lambda x: arb(int(x.p))/int(x.q)
    prod = arb(1)/(2**7*5000**2)
    assert rational(coefficients[-1]).overlaps((1+prod)**2)
    def moment_bounds(cosine):
        c0 = 1+prod**2+2*prod*cosine
        first = rational(coefficients[-2])/c0
        return first,first**2-2*rational(coefficients[-3])/c0

    target = abs(p*absolute["angle"](11431)-2*arb.pi()/3)
    second = abs(p*absolute["angle"](1144)-14*arb.pi()/3)
    margin = arb("2e-6")
    # 所有正谱的初等对称均值夹在几何与算术均值的相应幂之间。
    average = 1+(arb(7)/4+arb(2)/5000**2)/9
    geometric_lower = ((1-prod)**2)**(arb(1)/9)
    normalization = 4*2*prod/(8*arb.pi()**2*56)
    slope_lower = normalization/average**7
    slope_upper = normalization/geometric_lower**7
    gap = slope_lower*((target+margin).cos()-(second-margin).cos())
    bias = slope_upper*2*((target+margin)/2).sin()**2
    thermal_scale, stiffness = arb("1e24"),arb("1e25")
    tail = ((len(charges)-1)*arb.pi()*stiffness.sqrt()*
            (arb(1)/2-2*stiffness*(margin/(2*total_error_weight)).sin()**2).exp())
    reference = (1+arb.const_euler()/2-(4*arb.pi()).log()/2)*(arb(1)/4+acb.zeta_zero(11431).imag**2)
    error = 2*reference*((-thermal_scale*gap).exp()+tail*(thermal_scale*bias).exp())/(1-tail)
    assert error < arb("1e-60")

    time = arb("26.81676714125598")
    gauge = absolute["gauge"]
    model = gauge["shortest"]["model"]
    original = gauge["gauge_state"](gauge["word"],time,(18,6,3))
    mixing = [a+b for a,b in gauge["events"]["mixing"]["displacement"](27)]
    absolute["ground"]["flow_bounds"]()
    beta = [arb(l).union(arb(h)) for l,h in (("-.055","-.054"),(".036",".038"),(".089",".09"))]
    mass_ratio = 100
    def one_sum(c, q):
        first, second_moment = moment_bounds(c)
        u = (q/mass_ratio)**2
        return (u*first/5-3*u**2*second_moment/140).union(u*first/5)
    allowed_cos = (target+margin).cos().union(arb(1))
    raw = one_sum(allowed_cos, arb(1))/(12*arb.pi()**2)
    offset = raw.mid()/(2*model["target"](11431,time))
    changes = [one_sum(c,offset.exp())-one_sum(c,(-time).exp()) for c in (allowed_cos,arb(1))]
    shifts = [(b*changes[0]+s*changes[1])/(16*arb.pi()**2) for b,s in
              ((arb(4)/5,arb(8)/15),(arb(4)/3,0),(0,arb(4)/3))]
    outputs = [1/(v*x)+s*offset+arb(0,m.upper())-y/v-model["target"](11431,time+offset)
               for v,x,s,m,y in zip(gauge["shortest"]["ratios"],original,beta,mixing,shifts)]
    worst = max(abs(x).upper() for x in outputs)
    assert worst < arb(2)**-21
    assert mass_ratio/(5000*arb("5e-10").sqrt()*1000*arb(".999"))<arb(".896")
    assert Fraction(-16,3)+4*Fraction(4,3)==0
    assert Fraction(-16,3)+5*Fraction(4,3)==Fraction(4,3)
    print("时钟标量数",len(charges)-1,"净角势斜率界",slope_lower.str(9),slope_upper.str(9),
          "错配上界",error.str(10),"尺度移量",offset.str(11),"规范残差上界",worst.str(11))


if __name__ == "__main__":
    verify()
