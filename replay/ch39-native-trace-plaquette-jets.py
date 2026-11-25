"""原生群上相同有限阶圆弧展开与不同的两面精确积分。"""
from collections import defaultdict
from fractions import Fraction
from math import prod

F = Fraction
# Entries are (dimension, trace(T^2)/dimension) for one normalized generator.
weak = {"one":(1,F(0)), "two":(2,F(1,4)), "three":(3,F(2,3))}
colour = {"one":(1,F(0)), "three":(3,F(1,6)), "bar":(3,F(1,6)), "eight":(8,F(3,8))}
conjugate = {"one":"one","three":"bar","bar":"three","eight":"eight"}
family = ((1,"two","three"),(-4,"one","bar"),(2,"one","bar"),
          (-3,"two","one"),(6,"one","one"))


def dimension(rep):
    _,w,c = rep
    return weak[w][0]*colour[c][0]


def index(rep):
    q,w,c = rep
    return (q*q,weak[w][1],colour[c][1])


def verify_order(order):
    eta = F(1,8)
    charges = [6*j for j in range(1,order+3)]
    base = [eta/(len(charges)*q*q) for q in charges]
    change = [F(1,prod(q*q-p*p for p in charges if p!=q)) for q in charges]
    step = min(a/abs(v) for a,v in zip(base,change))/2
    assert all(sum(v*q**(2*k) for q,v in zip(charges,change))==0 for k in range(order+1))
    outputs = []
    for t in (-step,F(0),step):
        coeff = defaultdict(Fraction)
        for q,a,v in zip(charges,base,change):
            coeff[q,"one","one"] += (a+t*v)/2
            coeff[-q,"one","one"] += (a+t*v)/2
        coeff[0,"three","one"] += eta/40
        coeff[0,"one","eight"] += 2*eta/45
        for rep in family:
            q,w,c = rep
            weight = eta*dimension(rep)/30
            coeff[rep] += weight
            coeff[-q,w,conjugate[c]] += weight
        coeff[0,"one","one"] = 1-sum(coeff.values())
        assert all(a>0 for a in coeff.values()) and coeff[0,"one","one"] > F(1,2)
        bare = [sum(a*index(rep)[j] for rep,a in coeff.items()) for j in range(3)]
        assert bare == [3*eta*v/40 for v in (120,2,2)]
        # Normalized character convolution: chi/d * chi/d = (chi/d)/d².
        squared = {rep:a*a/dimension(rep)**2 for rep,a in coeff.items()}
        moments = [sum(a*index(rep)[j] for rep,a in squared.items()) for j in range(3)]
        outputs.append(moments)
    minus,mid,plus = outputs
    assert minus[1:] == mid[1:] == plus[1:]
    difference = minus[0]+plus[0]-2*mid[0]
    assert difference == step**2*sum(q*q*v*v for q,v in zip(charges,change))
    assert difference > 0
    # In a four-dimensional cubical lattice six plaquettes meet a link.
    # Vary one of their staples, fixing the other five. The leading Haar
    # Hessian after link integration is 5 epsilon² times these same moments.
    # The finite polynomial tail is bounded without sampling the group.
    epsilon = F(1,10**10)
    EY, EW = [26*epsilon*k/5 for k in bare[:2]]
    intervals = [((y-EY)/(w+EW),(y+EY)/(w-EW)) for y,w,_ in outputs]
    assert all(w>EW for _,w,_ in outputs)
    if order in (1,2,4,8,16):
        assert any(a[0]>b[1] for a in intervals for b in intervals)
        print("四维单边积分粗化后的刚度比包络",
              [(round(float(lo),7),round(float(hi),7)) for lo,hi in intervals])
    return [row[0]/row[1] for row in outputs]


def verify():
    for m in (1,2,4,8,16):
        ratios = verify_order(m)
        print("共同 Taylor 阶数、三个两面超荷/弱刚度比、二阶差",2*m+1,
              [float(r) for r in ratios],float(ratios[0]+ratios[2]-2*ratios[1]))
    # For an unknown positive marginal on at most s frequencies, moments
    # 0 through 2s-1 suffice. Alternating supports show the necessity of
    # the last moment for reconstructing the marginal.
    for s in (2,3,8):
        xs = [(6*j)**2 for j in range(2,2+2*s)]
        v = [F(1,prod(x-y for y in xs if y!=x)) for x in xs]
        assert sum(w>0 for w in v)==s and sum(w<0 for w in v)==s
        assert all(sum(w*x**k for w,x in zip(v,xs))==0 for k in range(2*s-1))
        assert sum(w*x**(2*s-1) for w,x in zip(v,xs))==1


if __name__ == "__main__":
    verify()
