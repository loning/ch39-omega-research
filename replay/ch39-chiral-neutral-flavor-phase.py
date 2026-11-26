"""用双钟三角形决定中性味标量的真空相位并检验同一组实 Yukawa 系数。"""
from itertools import combinations
from pathlib import Path
from runpy import run_path

from flint import acb, acb_mat, arb, ctx

ROOT = Path(__file__).parent
geom = run_path(str(ROOT / "ch39-arc-relative-determinant.py"))
ckm = run_path(str(ROOT / "ch39-shell-clock-ckm-blocks.py"))


def singlet(n):
    clocks = geom["phases"](acb.zeta_zero(n).imag)
    edges = {(i,j):(d,u*clocks.get(d,1)) for i,j,d,u in geom["edges"]}
    vals = []
    for i,j,k in combinations(range(19),3):
        links = [edges.get(e) for e in ((i,j),(j,k),(i,k))]
        if None in links:
            continue
        lens = sorted(x[0] for x in links)
        if 724 in lens and 804 in lens:
            assert lens[0]>=32
        if lens==[32,724,804]:
            vals.append((links[0][1]*links[1][1]*links[2][1].conjugate()).real)
    assert len(vals)==6
    cosine = sum(vals)/6
    assert -1<cosine<1
    theta = cosine.acos()/3
    return acb(0,theta).exp(),cosine


def real_pair(u,v,z):
    det = (u.conjugate()*v).imag
    assert not det.contains(0)
    return (z.conjugate()*v).imag/det,(u.conjugate()*z).imag/det


def pair(s,i,j):
    return (s,s.conjugate()**2) if (i-j)%3==1 else (s.conjugate(),s**2)


def read(Y):
    H = Y*Y.conjugate().transpose()
    eig = sorted((z.real for z in H.eig(algorithm="rump")),key=float)
    assert 0<eig[0]<eig[1]<eig[2]
    eye = acb_mat([[int(i==j) for j in range(3)] for i in range(3)])
    projs = []
    for i in (1,2):
        P = eye
        for j in range(3):
            if i!=j:
                P = P*(H-eig[j]*eye)/(eig[i]-eig[j])
        projs.append(P)
    p,q = projs
    s13 = q[0,0].real.sqrt()
    c13 = (1-s13**2).sqrt()
    J = (H[0,1]*H[1,2]*H[2,0]).imag/((eig[1]-eig[0])*(eig[2]-eig[0])*(eig[2]-eig[1]))
    return p[0,0].real.sqrt()/c13,q[1,1].real.sqrt()/c13,s13,J


def verify():
    ctx.prec = 192
    masses = list(map(arb,(".0000135",".000268",".01401")))
    targets = ckm["standard_matrix"](*ckm["centers"])
    # 独立右手重定相使两个 Yukawa 的对角元实；上型固定对角。
    target = acb_mat([[targets[i][j]*masses[j]*acb(targets[j][j]).conjugate()/abs(targets[j][j])
                       for j in range(3)] for i in range(3)])
    first,_ = singlet(11661)
    coefficients = {(i,j):real_pair(*pair(first,i,j),target[i,j])
                    for i in range(3) for j in range(3) if i!=j}
    for n in (11661,11934,12207):
        s,c = singlet(n)
        Y = acb_mat(3,3)
        for i in range(3):
            Y[i,i] = target[i,i].real
            for j in range(3):
                if i!=j:
                    a,b = coefficients[i,j]
                    u,v = pair(s,i,j)
                    Y[i,j] = a*u+b*v
        vals = read(Y)
        print(n,"六面余弦与三个混合角正弦、J",*[x.str(11) for x in (c,*vals)])
        if n==11661:
            assert all(a.contains(b.mid()) or a.overlaps(b) for a,b in zip(vals,ckm["centers"][:3]))
        else:
            assert abs(vals[2]-ckm["centers"][2])>arb(".000055")


if __name__=="__main__":
    verify()
