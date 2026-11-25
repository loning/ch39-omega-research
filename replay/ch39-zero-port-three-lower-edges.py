"""第一、第二角与 CP 相位都在窗口下边时，28% 外部谱质量的端口界。"""
from pathlib import Path
from runpy import run_path

from flint import acb, arb, arb_mat

prev = run_path(str(Path(__file__).with_name("ch39-zero-port-phase-branch.py")))
box, study, sharp = (prev[k] for k in ("box", "study", "sharp"))
S, T, h = sharp["S"], prev["T"], prev["half_tan"]
K = (1-S)*(1-T)
r = arb(7)/25
zero_ids = (*prev["ids"], 11415)


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def rows_relative(values):
    return [[v-base for v,base in zip(row,values[1])] for row in values]


def solve_centered(matrix,rhs):
    middle=arb_mat([[matrix[i,j].mid() for j in range(matrix.ncols())]
                    for i in range(matrix.nrows())])
    guess=middle.solve(arb_mat([[rhs[i,j].mid() for j in range(rhs.ncols())]
                                for i in range(rhs.nrows())]))
    return guess+matrix.solve(rhs-matrix*guess)


def candidate(coords,values):
    dif=rows_relative(values)
    # 用 11432 承担尚未分给另四个外部编号的权重，消去总质量方程。
    start=[a+r*(b-a) for a,b in zip(values[0],values[1])]
    a,b,uc,dc,j=start
    domain=sharp["lo"].union(sharp["hi"])
    def solve_at(u):
        cu=K*(1-u-uc)+dc-S*T
        phase=[0,0,-h*cu,h*(1-u),1]
        mat=arb_mat([[coords[k]-coords[1] for k in range(2,6)],
                     [a+S*v for a,b,v,d,j in dif[2:]],
                     [b+T*v for a,b,v,d,j in dif[2:]],
                     [dot(phase,row) for row in dif[2:]]])
        rhs=arb_mat([[-r*coords[1]],[S*(1-uc)-a],[T*(1-uc)-b],
                     [-j+h*(1-uc)*(K*uc-dc+S*T)]])
        ws=solve_centered(mat,rhs)
        exterior=[ws[k,0] for k in range(4)]
        U=uc+dot(exterior,[row[2] for row in dif[2:]])
        D=dc+dot(exterior,[row[3] for row in dif[2:]])
        return mat,exterior,U,D
    mat,exterior,U,D=solve_at(domain)
    assert domain.contains_interior(U), (str(domain),str(U))
    # 求固定点映射 u ↦ U 的导数。只有第四行依赖 u。
    deriv=mat.solve(arb_mat([[0],[0],[0],[h*((D-dc)-K*(U-uc))]]))
    assert abs(dot([deriv[i,0] for i in range(4)],[row[2] for row in dif[2:]]))<1
    for _ in range(5):
        domain=domain.intersection(U)
        _,exterior,U,D=solve_at(domain)
    full=[1-r,r-sum(exterior),*exterior]
    assert all(w>0 for w in full)
    assert dot(full,[row[4] for row in values])>0
    return full,(U,D)


def dual_plane(coords,costs,values,reference,u):
    uc,dc=reference
    phase=[0,0,-h*(K*(1-u-uc)+dc-S*T),h*(1-u),1]
    dif=rows_relative(values)
    mat=arb_mat([[coords[i]-coords[1],a+S*v,b+T*v,dot(phase,dif[i])]
                 for i,(a,b,v,d,j) in enumerate(dif) if i>=2])
    dual=solve_centered(mat,arb_mat([[c-costs[1]] for c in costs[2:]]))
    beta,first,second,cp=[dual[i,0] for i in range(4)]
    assert first>0 and second>0 and cp>0
    coeff=[cp*z+first*x+second*y for z,x,y in zip(phase,(1,0,S,0,0),(0,1,T,0,0))]
    return mat.det(),beta,coeff


def verify():
    cost,strength,scale,xs,costs,inputs,_=prev["setup"]()
    height=acb.zeta_zero(11415).imag
    x=study["center"]/(height*height+study["offset"])-1
    xs.append(x)
    costs.append(cost(x)[0])
    inputs.append(study["native"]["plane_response"](height))
    theta=arb(0).union(1)
    coords=[x+theta*scale*y for x,y in zip(xs,costs)]
    delta=arb(2)**-28
    cube=[c+arb(0,delta) for c in box["centers"]]
    data=[box["tight_data"](t,box["centers"],[delta]*4) for t in inputs]
    rows,ds=zip(*data)
    weights,ref=candidate(coords,rows)
    _,beta,coeff=dual_plane(coords,costs,rows,ref,ref[0])
    signs=[]
    for k in range(4):
        deriv=-sum(w*dot(coeff,der[k]) for w,der in zip(weights,ds))
        assert 0 not in deriv
        signs.append(-1 if deriv>0 else 1)
    print("Minimizing flavor endpoint signs",signs,flush=True)
    corner=[c+sign*delta for c,sign in zip(box["centers"],signs)]
    best=[box["flavor_data"](t,corner)[0] for t in inputs]
    for t in (arb(0),arb(1)/2,arb(1)):
        ws,_=candidate([x+t*scale*y for x,y in zip(xs,costs)],best)
        optimum=dot(ws,costs)
        print("Direction",t.str(3),"gauge tolerance",(t*scale*optimum).str(19),
              "port tolerance",((1-t)*strength*optimum).str(19),
              "weights",*[w.str(14) for w in ws],flush=True)
    planes=[]
    determinant_signs=[]
    for u in (sharp["lo"],sharp["hi"]):
        det,beta,coeff=dual_plane(coords,costs,rows,ref,u)
        assert 0 not in det
        determinant_signs.append(det>0)
        assert 1-theta*scale*beta>0 and 1+(1-theta)*scale*beta>0
        planes.append([beta,*coeff])
    assert determinant_signs[0]==determinant_signs[1]
    beta,*coeff=[a.union(b) for a,b in zip(*planes)]
    def slack(x,c,obs):
        diff=[a-b for a,b in zip(obs,rows[1])]
        return c-costs[1]-beta*(x+theta*scale*c-coords[1])-dot(coeff,diff)
    assert slack(0,0,rows[0])<0
    for lower in (S,T):
        assert -sharp["hi"]+(1+sharp["lo"])*lower>0
    whole=arb(0).union(1)
    outer=[whole,arb(0).union(arb(104)/867),whole,whole,arb(-1).union(1)]
    for n in (8499,16001):
        height=acb.zeta_zero(n).imag
        x=study["center"]/(height**2+study["offset"])-1
        c,dx=cost(x)
        speed=dx/(1+theta*scale*dx)
        assert (speed>beta if n==8499 else speed<beta)
        assert slack(x,c,outer)>0
    floor=arb(1)
    for start in range(8500,16001,256):
        count=min(256,16001-start)
        zeros=acb.zeta_zeros(start,count)
        assert len(zeros)==count
        for n,zero in enumerate(zeros,start):
            if n in zero_ids:
                continue
            x=study["center"]/(zero.imag**2+study["offset"])-1
            c=cost(x)[0]
            t=study["native"]["plane_response"](zero.imag)
            obs=box["observations"](box["flavor"]["mixing"](t,cube))
            gap=slack(x,c,obs)
            if not gap>0:
                gap=slack(x,c,box["tight_data"](t,box["centers"],[delta]*4)[0])
            assert gap>0,(n,gap.str(12))
            floor=min(floor,gap.lower())
    print("Strict finite slack",floor.str(13))


if __name__ == "__main__":
    verify()
