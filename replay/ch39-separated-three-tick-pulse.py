"""Joint readout of a split three-slot pulse and a disk bound for all matched two-slot inputs."""
from pathlib import Path
from runpy import run_path

from flint import arb

local = run_path(str(Path(__file__).with_name("ch39-local-product-contact.py")))
contact = local["contact"]
witnesses = (11016,11959,12087,11143,11542,11925,11270,11397,10998,11797,11525,11541)


def coefficients(pair):
    a,b = pair
    cross = a.conjugate()*b
    return abs(a)**2, 2*cross.real, -2*cross.imag, abs(b)**2


def verify():
    rows = []
    first_two = {}
    for n, matrix, weight, obs in contact["candidate_sources"]():
        data = [v for v,_ in contact["pulses"]["port_outputs"](matrix,3,impulse=True)]
        first_two[n] = coefficients(data[:2])
        rows.append((n,abs(data[0]+data[2]/2)**2,weight,obs))
    total = sum(row[2] for row in rows)
    rows = [(n,p,w/total,obs) for n,p,w,obs in rows]
    assert max(w.upper() for n,p,w,obs in rows)<arb(".001")
    reference = first_two[11431]
    differences = [[x-y for x,y in zip(first_two[n],reference)] for n in witnesses]
    half=arb(1)/2
    cells = [(-half,half,-half,half)]
    while cells:
        xlo,xhi,ylo,yhi = cells.pop()
        x,y = xlo.union(xhi),ylo.union(yhi)
        radius = x**2+y**2
        if radius>half**2:
            continue
        if any(a+b*x+c*y+d*radius>arb(".0029") for a,b,c,d in differences):
            continue
        assert xhi-xlo > arb("1e-6")
        xm,ym = ((xlo+xhi)/2).mid(),((ylo+yhi)/2).mid()
        cells.extend((l,r,b,t) for l,r in ((xlo,xm),(xm,xhi)) for b,t in ((ylo,ym),(ym,yhi)))
    assert 9*arb(".001")/(625*arb(".0029")) < arb(".005")
    n,_,_,_,probability,drift,vals,residual = contact["pulse_measurement"](rows,local["graph_spectrum"](),3)
    radii = [arb(".000055")]*3+[arb(".006")]
    assert n==11431 and probability>arb(".999847") and residual<arb(2)**-21
    assert all(abs(v-c)<r for v,c,r in zip(vals,contact["flavor"]["shell"]["centers"],radii))
    print("three-slot (1,0,1) label probability",probability.str(10),"gauge residual <",residual.upper().str(9))
    print("flavor",*(v.str(10) for v in vals),"two-slot label probability below 0.005")


if __name__=="__main__":
    verify()
