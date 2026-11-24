"""An empirical-input matched shell ground with one complex scalar threshold."""
from pathlib import Path
from runpy import run_path

from flint import arb

root = Path(__file__).parent
data = run_path(str(root/"ch39-published-gauge-inputs.py"))
comp, q, ratios = data["comp"], data["q"], data["ratios"]
leads = run_path(str(root/"ch39-unit-lead-gauge-shell.py"))
baseline, loop = comp["running"], comp["loop"]
scale, duration = q("26.817228402993100"), q("2.3597484539949534")
threshold = scale-duration


def scalar_flow():
    # Complex scalar (3,1)_{2/3}; g1 uses the 5/3 normalization.
    y = q("2/3")
    casimir = [3*y*y/5, arb(0), q("4/3")]
    index = [3*casimir[0],arb(0),arb(1)/2]
    change = [[index[i]*(4*casimir[j]+q("2/3")*(0,2,3)[i]*int(i == j))
               for j in range(3)] for i in range(3)]
    assert all((x-y).contains(0) for row,want in zip(change,(("64/75","0","64/15"),
               ("0","0","0"),("8/15","0","11/3"))) for x,y in zip(row,map(q,want)))
    return {**baseline, "BETA":[b+t/3 for b,t in zip(baseline["BETA"],index,strict=True)],
            "B":[[x+d for x,d in zip(row,extra,strict=True)] for row,extra in zip(baseline["B"],change,strict=True)]}


def verify_ground(values, intervals, trial_index, scale, allowance, spread_rate):
    lower = upper = [v*v for v in values]
    exact_target = comp["model"]["target"]
    target = exact_target(11431,scale)
    for index,(start,stop,n,flow) in enumerate(intervals):
        step = (stop-start)/n
        for i in range(n):
            lower, lt = comp["advance"](lower,step,False,flow=flow,need_tube=True)
            upper, ut = comp["advance"](upper,step,True,flow=flow,need_tube=True)
            if start+(i+1)*step >= 23:
                # Upper X,Y corresponds to lower squared gauge coordinates.
                g = [a.union(b) for a,b in zip(ut[:3],lt[:3],strict=True)]
                tr = [sum(lt[k:k+3]).union(sum(ut[k:k+3])) for k in (3,6,9)]
                speeds = [-2/(loop*r)*(b+sum(p*z-d*y for p,z,d,y in zip(br,g,dr,tr,strict=True))/loop)
                          for r,b,br,dr in zip(ratios,flow["BETA"],flow["B"],flow["D"],strict=True)]
                assert max(abs(v) for v in speeds) < q(".091")
                assert speeds[2]-speeds[0] > spread_rate
        if index == trial_index:
            residuals = [abs((1/(r*a)).union(1/(r*b))-target) for r,a,b in zip(ratios,lower,upper)]
            assert max(residuals)+q("1.3e-16") < allowance
            print("matching-scale residual bounds", *(v.upper().str(10) for v in residuals))

    # Reuse the arithmetic comparison independent of the running: in
    # [10400,12499] the bare winning gap exceeds 8e-6, and anywhere else
    # the maximum possible bare advantage is less than .000011193.
    radius = q(".061")
    edge = min((exact_target(10399,scale+radius)-target).lower(),
               (target-exact_target(12500,scale-radius)).lower())
    edge -= allowance+q(".091")*radius+q("1.3e-16")
    spread = spread_rate*radius-2*allowance
    assert edge > 0 and spread > 0
    for count in (1,2,3):
        m,d = map(arb,leads["terminal_modes"](count))
        trial = 3*d*allowance**2
        assert min((3*m*edge**2).lower(), (d*spread**2/2).lower())-trial-q(".000011193") > q(".000001")
        assert q("8e-6") > trial
        assert (3*d/m).sqrt()*allowance < arb(2)**-21


def verify():
    extended = scalar_flow()
    values = data["centers"].copy()
    values[2] = q("1.0576412081185345")
    assert abs(values[2]-data["centers"][2]) < q(".22")*data["widths"][2]
    intervals = ((arb(0),threshold,160,baseline),(threshold,scale,32,extended),
                 (scale,arb(33),64,extended))
    verify_ground(values,intervals,1,scale,q("9.02e-8"),q(".143"))
    print("threshold and trial scale in GeV", (1000*threshold.exp()).str(13),(1000*scale.exp()).str(13))
    print("all three lead geometries: unique ground label 11431; at any minimizing scale all residuals below 4.21e-7")


if __name__ == "__main__":
    verify()
