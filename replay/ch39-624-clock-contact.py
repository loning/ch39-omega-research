"""Use the native external-plus-624 shell reflector to normalize a single unit contact."""
from itertools import islice
from pathlib import Path
from runpy import run_path

from flint import acb, arb, ctx

root = Path(__file__).parent
mixture = run_path(str(root/"ch39-clock-flavor-mixture.py"))
clock, pulses, flavor = (mixture[k] for k in ("clock","pulses","flavor"))
gauge = run_path(str(root/"ch39-schur-winner-gauge-yukawa.py"))


def clock_window():
    ctx.prec = 192
    run_path(str(root/"ch39-intrinsic-shell-comparisons.py"))["verify"]()
    coeff = [arb(n)/2048 for n in flavor["codes"]]
    offset = gauge["shortest"]["q"](gauge["shortest"]["model"]["fixed"]["OFFSET"])
    rows = []
    for n, z in enumerate(acb.zeta_zeros(10815, 1278),10815):
        height = z.imag
        stream = pulses["pulse_records"](pulses["generator"](height))
        p = next(islice(stream,6,7))[0]
        response = flavor["native"]["plane_response"](height)
        obs = mixture["direct_observables"](flavor["mixing"](response,coeff))
        obs += [1/(height**2+offset)]
        rows.append((n,p,1/(arb(1)/4+height**2),obs))
    total = sum(w for n,p,w,obs in rows)
    return [(n,p,w/total,obs) for n,p,w,obs in rows]


def ground_readouts(rows, positive_modes):
    # Each spectral weight is relative to the 1/625 uniform-mode weight.
    assert 624 in sum(weight for _, weight in positive_modes)
    peak = next(row for row in rows if row[0] == 11431)
    second = max(p.upper() for n,p,w,obs in rows if n != 11431)
    def boundary(energy):
        a,da = clock["endpoint"](energy,1)
        b,db = clock["endpoint"](energy,3)
        return a-b,b,da-db,db

    def trace(t, full=False):
        energy = clock["free_ground"](peak[1]-t)
        span,base,dspan,dbase = boundary(energy)
        higher_blocks = [(weight,*boundary(energy-level)) for level,weight in positive_modes]
        low_trace, high_trace = arb(0), arb(0)
        low_norm, high_norm, peak_norm = arb(0), arb(0), arb(0)
        differences = [arb(0) for _ in peak[3]]
        for n,p,w,obs in rows:
            m = base+span*p
            # 1-2m is exact in this form because p* - t sets m = 1/2.
            den = 2*span*((peak[1]-p)-t) if n != 11431 else -2*span*t
            low_trace += w*m/den
            higher = arb(0)
            for weight,ea,eb,eda,edb in higher_blocks:
                lifted = eb+ea*p
                high_den = 1-2*lifted
                high_trace += weight*w*lifted/high_den
                if full:
                    higher += weight*w*(edb+eda*p)/high_den**2
            if full:
                ground = w*(dbase+dspan*p)/den**2
                low_norm += ground
                high_norm += higher
                if n == 11431:
                    peak_norm = ground+higher
                else:
                    for k,x in enumerate(obs):
                        differences[k] += (ground+higher)*(x-peak[3][k])
        if full:
            norm = low_norm+high_norm
            return (1/(625+high_trace), high_norm/norm, peak_norm/norm,
                    [base+dv/norm for base,dv in zip(peak[3], differences)])
        return (low_trace+high_trace+625)*t

    lo,hi = arb(0), ((peak[1]-second)/2).mid()
    assert trace(hi)>0
    for _ in range(32):
        mid = ((lo+hi)/2).mid()
        sign = trace(mid)
        if sign < 0:
            lo = mid
        else:
            assert sign > 0
            hi = mid
    t = lo.union(hi)
    shift_energy = clock["free_ground"](peak[1]-t)-clock["free_ground"](peak[1])
    effective, excitation, probability, means = trace(t,True)
    drift = means[-1]/peak[3][-1]-1
    shift = drift.log1p()/2
    residual = arb("1.005e-7")+arb(".13")*abs(shift)
    vals = mixture["readouts"](means)
    tolerances = [flavor["shell"]["sine_radius"]]*3+[flavor["shell"]["phase_radius"]]
    assert all(abs(v-c) < r for v,c,r in zip(vals,flavor["shell"]["centers"],tolerances))
    assert residual < arb(2)**-21
    print("joint ground elevation above the uncoupled clock",shift_energy.str(11))
    print("effective coupling",effective.str(11),"excited channel weight",excitation.str(10),
          "11431 probability",probability.str(10))
    print("original four flavor parameters give",*(v.str(11) for v in vals))
    print("mixed gauge relative shift",drift.str(9),"matched clock shift",shift.str(9),
          "three residuals <",residual.upper().str(9))
    return shift_energy, effective, excitation, probability, residual


def verify():
    shift, effective, excitation, probability, residual = ground_readouts(
        clock_window(), [(1,arb(624))])
    assert arb("8.63959e-8") < shift < arb("8.63960e-8")
    assert arb(".00098368575") < effective < arb(".00098368576")
    assert excitation < arb("2.213e-8") and probability > arb(".99994")
    assert residual < arb("1.88e-7")


if __name__ == "__main__":
    verify()
