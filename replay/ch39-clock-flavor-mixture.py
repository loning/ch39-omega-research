"""Mix the flavor observables with the actual label weights of the static clock ground state."""
from itertools import islice
from pathlib import Path
from runpy import run_path

from flint import acb, arb, arb_series

clock = run_path(str(Path(__file__).with_name("ch39-static-pulse-clock.py")))
flavor = run_path(str(Path(__file__).with_name("ch39-fixed-local-flavor.py")))
pulses = clock["pulses"]


def flavor_observables(height, coefficients):
    matrix = flavor["mixing"](flavor["native"]["plane_response"](height), coefficients)
    j = (matrix[0, 0]*matrix[1, 0].conjugate()*matrix[1, 1]*matrix[0, 1].conjugate()).imag
    return [abs(matrix[i, k])**2 for i, k in ((0, 1), (1, 2), (0, 2), (1, 0))]+[j]


def readouts(observables):
    us, cb, ub, cd, j = observables
    s, t, u = (us/(1-ub)).sqrt(), (cb/(1-ub)).sqrt(), ub.sqrt()
    q = (cd-s*s*(1-t*t)-(1-s*s)*t*t*ub)/2
    phase = acb((1-ub)*q, j).arg()
    return [s, t, u, phase]


def phase_radius_squared(observables):
    us, cb, ub, cd, j = observables
    c2 = 1-ub
    s2, t2 = us/c2, cb/c2
    q = (cd-s2*(1-t2)-(1-s2)*t2*ub)/2
    return ((j/c2)**2+q**2)/(s2*(1-s2)*t2*(1-t2)*ub)


def verify():
    data = {}
    coefficients = [arb(n)/2048 for n in flavor["codes"]]
    for n, zero in enumerate(acb.zeta_zeros(10815, 1278), 10815):
        stream = pulses["pulse_records"](pulses["generator"](zero.imag))
        p = next(islice(stream, 6, 7))[0]
        weight = 1/(arb(1)/4+zero.imag**2)
        data[n] = (p, weight, flavor_observables(zero.imag, coefficients))

    p0, w0, center = data.pop(11431)
    total_weight = w0+sum(w for _, w, _ in data.values())
    competitors = [(p0-p, w/w0, [v-c for v,c in zip(obs, center)])
                   for p, w, obs in data.values()]
    gap = min(d.lower() for d, _, _ in competitors)
    def geometry(t):
        e = clock["free_ground"](p0-t)
        a, da = clock["endpoint"](e, 1)
        b, db = clock["endpoint"](e, 3)
        return a-b, db+p0*(da-db), da-db

    def secular(t, inverse_strength):
        span, _, _ = geometry(t)
        return (4*span*(inverse_strength-arb(1)/2)*total_weight/w0*t
                -1+sum(w*t/(d-t) for d,w,_ in competitors))

    def isolate(inverse_strength):
        lo, hi = arb(0), (gap*arb(".99")).mid()
        assert secular(hi,inverse_strength) > 0
        for _ in range(30):
            mid = ((lo+hi)/2).mid()
            sign = secular(mid, inverse_strength)
            if sign > 0:
                hi = mid
            else:
                assert sign < 0
                lo = mid
        return lo.union(hi)

    def average(t):
        _, derivative, difference = geometry(t)
        norm = arb(1)
        changes = [arb(0) for _ in center]
        for d, w, offsets in competitors:
            ratio = w*(1-d*difference/derivative)*(t/(d-t))**2
            norm += ratio
            for k, offset in enumerate(offsets):
                changes[k] += ratio*offset
        return [c+v/norm for c,v in zip(center, changes)]

    endpoints = flavor["shell"]["centers"]
    radius = flavor["shell"]["sine_radius"]
    radii = [radius]*3+[flavor["shell"]["phase_radius"]]
    lower_square = (endpoints[0]-radius)**2
    base_test = center[0]+lower_square*center[2]-lower_square
    test_rows = [(d, w, base_test+v[0]+lower_square*v[2]) for d,w,v in competitors]
    def lower_test(t):
        _, slope, diff = geometry(t)
        return base_test + sum(w*h*(1-d*diff/slope)*(t/(d-t))**2 for d,w,h in test_rows)

    # Test the derivative divided by t, which is regular at the pure-label endpoint.
    def derivative(t):
        var = arb_series([t, 1], prec=2)
        _, slope, diff = geometry(var)
        q, dq = (diff/slope)[0], (diff/slope)[1]
        return sum(w*h*(2*d*(1-d*q)/(d-t)**3 -d*dq*t/(d-t)**2)
                   for d,w,h in test_rows)

    end = isolate(arb(0)).upper()
    regions = [(arb(0), end)]
    while regions:
        lo, hi = regions.pop()
        t = lo.union(hi)
        derived = readouts(average(t))
        if (derivative(t) < 0 and derived[0] < endpoints[0]+radius
                and all(abs(v-c) < r for v,c,r in zip(derived[1:], endpoints[1:], radii[1:]))):
            continue
        assert hi-lo > arb("1e-13")
        mid = ((lo+hi)/2).mid()
        regions.extend(((lo,mid),(mid,hi)))
    lo, hi = arb(0), end
    assert lower_test(lo) > 0 > lower_test(hi)
    for _ in range(22):
        mid = ((lo+hi)/2).mid()
        sign = lower_test(mid)
        if sign > 0:
            lo = mid
        else:
            assert sign < 0
            hi = mid
    t = lo.union(hi)
    span = geometry(t)[0]
    inverse = arb(1)/2 + (1/t - sum(w/(d-t) for d,w,_ in competitors))*w0/(4*span*total_weight)
    critical = 1/inverse
    assert arb(".0838677") < critical < arb(".0838683")
    print("critical coupling", critical.str(10))
    for value in (arb(0), arb(1)/32, arb(1)):
        offset = isolate(1/value) if value > 0 else arb(0)
        observations = average(offset)
        squared = phase_radius_squared(observations)
        if value == arb(1)/32:
            assert arb(".9996255") < squared < arb(".9996256")
        if value == 1:
            assert arb(".9993900") < squared < arb(".9993902")
        print("coupling", value.str(5), "readouts", *(z.str(12) for z in readouts(observations)),
              "squared phase radius", squared.str(10))


if __name__ == "__main__":
    verify()
