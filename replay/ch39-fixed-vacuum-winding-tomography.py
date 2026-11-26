"""用固定有限高度的一百个作用差唯一辨识共用同一真空的105个绕数权。"""
from pathlib import Path
from runpy import run_path

from flint import acb, arb, arb_mat, ctx


def verify():
    root = Path(__file__).resolve().parent
    lib = run_path(str(root / "ch39-mincut-global-winding.py"))
    shell = run_path(str(root / "ch39-three-ray-chiral-boundary.py"))
    fixed = run_path(str(root / "ch39-height-independent-flavor-vacuum.py"))
    angles = run_path(str(root / "ch39-mincut-reference-ckm-action.py"))
    flavor = run_path(str(root / "ch39-chiral-neutral-flavor-phase.py"))
    ctx.prec = 320
    tri = lib["terms"](shell)
    x = arb("-.01046")
    y = x-angles["reference_angles"](flavor)[0]
    cap = arb(23)/25*arb(2)**-20/14**2
    delta = arb(0).union(cap)
    orders = range(1,106)
    columns = [[arb(1), *fixed["gradient_columns"](lib, tri, q, x, y)] for q in orders]
    intervals = [lib["coeff"](tri, q, delta) for q in orders]
    included = [i for i in range(1,106) if i not in (1,3,5,6,53)]
    for i in included:
        for q, coeff, col in zip(orders,intervals,columns,strict=True):
            c = coeff[0]+coeff[2]
            phase = acb(0,arb.pi()*((q*i)%210)/105).exp()
            col.append((c*(1-phase)).real)
    matrix = arb_mat(columns).transpose()
    assert matrix.ncols()==matrix.nrows()==105 and len(included)==100
    inverse = matrix.inv()
    # 前五条方程来自归一与跨高度固定驻点，视为已知。每项读数带独立绝对误差。
    row_bounds = [sum(abs(inverse[i,j]) for j in range(5,105)) for i in range(105)]
    bound = max(v.upper() for v in row_bounds)
    print("100个作用差对各绕数权的最大绝对误差放大上界", bound.str(13))
    assert bound < arb("11.584")


if __name__=="__main__":
    verify()
