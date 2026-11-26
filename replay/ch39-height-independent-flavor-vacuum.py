"""同一组正绕数权在全部正零点高度将同一精确参考味相位选为全局真空。"""
from pathlib import Path
from runpy import run_path

from flint import acb, arb, arb_mat, ctx


def fixed_weights(lib, tri, x, y):
    # 八个有理数固定一次，余下五权仅由不含高度的五个线性方程决定。
    fixed = {1: ".0114566", 3: ".0132862", 28: ".0766797", 33: ".0506797",
             40: ".0212736", 69: ".0891141", 112: ".0228542", 125: ".0018878"}
    fixed = {q: arb(w) for q, w in fixed.items()}
    unknown = [7, 35, 42, 83, 105]
    columns = {}
    for q in [*fixed, *unknown]:
        bare = lib["coeff"](tri, q, arb(0))
        soft = [sum(z**q for z, count in tri[k] if count == 1)
                for k in ((1, 0), (0, 1), (1, -1))]
        columns[q] = [arb(1), *lib["derivatives"]([(q, bare)], x, y)[:2],
                      *lib["derivatives"]([(q, soft)], x, y)[:2]]
    target = [arb(int(k == 0))-sum(w*columns[q][k] for q, w in fixed.items())
              for k in range(5)]
    matrix = arb_mat([columns[q] for q in unknown]).transpose()
    assert not matrix.det().contains(0)
    solution = matrix.solve(arb_mat([[v] for v in target])).entries()
    weights = {**fixed, **dict(zip(unknown, solution, strict=True))}
    assert min(weights.values()) > arb(".0018")
    return weights


def verify():
    root = Path(__file__).resolve().parent
    lib = run_path(str(root / "ch39-mincut-global-winding.py"))
    shell = run_path(str(root / "ch39-three-ray-chiral-boundary.py"))
    global_model = run_path(str(root / "ch39-reference-flavor-global-winding.py"))
    local = run_path(str(root / "ch39-mincut-reference-ckm-action.py"))
    flavor = run_path(str(root / "ch39-chiral-neutral-flavor-phase.py"))
    ctx.prec = 224
    tri = lib["terms"](shell)
    global_model["conjugation_symmetry"](root)
    cap = arb(23)/25*arb(2)**-20/14**2
    assert acb.zeta_zero(1).imag > 14 and shell["soft"]["point"]["study"]["offset"] > 0
    x = arb("-.01046")
    y = x-local["reference_angles"](flavor)[0]
    weights = fixed_weights(lib, tri, x, y)
    delta = arb(0).union(cap)
    modes = [(q, [w*z for z in lib["coeff"](tri, q, delta)]) for q, w in weights.items()]
    drift = global_model["global_cover"](lib, tri, weights, modes, x, y)
    assert drift < arb("-.05")
    global_model["physical_all_heights"](root, x, y, cap)


if __name__ == "__main__":
    verify()
