"""
Generate the LaTeX tables of docs/MATHEMATICAL_FOUNDATION.tex from
results/results.json, so the PDF can never disagree with the experiments.

    python scripts/make_tex_tables.py      # → results/tables.tex
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

LADDER = ["legacy", "offset", "brown", "g_v20", "g_fgls", "gamma", "v21", "v21_norot"]
TEX_NAMES = {
    "legacy": r"v1 original (reproduced exactly)",
    "offset": r"Pinhole + disparity offset (2 params)",
    "brown": r"Brown--Conrady, 18 params (same loss, FGLS)",
    "g_v20": r"$\Gamma$ grid + rotation, v2.0 loss",
    "g_fgls": r"\quad + estimated noise model (FGLS)",
    "gamma": r"\quad + radial prior ($\Gamma$ grid alone)",
    "v21": r"\quad + Brown--Conrady companion (\textbf{v2.1})",
    "v21_norot": r"v2.1 without rig rotation",
}
SHORT = {"legacy": "v1", "offset": "offset", "brown": "Brown--Conrady", "gamma": r"$\Gamma$ alone",
         "v21": "v2.1", "g_v20": r"$\Gamma$ v2.0 loss", "g_fgls": r"$\Gamma$ + FGLS"}
SYN = [("oracle", r"oracle (true cameras)"), ("offset", "pinhole + offset"), ("legacy", "v1 original"),
       ("pinrot", "pinhole + rotation"), ("brown", "Brown--Conrady"), ("gamma", r"$\Gamma$ grid alone"),
       ("v21", r"\textbf{v2.1 ensemble}")]


def pct(x, d=2):
    return f"{x:.{d}f}\\,\\%"


def main():
    with open(os.path.join(RES, "results.json"), encoding="utf-8") as f:
        r = json.load(f)
    out = []
    rows = r["real_loocv"]
    best = min(rows[k]["mape_pct"] for k in LADDER)
    out += [r"\newcommand{\RealLadderTable}{", r"\begin{tabular}{lccccc}", r"\toprule",
            r"Model & MAPE & 95\,\% CI & median & $\delta{<}1.05$ & worst\\", r"\midrule"]
    for k in LADDER:
        m = rows[k]
        v = pct(m["mape_pct"])
        v = rf"\textbf{{{v}}}" if m["mape_pct"] == best else v
        out.append(f"{TEX_NAMES[k]} & {v} & {m['mape_ci95'][0]:.1f}--{m['mape_ci95'][1]:.1f} & "
                   f"{pct(m['median_ape_pct'])} & {m['delta_1.05_pct']:.0f}\\,\\% & {m['max_ape_pct']:.0f}\\,\\%\\\\")
    out += [r"\bottomrule", r"\end{tabular}}", ""]

    out += [r"\newcommand{\SignificanceTable}{", r"\begin{tabular}{lccc}", r"\toprule",
            r"Comparison & $\Delta$ MAPE (pp) & $p$ & $p_\text{Holm}$\\", r"\midrule"]
    for t in r["significance"]:
        out.append(f"{SHORT[t['a']]} vs {SHORT[t['b']]} & ${t['mean_diff_pp']:+.2f}$ & {t['p']:.4f} & {t['p_holm']:.4f}\\\\")
    out += [r"\bottomrule", r"\end{tabular}}", ""]

    full = r["full_fit"]
    out += [r"\newcommand{\ParamTable}{", r"\begin{tabular}{lccc}", r"\toprule",
            r"Rig parameter & value & s.e.\ (sandwich) & s.e.\ (bootstrap)\\", r"\midrule"]
    for k, p in full["params"].items():
        fmt = "{:.4f}" if p["unit"] == "m" else "{:.3f}"
        unit = "m" if p["unit"] == "m" else r"$^\circ$"
        out.append(f"{k} ({unit}) & {fmt.format(p['value'])} & {fmt.format(p['se_laplace'])} & "
                   f"{fmt.format(p['se_bootstrap'])}\\\\")
    out += [r"\bottomrule", r"\end{tabular}}", ""]

    syn = r["synthetic"]
    for rk, macro in (("brown", "SynBrown"), ("freeform", "SynFree")):
        for s in syn["sigmas"]:
            name = f"{macro}{'A' if s == syn['sigmas'][0] else 'B'}"
            out += [rf"\newcommand{{\{name}}}{{", r"\begin{tabular}{l" + "c" * len(syn["N"]) + "}", r"\toprule",
                    rf"$\sigma={s}$\,px, $N=$ & " + " & ".join(str(n) for n in syn["N"]) + r"\\", r"\midrule"]
            res = syn["results"][rk][str(s)]
            fitted = [k for k, _ in SYN if k != "oracle"]
            for k, lab in SYN:
                cells = []
                for j in range(len(syn["N"])):
                    v = res[k][j][0]
                    is_best = k != "oracle" and v == min(res[m][j][0] for m in fitted)
                    cells.append(rf"\textbf{{{v:.2f}}}" if is_best else f"{v:.2f}")
                out.append(f"{lab} & " + " & ".join(cells) + r"\\")
                if k == "oracle":
                    out.append(r"\midrule")
            out += [r"\bottomrule", r"\end{tabular}}", ""]

    cov, ex = r["coverage"], r["real_extra"]
    nm, eb = full["noise_model"], full["error_budget"]
    scalars = {
        "CovOne": f"{cov['within_1sigma_pct']:.0f}", "CovTwo": f"{cov['within_2sigma_pct']:.0f}",
        "RmsZ": f"{cov['rms_z']:.2f}", "ErrCorr": f"{ex['error_correlation_gamma_brown']:.2f}",
        "DfEff": f"{full['df_eff']:.1f}", "NParams": str(full["n_params"]), "NFree": str(full["n_params_free"]),
        "SigmaLabel": f"{100 * nm['sigma_label']:.1f}", "SigmaPx": f"{nm['sigma_px']:.1f}",
        "Kappa": f"{nm['kappa']:.2f}", "LamSel": f"{full['smooth_selected']:g}",
        "BudgetClick": f"{eb['predicted_mape_click_only_pct']:.2f}",
        "BudgetTotal": f"{eb['predicted_mape_click_and_label_pct']:.2f}",
        "MapeVOne": f"{rows['v21']['mape_pct']:.2f}", "MapeLegacy": f"{rows['legacy']['mape_pct']:.2f}",
        "MedVOne": f"{rows['v21']['median_ape_pct']:.2f}",
    }
    m = r.get("matching")
    if m:
        scalars.update({"MatchAcc": f"{m['accepted_pct']:.0f}", "MatchPrec": f"{m['precision_pct']:.1f}",
                        "MatchMed": f"{m['median_err_pct_accepted']:.2f}", "MatchMs": f"{m['ms_per_click']:.0f}"})
    for k, v in scalars.items():
        out.append(rf"\newcommand{{\{k}}}{{{v}}}")
    with open(os.path.join(RES, "tables.tex"), "w", encoding="utf-8") as f:
        f.write("% generated by scripts/make_tex_tables.py — do not edit\n" + "\n".join(out) + "\n")
    print(f"wrote {os.path.join(RES, 'tables.tex')}")


if __name__ == "__main__":
    main()
