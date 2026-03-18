#!/usr/bin/env python3
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


tabgap_file = "draw_pics/LTC_tabGAP.txt"
nep_file = "draw_pics/LTC_NEP.csv"
output_file = "draw_pics/output/ltc_direction_comparison.png"

a_len = 12.214
b_len = 3.037
c_len = 5.798
beta_deg = 103.83

sim_x_hkl = np.array([2.0, 0.0, 1.0])
sim_y_hkl = np.array([0.0, 1.0, 0.0])
sim_z_reference_hkl = np.array([0.0, 0.0, 1.0])

target_directions = {
    "[100]": np.array([1.0, 0.0, 0.0]),
    "[010]": np.array([0.0, 1.0, 0.0]),
    "[001]": np.array([0.0, 0.0, 1.0]),
}

exp_mean = {
    "[100]": 9.5,
    "[010]": 22.5,
    "[001]": 13.3,
}

exp_sem = {
    "[100]": 1.8,
    "[010]": 2.5,
    "[001]": 1.8,
}


def unit_vector(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length vector")
    return v / n


def crystal_basis():
    beta = np.deg2rad(beta_deg)
    a_vec = np.array([a_len, 0.0, 0.0])
    b_vec = np.array([0.0, b_len, 0.0])
    c_vec = np.array([c_len * np.cos(beta), 0.0, c_len * np.sin(beta)])
    return a_vec, b_vec, c_vec


def hkl_to_cart(hkl):
    a_vec, b_vec, c_vec = crystal_basis()
    u, v, w = hkl
    return u * a_vec + v * b_vec + w * c_vec


def build_sim_axes():
    ex = unit_vector(hkl_to_cart(sim_x_hkl))
    ey = unit_vector(hkl_to_cart(sim_y_hkl))
    ez = unit_vector(np.cross(ex, ey))
    z_ref = hkl_to_cart(sim_z_reference_hkl)
    if np.dot(ez, z_ref) < 0:
        ez = -ez
    return ex, ey, ez


def direction_in_sim_axes(hkl):
    ex, ey, ez = build_sim_axes()
    n = unit_vector(hkl_to_cart(hkl))
    n_sim = np.array([np.dot(n, ex), np.dot(n, ey), np.dot(n, ez)])
    return unit_vector(n_sim)


def parse_tabgap(path):
    raw = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, val = line.split(":")
            raw[key.strip()] = float(val.strip())
    k = np.zeros((3, 3), dtype=float)
    sem = np.zeros((3, 3), dtype=float)
    k[0, 0] = raw["kappa_xx_mean"]
    k[1, 1] = raw["kappa_yy_mean"]
    k[2, 2] = raw["kappa_zz_mean"]
    sem[0, 0] = raw["kappa_xx_sem"]
    sem[1, 1] = raw["kappa_yy_sem"]
    sem[2, 2] = raw["kappa_zz_sem"]
    return k, sem


def parse_nep(path):
    k = np.zeros((3, 3), dtype=float)
    sem = np.zeros((3, 3), dtype=float)
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in lines[1:]:
        parts = [x for x in re.split(r"\t+", line) if x]
        if len(parts) < 3:
            continue
        comp = parts[0].replace("κ_{", "").replace("}", "")
        mean = float(parts[1])
        sem_v = float(parts[-1])
        i = "xyz".index(comp[0])
        j = "xyz".index(comp[1])
        k[i, j] = mean
        sem[i, j] = sem_v
    k_sym = 0.5 * (k + k.T)
    sem_sym = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            sem_sym[i, j] = 0.5 * np.sqrt(sem[i, j] ** 2 + sem[j, i] ** 2)
    return k_sym, sem_sym


def project_kappa(k_tensor, sem_tensor, direction_hkl):
    n = direction_in_sim_axes(direction_hkl)
    value = float(n @ k_tensor @ n)
    coeff = np.outer(n, n)
    err = float(np.sqrt(np.sum((coeff * sem_tensor) ** 2)))
    return value, err, n


def collect_model_values(k_tensor, sem_tensor):
    values = {}
    for label, hkl in target_directions.items():
        v, e, n = project_kappa(k_tensor, sem_tensor, hkl)
        values[label] = {"mean": v, "sem": e, "n_sim": n}
    return values


def plot_grouped_bar(tabgap_values, nep_values, output_path):
    plt.rcParams["font.family"] = "Arial"
    figsize = 10
    fontsize = 13
    n = 100
    x0 = 10 * n
    y0 = 7 * n
    fig = plt.figure(figsize=(figsize, y0 / (x0 / figsize)))
    gs = GridSpec(y0, x0, figure=fig, width_ratios=np.ones(x0), height_ratios=np.ones(y0))
    ax = fig.add_subplot(gs[0:y0, 0:x0])

    labels = ["[100]", "[010]", "[001]"]
    x = np.arange(len(labels))
    width = 0.24

    tabgap_means = [tabgap_values[d]["mean"] for d in labels]
    tabgap_errs = [tabgap_values[d]["sem"] for d in labels]
    nep_means = [nep_values[d]["mean"] for d in labels]
    nep_errs = [nep_values[d]["sem"] for d in labels]
    exp_means = [exp_mean[d] for d in labels]
    exp_errs = [exp_sem[d] for d in labels]

    ax.bar(x - width, tabgap_means, width, yerr=tabgap_errs, capsize=4, label="tabGAP", color="#d84315", alpha=0.9)
    ax.bar(x, nep_means, width, yerr=nep_errs, capsize=4, label="NEP", color="#0d47a1", alpha=0.9)
    ax.bar(x + width, exp_means, width, yerr=exp_errs, capsize=4, label="Experiment", color="#2e7d32", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_ylabel("Lattice Thermal Conductivity (W/mK)", fontsize=fontsize, fontweight="bold")
    ax.set_xlabel("Crystallographic Direction", fontsize=fontsize, fontweight="bold")
    ax.tick_params(axis="y", labelsize=fontsize - 1)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=fontsize - 1, loc="upper right")
    ymax = max(tabgap_means + nep_means + exp_means)
    ax.set_ylim(0, ymax * 1.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    print(f"图已保存: {output_path}")


def print_conversion_report(tabgap_values, nep_values):
    labels = ["[100]", "[010]", "[001]"]
    print("\n方向投影（晶向在模拟坐标中的单位向量 n = [nx, ny, nz]）")
    for d in labels:
        n = nep_values[d]["n_sim"]
        print(f"{d}: n = [{n[0]: .4f}, {n[1]: .4f}, {n[2]: .4f}]")
    print("\n热导对比 (W/mK)")
    print(f"{'Direction':<10}{'tabGAP':>16}{'NEP':>16}{'Experiment':>16}")
    for d in labels:
        tg = tabgap_values[d]
        npv = nep_values[d]
        ex = exp_mean[d]
        ex_err = exp_sem[d]
        print(
            f"{d:<10}"
            f"{tg['mean']:>8.3f}±{tg['sem']:<7.3f}"
            f"{npv['mean']:>8.3f}±{npv['sem']:<7.3f}"
            f"{ex:>8.3f}±{ex_err:<7.3f}"
        )


def main():
    root = Path.cwd()
    tabgap_path = root / tabgap_file
    nep_path = root / nep_file
    out_path = root / output_file

    tabgap_k, tabgap_sem_tensor = parse_tabgap(tabgap_path)
    nep_k, nep_sem_tensor = parse_nep(nep_path)

    tabgap_values = collect_model_values(tabgap_k, tabgap_sem_tensor)
    nep_values = collect_model_values(nep_k, nep_sem_tensor)

    print_conversion_report(tabgap_values, nep_values)
    plot_grouped_bar(tabgap_values, nep_values, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
