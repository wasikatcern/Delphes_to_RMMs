#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daw_event_from_csv.py

RMM CSV (.csv/.csv.gz) → reconstruct (pT≈ET, |η|, φ) and save TWO event displays:

1) POLAR (top-down):
   • φ : 0→2π (angle), |η| as concentric rings (|η| increases toward center).
   • Each object: radial line from the collision point at its φ; length ∝ pT,
     capped by its |η| ring so it lies in the correct η region.
   • MET drawn at φ=0.

2) η–φ (unwrapped):
   • x-axis: φ (0→2π), y-axis: η (−ηmax→+ηmax).
   • Each object: vertical tick centered at its η (we use +|η|) with length ∝ pT.
   • Vertical reference line at φ=0 and horizontal η=0 (θ=90°) line.
   • Horizontal guide lines at η = ±1.0, ±2.5, ±4.0 with θ(°) labels:
       +1.0 ↔  40.4°, +2.5 ↔   9.39°, +4.0 ↔   2.10°
       -1.0 ↔ 139.6°, -2.5 ↔ 170.6°, -4.0 ↔ 177.9°

Usage:
  python draw_event_from_csv.py --csv rmm_events_100.csv.gz --event 10
Common options:
  --cms 13000
  --eta-guides 0 1 2.5 4.0
  --eta-max 4.0
  --pt-scale 0.005      (polar radial scaling)
  --pt-scale-eta 0.02   (η–φ tick length scaling, in η units)
  --pairs               (print per-block pairwise Δφ debug)
  --outdir plots
"""

import argparse, gzip, math, os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# ---------------------- CSV / RMM helpers ----------------------
def load_csv(path: str) -> pd.DataFrame:
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return pd.read_csv(f)
    return pd.read_csv(path)

def infer_matrix_size(columns: List[str]) -> int:
    rc_cols = [c for c in columns if c.startswith("R") and "C" in c]
    m = int(round(math.sqrt(len(rc_cols))))
    if m * m != len(rc_cols):
        raise ValueError("RMM columns do not form a square matrix.")
    return m

def row_to_matrix(row: pd.Series, m: int) -> np.ndarray:
    M = np.zeros((m, m), dtype=float)
    for r in range(m):
        for c in range(m):
            key = f"R{r:02d}C{c:02d}"
            M[r, c] = row[key]
    return M

def block_index(t: int, k: int, maxN: int) -> int:
    return 1 + t * maxN + k  # MET at 0

def count_present_in_block(M: np.ndarray, t: int, maxN: int) -> int:
    n = 0
    for k in range(maxN):
        r = block_index(t, k, maxN)
        if np.allclose(M[r, :], 0.0) and np.allclose(M[:, r], 0.0):
            break
        n += 1
    return n

def recover_block_Et_eta_abs(M: np.ndarray, t: int, n_present: int, maxN: int, CMS: float
                             ) -> Tuple[np.ndarray, np.ndarray]:
    ET, ETA_abs = [], []
    for k in range(n_present):
        r = block_index(t, k, maxN)
        diag = M[r, r]
        if k == 0:
            et = diag * CMS
        else:
            I = diag
            et_prev = ET[-1]
            et = np.nan if abs(1.0 + I) < 1e-12 else et_prev * (1.0 - I) / (1.0 + I)
        ET.append(et)
        HL = M[r, 0]  # HL = cosh(y) - 1
        val = HL + 1.0
        ETA_abs.append(float(np.arccosh(val)) if val >= 1.0 else np.nan)
    return np.array(ET, float), np.array(ETA_abs, float)

def delta_phi_from_MT(ET_MET: float, ET_obj: float, MT: float) -> float:
    if not (ET_MET > 0 and ET_obj > 0 and MT >= 0):
        return np.nan
    c = 1.0 - (MT * MT) / (2.0 * ET_MET * ET_obj)
    c = max(-1.0, min(1.0, c))
    return float(math.acos(c))  # in [0, π]

def pairwise_delta_phi_within_block(M: np.ndarray, t: int, n_present: int, maxN: int,
                                    ET: np.ndarray, ETA_abs: np.ndarray, CMS: float
                                    ) -> np.ndarray:
    if n_present < 2:
        return np.full((n_present, n_present), np.nan)
    dphi = np.full((n_present, n_present), np.nan, dtype=float)
    np.fill_diagonal(dphi, 0.0)
    for i in range(n_present):
        for j in range(i + 1, n_present):
            r_i, r_j = block_index(t, i, maxN), block_index(t, j, maxN)
            m_ij = M[r_i, r_j] * CMS
            pt1, pt2 = ET[i], ET[j]
            if not (np.isfinite(pt1) and np.isfinite(pt2) and pt1 > 0 and pt2 > 0):
                continue
            dEta_same = abs(ETA_abs[i] - ETA_abs[j])
            dEta_opp  = ETA_abs[i] + ETA_abs[j]
            def cos_dphi_for(dE): return np.cosh(dE) - (m_ij*m_ij)/(2.0*pt1*pt2)
            cands = [cos_dphi_for(dEta_same), cos_dphi_for(dEta_opp)]
            def dist(x): return 0.0 if -1<=x<=1 else min(abs(x-1), abs(x+1))
            c = min(cands, key=dist)
            c = max(-1.0, min(1.0, float(c)))
            dphi[i, j] = dphi[j, i] = float(math.acos(c))
    return dphi

# ------------- φ sign solving (φ_MET = 0 reference) -------------
def wrap_dphi(a: float) -> float:
    return abs(((a + math.pi) % (2*math.pi)) - math.pi)

def solve_phi_signs(dphi_met: np.ndarray, dphi_pair: np.ndarray) -> np.ndarray:
    n = len(dphi_met)
    if n == 0: return np.array([], float)
    if n == 1 or not np.isfinite(dphi_pair).any(): return dphi_met.copy()
    best_cost, best_phi = float("inf"), None
    for mask in range(1 << n):
        s = np.array([1 if (mask >> k) & 1 else -1 for k in range(n)], int)
        phi = s * dphi_met  # φ_MET = 0
        cost, valid = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                if not np.isfinite(dphi_pair[i, j]): continue
                pred = wrap_dphi(phi[i] - phi[j])
                err = pred - dphi_pair[i, j]
                cost += err*err
                valid += 1
        if valid == 0:
            cost = np.sum((phi < 0).astype(float))
        if cost < best_cost:
            best_cost, best_phi = cost, phi.copy()
    return np.mod(best_phi, 2*np.pi)  # [0, 2π)

# ------------- helpers: η radius & θ degrees -------------
def eta_to_radius(abs_eta: float, eta_max: float, R_max: float) -> float:
    a = min(abs_eta, eta_max)
    return R_max * (eta_max - a) / eta_max

def eta_to_theta_deg(abs_eta: float) -> float:
    return 2.0 * math.degrees(math.atan(math.exp(-abs_eta)))

# ---------------------- plotting (polar) ----------------------
DEFAULT_COLORS = {
    "jet":      "#1f77b4",
    "bjet":     "#9467bd",
    "muon":     "#2ca02c",
    "electron": "#ff7f0e",
    "photon":   "#d62728",
    "met":      "#000000",
}

def type_title(typ: str) -> str:
    return "Bjet" if typ == "bjet" else typ.capitalize()

def plot_event_polar(objects: Dict[str, List[dict]],
                     title: str,
                     outpath: str,
                     eta_guides: List[float],
                     eta_max: float,
                     pt_scale: float):
    R_max = 1.0
    fig = plt.figure(figsize=(8, 8), dpi=160)
    ax = plt.subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rmax(R_max * 1.02)
    ax.set_rticks([])
    ax.grid(alpha=0.15, linestyle=":")

    # Base η guide rings + labels
    for g in sorted(set(abs(v) for v in eta_guides), reverse=True):
        r_g = eta_to_radius(g, eta_max, R_max)
        ax.plot(np.linspace(0, 2*np.pi, 361), np.full(361, r_g),
                color="gray", alpha=0.35, lw=0.8)
        ax.text(math.radians(10), r_g + 0.012,
                f"|η|={g:g} (θ≈{eta_to_theta_deg(g):.0f}°)",
                fontsize=8, ha="left", va="bottom", color="gray")

    ax.set_title(title, pad=14)
    ax.text(0.5, -0.06, "Beam axis (Z) out of the page; φ wraps 0°→360°; |η| increases toward center",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)

    # --- MET arrow (φ=0 direction) ---
    if "met" in objects and len(objects["met"]) == 1:
        m = objects["met"][0]
        r_met = min(pt_scale * m["pt"], R_max)

        # Draw MET as a thick arrow on the φ=0 axis
        ax.annotate(
            "", xy=(0, r_met), xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",          # classic arrow head
                color=DEFAULT_COLORS["met"],
                lw=2.0,                    # thicker for visibility
                alpha=0.9,
                shrinkA=0, shrinkB=0,
                mutation_scale=14          # controls arrowhead size
            )
        )
        # 3-line label with MET info
        ax.text(
            math.radians(2), r_met + 0.05 * R_max,
            f"MET\n pT={m['pt']:.0f} GeV\n η={m['eta']:.2f}",
            fontsize=9,
            ha="left",
            va="bottom",
            color=DEFAULT_COLORS["met"]
        )

    # --- inside plot_event_polar(...) right before the objects loop, hard-code cone params ---
    cone_min = 0.06     # min base "arc length" (in plot units at r=r_len)
    cone_max = 0.40     # max base
    cone_k   = 10.0     # width ~ cone_k / pT
    max_dtheta = 0.6    # cap angular width (radians)

    def draw_polar_cone(ax, theta, r_tip, pt, color):
        """Draw a filled triangular cone from origin to (theta, r_tip).
        Base 'arc length' ~ clip(cone_k / pt, cone_min, cone_max).
        Convert arc length to angular width: dtheta = base / r_tip (capped).
        """
        pt = max(1e-6, float(pt))
        base = max(cone_min, min(cone_k / pt, cone_max))
        # convert base arc to angular width at this radius; avoid blowup at small r
        dtheta = base / max(r_tip, 1e-6)
        dtheta = min(dtheta, max_dtheta)

        # triangle in polar coords: origin -> (theta - dθ/2, r_tip) -> (theta + dθ/2, r_tip)
        thetas = [theta, theta - 0.5 * dtheta, theta + 0.5 * dtheta]
        rs     = [0.0,  r_tip,                 r_tip]
        ax.fill(thetas, rs, facecolor=color, edgecolor=color, linewidth=1.2, alpha=0.28)

    # ----------------- Objects (cones for jets/bjets; dashed lines for e/μ) -----------------
    handles = {}
    type_counts: Dict[str, int] = {}
    for typ in ["jet", "bjet", "muon", "electron", "photon"]:
        if typ not in objects or len(objects[typ]) == 0:
            continue
        color = DEFAULT_COLORS[typ]
        type_counts[typ] = 0
        for o in objects[typ]:
            type_counts[typ] += 1
            idx_disp = type_counts[typ]

            r_eta = eta_to_radius(abs(o["eta"]), eta_max, R_max)     # ring for this |η|
            r_len = min(pt_scale * o["pt"], r_eta)                   # line length ∝ pT, capped to ring
            theta = o["phi"]                                         # radians [0, 2π)

            if typ in ("jet", "bjet"):
                # draw filled cone (no inner line)
                draw_polar_cone(ax, theta, r_len, o["pt"], color)
            else:
                # leptons dashed; photons solid
                linestyle = "--" if typ in ("muon", "electron") else "-"
                ax.plot([theta, theta], [0, r_len], color=color, lw=2.0, alpha=0.95, linestyle=linestyle)

            # label at the η ring (outside the cone/tick)
            label = f"{type_title(typ)} {idx_disp}\n pT={o['pt']:.0f} GeV\n η={abs(o['eta']):.2f}"
            ax.text(theta, r_eta + 0.015, label, fontsize=8, ha="center", va="bottom", color=color)

        # legend proxy
        handles[typ] = ax.plot([], [], color=color, lw=2.0, linestyle="--" if typ in ("muon","electron") else "-",
                           label=type_title(typ))[0]

    if handles:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), frameon=True, fontsize=9)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

# ---------------------- plotting (η–pT, cones for jets/bjets) ----------------------

def plot_event_etaphi(objects: Dict[str, List[dict]],
                      title: str,
                      outpath: str,
                      eta_max: float,
                      pt_scale_eta: float):
    """
    Rectangular η–pT view:
      • x = η in [-eta_max, +eta_max]
      • y = scaled pT (pt_scale_eta × GeV)
      • Jets and b-jets drawn as CONES (triangles) from (0,0) → (η, pt_scale_eta*pT)
            - Cone width w = clip(cone_k / pT, cone_min, cone_max)
            - Narrow for large pT, wider for small pT
      • Other particles remain as simple lines
      • Vertical η guides at 0, ±1, ±2.5, ±4.0 with θ labels
      • MET line and 3-line annotation
    """

    # --- Hard-coded cone parameters ---
    cone_min = 0.1      # minimum cone base width (η units)
    cone_max = 0.60      # maximum cone base width (η units)
    cone_k   = 10.0      # scaling factor controlling width vs pT

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

    # Determine plot ranges
    all_pts = [o["pt"] for typ in objects for o in objects[typ]]
    ymax = max(1.0, *(pt_scale_eta * p for p in all_pts)) * 1.15

    ax.set_xlim(-eta_max, eta_max)
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel("η")
    ax.set_ylabel("Scaled pT  (GeV)")
    ax.set_title(title)

    # --- Vertical η guides ---
    guides = [
        (-4.0, 177.9, "ultra-backward"),
        (-2.5, 170.6, "very backward"),
        (-1.0, 139.6, "backward"),
        ( 0.0,  90.0, "barrel (θ=90°)"),
        (+1.0,  40.4, "forward"),
        (+2.5,   9.39,"very forward"),
        (+4.0,   2.10,"ultra-forward"),
    ]
    for eta_val, theta_deg, region in guides:
        ax.axvline(eta_val, color="gray", lw=1.0, alpha=0.7, linestyle="--")
        xlab = np.clip(eta_val, -eta_max*0.98, eta_max*0.98)
        ax.text(
            xlab, ymax*0.98,
            (f"η={eta_val:+.1f}")
            if eta_val != 0.0 else "η=0  (θ=90°)\nbarrel",
            fontsize=8, ha="center", va="top", color="gray",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5)
        )

    # --- MET line + label (thicker line + arrowhead) ---
    if "met" in objects and len(objects["met"]) == 1:
        m = objects["met"][0]
        x_end = 0.0
        y_end = min(pt_scale_eta * m["pt"], ymax * 0.95)

        # Draw MET as a thick arrow instead of a plain line
        ax.arrow(
            0.0, 0.0,                # start at origin
            x_end, y_end,            # delta x, delta y
            color=DEFAULT_COLORS["met"],
            lw=2.0,
            alpha=0.9,
            length_includes_head=True,
            head_width=0.25 * eta_max / 10,   # control size of arrowhead (adjust as needed)
            head_length=0.05 * ymax           # arrowhead length proportional to plot scale
        )

        # 3-line label (pT and η info)
        label = f"MET\n pT={m['pt']:.0f} GeV\n η={m['eta']:.2f}"
        ax.text(
            x_end, y_end + (0.03 * ymax),
            label,
            fontsize=9,
            ha="center",
            va="bottom",
            color=DEFAULT_COLORS["met"]
        )


    # --- Helper to draw one cone (triangle) ---
    def draw_cone(ax, x_tip, y_tip, width, color, lw=1.2, alpha=0.35):
        nrm = math.hypot(x_tip, y_tip)
        if nrm < 1e-9:
            return
        ux, uy = x_tip / nrm, y_tip / nrm
        px, py = -uy, ux
        hx, hy = (width / 2.0) * px, (width / 2.0) * py
        tri = Polygon([(0.0, 0.0),
                   (x_tip + hx, y_tip + hy),
                   (x_tip - hx, y_tip - hy)],
                  closed=True, facecolor=color, edgecolor=color,
                  linewidth=lw, alpha=alpha)
        ax.add_patch(tri)
        # (No line drawn inside cone anymore)

    # --- Draw objects ---
    handles = {}
    type_counts: Dict[str, int] = {}
    for typ in ["jet", "bjet", "muon", "electron", "photon"]:
        if typ not in objects or len(objects[typ]) == 0:
            continue
        color = DEFAULT_COLORS[typ]
        type_counts[typ] = 0
        for o in objects[typ]:
            type_counts[typ] += 1
            idx_disp = type_counts[typ]

            eta = abs(o["eta"])  # replace with signed η if available
            x_end = np.clip(eta, -eta_max, eta_max)
            y_end = min(pt_scale_eta * o["pt"], ymax * 0.95)

            if typ in ("jet", "bjet"):
                # Cone width w = clip(cone_k / pT, cone_min, cone_max)
                pt = max(1e-6, float(o["pt"]))
                w = max(cone_min, min(cone_k / pt, cone_max))
                draw_cone(ax, x_end, y_end, width=w, color=color, alpha=0.30)
            else:
                # Non-jet objects: simple line
                # Electrons and muons as dashed lines
                linestyle = "--" if typ in ("muon", "electron") else "-"
                ax.plot([0.0, x_end], [0.0, y_end],
                        color=color, lw=2.0, alpha=0.95, linestyle=linestyle)

        # Label near tip
            label = f"{type_title(typ)} {idx_disp}\n pT={o['pt']:.0f} GeV\n η={eta:.2f}"
            ha = "center"
            if x_end > eta_max * 0.9:
                ha = "right"
            elif x_end < -eta_max * 0.9:
                ha = "left"
            ax.text(x_end, y_end + 0.02 * ymax, label,
                fontsize=8, ha=ha, va="bottom", color=color)

        handles[typ] = ax.plot([], [], color=color, lw=2.0,
                           label=type_title(typ))[0]


    if handles:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02),
                  frameon=True, fontsize=9)

    ax.grid(alpha=0.25, linestyle=":")
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="RMM CSV → (pT, |η|, φ) event displays: polar and η–φ.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--event", type=int, required=True)
    ap.add_argument("--cms", type=float, default=13000.0)
    ap.add_argument("--eta-guides", type=float, nargs="*", default=[0, 1, 2.5, 4.0])
    ap.add_argument("--eta-max", type=float, default=4.0)
    ap.add_argument("--pt-scale", type=float, default=0.005, help="GeV→radius scaling (polar)")
    ap.add_argument("--pt-scale-eta", type=float, default=0.02, help="GeV→η tick length scaling (η–φ)")
    ap.add_argument("--pairs", action="store_true", help="Print pairwise Δφ tables per block")
    ap.add_argument("--outdir", type=str, default="event_from_csv_plots")

    args = ap.parse_args()

    df = load_csv(args.csv)
    if args.event < 0 or args.event >= len(df):
        raise IndexError(f"--event {args.event} out of range [0, {len(df)-1}]")

    m = infer_matrix_size(df.columns)
    TYPES = 5
    maxN = (m - 1) // TYPES
    if TYPES * maxN + 1 != m:
        print(f"[warn] Matrix size {m} != 5*maxN + 1 (proceeding).")

    type_labels = ["jet", "bjet", "muon", "electron", "photon"]

    # Build RMM for selected event
    row = df.iloc[args.event]
    M = row_to_matrix(row, m)
    ET_MET = M[0, 0] * args.cms

    objects_for_plot: Dict[str, List[dict]] = {t: [] for t in type_labels}
    objects_for_plot["met"] = [{"pt": ET_MET, "eta": 0.0, "phi": 0.0, "idx": 0}]

    for t, lab in enumerate(type_labels):
        n_present = count_present_in_block(M, t, maxN)
        if n_present == 0:
            continue

        ET, ETA_abs = recover_block_Et_eta_abs(M, t, n_present, maxN, args.cms)

        # Δφ(MET,obj_k)
        dphi_met = np.zeros(n_present, dtype=float)
        for k in range(n_present):
            c = block_index(t, k, maxN)
            MT = M[0, c] * args.cms
            dphi_met[k] = delta_phi_from_MT(ET_MET, ET[k], MT)

        # Pairwise Δφ within the block
        dphi_pair = pairwise_delta_phi_within_block(M, t, n_present, maxN, ET, ETA_abs, args.cms)
        if args.pairs and n_present >= 2:
            print(f"\nEvent {args.event}  [{lab}] pairwise Δφ (rad):")
            for i in range(n_present):
                row_str = " ".join(f"{dphi_pair[i,j]:6.3f}" if np.isfinite(dphi_pair[i,j]) else "  nan  "
                                   for j in range(n_present))
                print(f"{lab}[{i:>2d}] {row_str}")

        # Solve φ signs (φ_MET = 0) and map into [0, 2π)
        phi_vals = solve_phi_signs(dphi_met, dphi_pair)

        # Collect objects (η sign not recoverable → use +|η|)
        for k in range(n_present):
            objects_for_plot[lab].append({
                "pt": float(ET[k]),
                "eta": float(ETA_abs[k]),
                "phi": float(phi_vals[k]),
                "idx": k
            })

    # ---- Save both plots ----
    base = os.path.join(args.outdir, f"event_{args.event:05d}")
    polar_path  = base + "_polar.png"
    etaphi_path = base + "_etaphi.png"

    title_polar  = f"Event {args.event} — Polar (φ 0→360°, η rings)"
    title_etaphi = f"Event {args.event} — pT-η view"

    plot_event_polar(objects_for_plot, title_polar, polar_path,
                     eta_guides=args.eta_guides, eta_max=args.eta_max, pt_scale=args.pt_scale)

    plot_event_etaphi(objects_for_plot, title_etaphi, etaphi_path,
                      eta_max=args.eta_max, pt_scale_eta=args.pt_scale_eta)

    print(f"[info] Saved: {polar_path}")
    print(f"[info] Saved: {etaphi_path}")

if __name__ == "__main__":
    main()
