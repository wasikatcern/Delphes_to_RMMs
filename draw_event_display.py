#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse, sys

# ================== User-config ==================
ROOT_PATH  = "skimmed_delphes_selected.root"
EVENT_INDEX = 5
TREE_NAME  = None
OUTDIR = "event_displays"
ETA_GUIDES = [2.5, 4.0]   # keep these guides
ANGLE_SCALE = 4.0        # >1 makes guide lines steeper (bigger angle wrt X)
# =================================================

# Cone drawing controls
MIN_CONE_LENGTH   = 0.55   # absolute minimum drawn length (in plot units)
MAX_APERTURE_RATIO = 0.4  # cap (base_half_width / length) to avoid super-fat cones

os.makedirs(OUTDIR, exist_ok=True)

# Dependencies
try:
    import uproot
    import awkward as ak
except Exception as e:
    raise SystemExit(f"[FATAL] Need uproot+awkward. Try: pip install uproot awkward\n{e}")

# Colors
COLORS = {
    "jet": "#145105",
    "bjet": "#F90101",
    "electron": "#1f77b4",
    "muon": "#d0339b",
    "photon": "#452f7a",
    "MET": "#2f2ad2",
    "beam": "#000000",
    "eta25": "#444444",
    "eta40": "#7a3b2e",
}

PREFIX  = {"jet":"Jet ", "bjet":"bJet ", "electron":"e", "muon":"μ", "photon":"γ"}

def type_label(typ, idx):
    return f"{PREFIX.get(typ, typ)}{idx+1}"

def eta_to_theta(eta):
    return 2.0 * math.atan(math.exp(-eta))

def unit_vec_from_etaphi(eta, phi):
    """Return unit 3D direction vector (ux,uy,uz) from (eta,phi)."""
    theta = eta_to_theta(eta)
    st, ct = math.sin(theta), math.cos(theta)
    return np.array([st*math.cos(phi), st*math.sin(phi), ct])

def choose_tree(f, prefer=None):
    if prefer and prefer in f:
        t = f[prefer]
        if isinstance(t, uproot.behaviors.TTree.TTree):
            return t
    for k in f.keys():
        obj = f[k]
        if isinstance(obj, uproot.behaviors.TTree.TTree):
            b = obj.keys()
            if ("JET_pt" in b) and ("MET_met" in b):
                return obj
    for k in f.keys():
        obj = f[k]
        if isinstance(obj, uproot.behaviors.TTree.TTree):
            return obj
    return None

def load_event(path, evt_index, tree_name=None):
    f = uproot.open(path)
    t = choose_tree(f, tree_name)
    if t is None:
        raise RuntimeError("No suitable TTree found. Set TREE_NAME.")
    need = [
        "JET_pt","JET_eta","JET_phi","JET_mass",
        "bJET_pt","bJET_eta","bJET_phi","bJET_mass",
        "EL_pt","EL_eta","EL_phi",
        "MU_pt","MU_eta","MU_phi",
        "PH_pt","PH_eta","PH_phi",
        "MET_met","MET_phi","MET_eta"
    ]
    avail = set(t.keys())
    arr = {n: (t[n].array(library="ak") if n in avail else None) for n in need}
    n_events = None
    for v in arr.values():
        if v is not None:
            n_events = len(v); break
    def extract(pt,eta,phi,mass=None, typ="obj"):
        out = []
        if pt is None or eta is None or phi is None:
            return out
        pts, etas, phis = pt[evt_index], eta[evt_index], phi[evt_index]
        m = mass[evt_index] if mass is not None else None
        for i in range(len(pts)):
            out.append({
                "pt": float(pts[i]), "eta": float(etas[i]), "phi": float(phis[i]),
                "mass": float(m[i]) if m is not None and i < len(m) else 0.0,
                "type": typ, "index": i
            })
        return out
    data = {
        "jets":      extract(arr["JET_pt"], arr["JET_eta"], arr["JET_phi"], arr["JET_mass"], "jet"),
        "bjets":     extract(arr["bJET_pt"], arr["bJET_eta"], arr["bJET_phi"], arr["bJET_mass"], "bjet"),
        "electrons": extract(arr["EL_pt"],  arr["EL_eta"],  arr["EL_phi"],  None, "electron"),
        "muons":     extract(arr["MU_pt"],  arr["MU_eta"],  arr["MU_phi"],  None, "muon"),
        "photons":   extract(arr["PH_pt"],  arr["PH_eta"],  arr["PH_phi"],  None, "photon"),
        "met": None
    }
    # MET
    met_val = None; met_phi = None; met_eta = 0.0
    if arr["MET_met"] is not None and len(arr["MET_met"][evt_index])>0:
        met_val = float(arr["MET_met"][evt_index][0])
    if arr["MET_phi"] is not None and len(arr["MET_phi"][evt_index])>0:
        met_phi = float(arr["MET_phi"][evt_index][0])
    if arr["MET_eta"] is not None and len(arr["MET_eta"][evt_index])>0:
        met_eta = float(arr["MET_eta"][evt_index][0])
    if met_val is not None and met_phi is not None:
        data["met"] = {"met": met_val, "phi": met_phi, "eta": met_eta}
    return data

# -------- Helpers --------
def normalize(vx, vy):
    n = math.hypot(vx, vy)
    if n <= 0: return 0.0, 0.0
    return vx/n, vy/n

def draw_cone_2d(ax, dirx, diry, length, base_half_width, color, alpha=0.35):
    dx, dy = normalize(dirx, diry)
    bx, by = dx*length, dy*length
    nx, ny = -dy, dx
    x1, y1 = bx + nx*base_half_width, by + ny*base_half_width
    x2, y2 = bx - nx*base_half_width, by - ny*base_half_width
    ax.fill([0.0, x1, x2], [0.0, y1, y2], alpha=alpha, color=color, linewidth=0)

def dir2d_from_eta_phi(eta, phi):
    # Beam (z) -> X; transverse x-component -> Y.
    u = unit_vec_from_etaphi(eta, phi)
    x2d = float(u[2])  # uz -> horizontal (right = +eta)
    y2d = float(u[0])  # ux -> vertical
    return x2d, y2d

def remove_duplicate_jets(jets, bjets, tol=1e-6):
    filtered = []
    for j in jets:
        dup = False
        for b in bjets:
            if (math.isclose(j["pt"], b["pt"], rel_tol=1e-6, abs_tol=tol) and
                math.isclose(j["eta"], b["eta"], rel_tol=1e-6, abs_tol=tol) and
                math.isclose(j["phi"], b["phi"], rel_tol=1e-6, abs_tol=tol)):
                dup = True
                break
        if not dup:
            filtered.append(j)
    return filtered

def compute_pt_scaling(data):
    pts = []
    for key in ["jets","bjets","electrons","muons","photons"]:
        pts += [o["pt"] for o in data[key]]
    if data["met"]:
        pts.append(data["met"]["met"])
    if not pts:
        return 1.0, (0.35, 1.0), (0.30, 0.06)
    arr = np.array(pts, dtype=float)
    pref = np.percentile(arr, 95) if len(arr) > 4 else arr.max()
    pref = max(pref, 1.0)
    # larger pT => longer line; lower pT => wider cone
    return pref, (0.35, 1.0), (0.30, 0.06)

def scale_length(pt, pt_ref, Lmin, Lmax):
    frac = min(max(pt/pt_ref, 0.0), 1.0)
    return Lmin + frac*(Lmax - Lmin)

def scale_cone_width(pt, pt_ref, Wmax, Wmin):
    frac = min(max(pt/pt_ref, 0.0), 1.0)
    return Wmax - frac*(Wmax - Wmin)

# -------- Plotting --------
def plot_beam_2d(data, outpath, eta_guides=ETA_GUIDES):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.set_title(f"2D Event Display - Event #{EVENT_INDEX}")
    ax.set_xlabel("Beam axis (Z)")
    ax.set_xticklabels([])     # hide x-axis numbers
    ax.set_xticks([])          # (optional) remove tick marks too
    ax.set_ylabel("Transverse projection")

    xlim = 1.6
    # Draw beam axis (horizontal line) with an arrow indicating +X (beam direction)
    ax.annotate(
        "", xy=(1.55, 0), xytext=(-1.55, 0),
        arrowprops=dict(
        arrowstyle="->", color=COLORS["beam"], lw=2, alpha=0.5
        )
    )

    # NEW: vertical line at X=0 to denote η=0
    ax.axvline(0.0, color="#222222", linewidth=2, linestyle="-", alpha=0.0)

    # η guide lines with bigger angles (ANGLE_SCALE>1 => steeper)
    styles = []
    for eta in eta_guides:
        theta = eta_to_theta(eta)
        theta_scaled = min(theta * ANGLE_SCALE, math.pi/2 * 0.98)  # avoid infinity slope
        m = math.tan(theta_scaled)
        color = COLORS["eta25"] if abs(eta - 2.5) < 1e-6 else COLORS["eta40"]
        for sgn in (+1, -1):
            xs = np.array([-xlim, xlim])
            ys = sgn * m * xs
            ax.plot(xs, ys, linestyle="--", linewidth=2, color=color, alpha=0.3)
        styles.append((eta, color, theta_scaled))

    # Annotate η values with their respective colors
    if styles:
        y0 = 0.08
        dy = 0.05  # vertical spacing between the two labels
        for i, (eta, color, _) in enumerate(styles):
            ax.text(0.86, y0 + i*dy,
                    rf"$\eta = {eta}$",
                    transform=ax.transAxes,
                    fontsize=12,
                    color=color,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))


    # PT scaling params
    pt_ref, (Lmin, Lmax), (Wmax, Wmin) = compute_pt_scaling(data)

    # Remove duplicate jets that are actually b-jets
    jets_clean = remove_duplicate_jets(data["jets"], data["bjets"])

    # Jets/b-jets as cones
    for typ, coll in [("jet", jets_clean), ("bjet", data["bjets"])]:
        for o in coll:
            dx, dy = dir2d_from_eta_phi(o["eta"], o["phi"])

            # Length/width from pT
            L = scale_length(o["pt"], pt_ref, Lmin, Lmax)
            W = scale_cone_width(o["pt"], pt_ref, Wmax, Wmin)

            # Enforce a minimum length so low-pT cones don’t look too fat
            if L < MIN_CONE_LENGTH:
                # Scale width up proportionally to keep visual opening angle similar,
                # or clamp by MAX_APERTURE_RATIO (prevents "pizza slice" look).
                scale_up = MIN_CONE_LENGTH / max(L, 1e-9)
                L = MIN_CONE_LENGTH
                W = min(W * scale_up, L * MAX_APERTURE_RATIO)
            else:
                # Even for longer cones, keep a sane max aperture
                W = min(W, L * MAX_APERTURE_RATIO)

            draw_cone_2d(ax, dx, dy, L, W, COLORS[typ], alpha=0.35)

            tx, ty = normalize(dx, dy)            
            ax.text(tx*L*1.05, ty*L*1.05,
                    f"{type_label(typ,o['index'])}\n"
                    f"pT={o['pt']:.1f} GeV\nη={o['eta']:.2f}",
                    color=COLORS[typ], fontsize=8, ha="center", va="center")

    # e/μ/γ dashed lines, pT-scaled
    for typ, key in [("electron","electrons"),("muon","muons"),("photon","photons")]:
        for o in data[key]:
            dx, dy = dir2d_from_eta_phi(o["eta"], o["phi"])
            L = scale_length(o["pt"], pt_ref, Lmin, Lmax) * 3.0
            ax.plot([0, dx*L], [0, dy*L], linestyle="--", linewidth=2, color=COLORS[typ])
            ax.text(dx*L*1.05, dy*L*1.05,
                f"{type_label(typ,o['index'])}\n"
                f"pT={o['pt']:.1f} GeV\nη={o['eta']:.2f}",
                color=COLORS[typ], fontsize=8, ha="center", va="center")


    # MET arrow (transverse)
    if data["met"]:
        mphi = data["met"]["phi"]; mpt = data["met"]["met"]
        L = scale_length(mpt, pt_ref, Lmin, Lmax) * 1.5
        mx, my = 0.0, math.cos(mphi)
        mx, my = normalize(mx, my)
        ax.arrow(0, 0, mx*L, my*L, length_includes_head=True, head_width=0.03,
                 head_length=0.06, color=COLORS["MET"])
        ax.text(mx*L*1.05, my*L*1.05,
            f"MET\npT={mpt:.1f} GeV", color=COLORS["MET"], fontsize=9, ha="center", va="center")


    # Legend
    handles = [
        plt.Line2D([0],[0], color=COLORS["jet"], linewidth=8, label="jet (cone, pT-scaled)"),
        plt.Line2D([0],[0], color=COLORS["bjet"], linewidth=8, label="b-jet (cone, pT-scaled)"),
        plt.Line2D([0],[0], color=COLORS["electron"], linestyle="--", linewidth=2, label="electron (-- line)"),
        plt.Line2D([0],[0], color=COLORS["muon"], linestyle="--", linewidth=2, label="muon (-- line)"),
        plt.Line2D([0],[0], color=COLORS["photon"], linestyle="--", linewidth=2, label="photon (-- line)"),
        plt.Line2D([0],[0], color=COLORS["MET"], linewidth=2, label="MET (arrow, pT)"),
        #plt.Line2D([0],[0], color=COLORS["beam"], linewidth=2, label="beam axis (X) / η=0 vertical"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def cli_event_index(default_idx):
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-e", "--event", type=int, help="event index to draw")
    # parse_known so we can also handle odd patterns
    args, extras = parser.parse_known_args()

    if args.event is not None:
        return args.event

    # Fallback: support `python event_beam_display.py - event 4`
    # i.e., if the word "event" appears, take the next token as int
    if "event" in extras:
        i = extras.index("event")
        if i + 1 < len(extras):
            try:
                return int(extras[i + 1])
            except ValueError:
                pass
    return default_idx

def main():
    data = load_event(ROOT_PATH, EVENT_INDEX, TREE_NAME)
    outname = f"Event_display_{EVENT_INDEX}.png"
    plot_beam_2d(data, os.path.join(OUTDIR, outname))

if __name__ == "__main__":
    # override global EVENT_INDEX from CLI if provided
    EVENT_INDEX = cli_event_index(EVENT_INDEX)
    main()
