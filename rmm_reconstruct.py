#!/usr/bin/env python3
"""
rmm_reconstruct.py — reconstruct kinematics from an RMM CSV (.csv or .csv.gz)
New option:
  --event N   → print only that event (0-based index)

# Print only event #5
python rmm_reconstruct.py --csv rmm_events_100.csv --event 5

# Also include pairwise Δφ matrices
python rmm_reconstruct.py --csv rmm_events_100.csv --event 5 --pairs

# Save tidy per-object table for that one event
python rmm_reconstruct.py --csv rmm_events_100.csv --event 5 --save reco_evt5.csv
"""

import argparse, gzip, math, sys
import numpy as np, pandas as pd

# ---------------------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame:
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return pd.read_csv(f)
    return pd.read_csv(path)

def infer_matrix_size(columns):
    rc = [c for c in columns if c.startswith("R") and "C" in c]
    m = int(round(math.sqrt(len(rc))))
    if m*m != len(rc): raise ValueError("RMM columns not square")
    return m

def row_to_matrix(row, m):
    M = np.zeros((m,m))
    for r in range(m):
        for c in range(m):
            key = f"R{r:02d}C{c:02d}"
            M[r,c] = row[key]
    return M

def block_index(t,k,maxN): return 1 + t*maxN + k

def count_present(M,t,maxN):
    n=0
    for k in range(maxN):
        r=block_index(t,k,maxN)
        if np.allclose(M[r,:],0) and np.allclose(M[:,r],0): break
        n+=1
    return n

def recover_block(M,t,n,maxN,CMS):
    ET,ETA=[],[]
    for k in range(n):
        r=block_index(t,k,maxN)
        diag=M[r,r]
        if k==0: et=diag*CMS
        else:
            I=diag; et_prev=ET[-1]
            et=et_prev*(1-I)/(1+I) if abs(1+I)>1e-12 else np.nan
        ET.append(et)
        HL=M[r,0]; val=HL+1
        eta=math.acosh(val) if val>=1 else np.nan
        ETA.append(eta)
    return np.array(ET),np.array(ETA)

def dphi_from_MT(ETmet,ETobj,MT):
    if not (ETmet>0 and ETobj>0 and MT>=0): return np.nan
    c=1-(MT*MT)/(2*ETmet*ETobj)
    c=max(-1,min(1,c))
    return math.acos(c)

def pairwise_dphi(M,t,n,maxN,ET,ETA,CMS):
    if n<2: return np.full((n,n),np.nan)
    dphi=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            m_ij=M[block_index(t,i,maxN),block_index(t,j,maxN)]*CMS
            pt1,pt2=ET[i],ET[j]
            dEta_same=abs(ETA[i]-ETA[j]); dEta_opp=ETA[i]+ETA[j]
            def cand(dE): return np.cosh(dE)-(m_ij*m_ij)/(2*pt1*pt2)
            cands=[cand(dEta_same),cand(dEta_opp)]
            c=min(cands,key=lambda x:0 if -1<=x<=1 else min(abs(x-1),abs(x+1)))
            c=max(-1,min(1,c))
            d=math.acos(c)
            dphi[i,j]=dphi[j,i]=d
    return dphi
# ---------------------------------------------------------------------

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",required=True)
    ap.add_argument("--cms",type=float,default=13000.0)
    ap.add_argument("--pairs",action="store_true")
    ap.add_argument("--event",type=int,default=None,
                    help="Print only this event (0-based index)")
    ap.add_argument("--save",type=str,default=None)
    args=ap.parse_args()

    df=load_csv(args.csv)
    m=infer_matrix_size(df.columns)
    TYPES=5; maxN=(m-1)//TYPES
    labels=["jet","bjet","muon","electron","photon"]

    ev_range=range(len(df)) if args.event is None else [args.event]
    tidy=[]
    for ev in ev_range:
        if ev<0 or ev>=len(df): 
            print(f"Event {ev} out of range (0-{len(df)-1})"); sys.exit(1)
        row=df.iloc[ev]; M=row_to_matrix(row,m)
        ETmet=M[0,0]*args.cms
        print(f"\n=== Event {ev} ===\nMET_ET: {ETmet:.3f} GeV")

        for t,lab in enumerate(labels):
            n=count_present(M,t,maxN)
            if n==0: continue
            ET,ETA=recover_block(M,t,n,maxN,args.cms)
            print(f"\n[{lab}]  n={n}")
            print(" idx   ET≈pT[GeV]   |eta|     Δφ(MET,obj)[rad]")
            for k in range(n):
                MT=M[0,block_index(t,k,maxN)]*args.cms
                dphi=dphi_from_MT(ETmet,ET[k],MT)
                print(f" {k:3d}   {ET[k]:10.2f}   {ETA[k]:7.3f}    {dphi:10.4f}")
                tidy.append({"event":ev,"type":lab,"index":k,
                             "ET_GeV":ET[k],"abs_eta":ETA[k],
                             "dphi_MET_obj":dphi})
            if args.pairs and n>1:
                dphiM=pairwise_dphi(M,t,n,maxN,ET,ETA,args.cms)
                print("\nPairwise Δφ (rad):")
                for i in range(n):
                    row_str=" ".join(f"{dphiM[i,j]:6.3f}" for j in range(n))
                    print(f"{lab}[{i}] {row_str}")

    if args.save:
        pd.DataFrame(tidy).to_csv(args.save,index=False)
        print(f"\nSaved tidy output to {args.save}")

if __name__=="__main__":
    main()
