"""Makes paper figures. Run from the project root: python3 paper/make_figures.py"""
import glob, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"font.family":"serif","axes.spines.top":False,"axes.spines.right":False})
OUT="paper/figures/"

# Fig 1: oracle (perfect-preview, non-causal) saving. Values checked against
# results/gate3b/{oracle_summary,scenarios_summary,s2_fixed_summary,s1b_summary}.txt.
# Speed breakers use S1b (oracle-map vs ACC-map, realistic speeds), not the failed S1 set.
labels=["OpenACC, all (n=160)","OpenACC, dynamic third","S2 cut-ins (n=45)","S3 hard braking (n=45)","S1b speed breakers (n=42)*"]
vals=[2.83,7.82,11.21,18.34,17.80]
fig,ax=plt.subplots(figsize=(3.5,2.4))
y=range(len(vals))[::-1]
ax.barh(list(y),vals,color="#4a6fa5")
for yi,v in zip(y,vals): ax.text(v+0.3,yi,f"{v:.1f}",va="center",fontsize=8)
ax.set_yticks(list(y)); ax.set_yticklabels(labels,fontsize=7.5)
ax.set_xlabel("Median energy saving of oracle MPC vs ACC (%)"); ax.set_xlim(0,21)
fig.tight_layout(); fig.savefig(OUT+"fig_oracle_ceiling.pdf"); fig.savefig(OUT+"fig_oracle_ceiling.png",dpi=300); plt.close(fig)

# Fig 2: S1b per-window change MPC-map-L vs ACC-map (whkm_adj, as in the paper's rule)
def L(a):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"results/gate3b/s1b_test_{a}_*.csv"))]).drop_duplicates("win").set_index("win")
A,M=L("acc_map"),L("mpc_map_L")
d=((M["whkm_adj"]/A["whkm_adj"]-1)*100).sort_values()
fig,ax=plt.subplots(figsize=(3.5,2.6))
ax.bar(range(len(d)),d.values,color=["#2a7f62" if x<0 else "#b5473a" for x in d.values])
ax.axhline(d.median(),ls="--",c="k",lw=0.8); ax.text(len(d)-1,d.median()-1.8,f"median {d.median():.2f}%",ha="right",fontsize=8)
ax.set_xlabel("Window (sorted)"); ax.set_ylabel("Energy change of MPC-map-L\nvs ACC-map (%)")
fig.tight_layout(); fig.savefig(OUT+"fig_s1b_per_window.pdf"); fig.savefig(OUT+"fig_s1b_per_window.png",dpi=300); plt.close(fig)
print(len(d),(d<0).sum(),d.median())
