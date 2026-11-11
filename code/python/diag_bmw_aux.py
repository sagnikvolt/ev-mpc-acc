import sys,glob,os,numpy as np,pandas as pd
sys.path.insert(0,'python'); import fit_bmw_i3 as F
rows=[]
for f in sorted(glob.glob('data/raw/bmw_i3/measurement/Trip*.csv')):
    n=os.path.basename(f)[:-4]
    if n in F.EXCLUDE: continue
    d=pd.read_csv(f,sep=';',encoding='latin-1'); x=F.load(f)
    st=(x['v']<0.1)&(abs(x['a'])<0.05)
    amb=F.col(d,'Ambient Temperature [')
    hp=F.col(d,'Heating Power CAN'); ac=F.col(d,'AirCon Power'); hl=F.col(d,'Heating Power LIN'); hv=F.col(d,'Heater Voltage'); hi=F.col(d,'Heater Current')
    rows.append(dict(trip=n,cat=n[4],stand_kw=np.median(x['Pb'][st])/1000 if st.sum()>50 else np.nan,n_st=st.sum(),
      Pb_mean=x['Pb'].mean()/1000,amb=np.nanmean(amb),heatCAN=np.nanmean(hp),ac=np.nanmean(ac),heatLIN_kw=np.nanmean(hl)/1000,heater_UI_kw=np.nanmean(hv*hi)/1000,pc_kw=x['pc'].mean()/1000,nan_pc=np.isnan(hp).mean()))
r=pd.DataFrame(rows); pd.set_option('display.width',220)
print(r.groupby('cat')[['stand_kw','amb','heatCAN','ac','heatLIN_kw','heater_UI_kw','nan_pc']].describe().T.round(2).loc[(slice(None),['mean','min','max']),:].to_string())
v=r.dropna(subset=['stand_kw'])
for c in ['amb','heatCAN','ac','heatLIN_kw','heater_UI_kw','pc_kw']:
    q=v.dropna(subset=[c]); print(c,'corr with standstill power',round(np.corrcoef(q.stand_kw,q[c])[0,1],2),'n',len(q))
