"""Stage 2: residual ML benchmark. Usage: python3 python/ml_bench.py <model> <mode>
model in {physics, ridge, knn, svr, lgbm, bilstm}; mode in {resid, pure} (pure = predict P_batt directly, no physics).
Same 5 folds as stage 1. Metrics on held-out trips only (2 Hz): power RMSE/MAE/R2, trip-energy median|err| and pooled err, timing.
Results -> results/ml/<model>_<mode>.json"""
import sys,os,json,time,numpy as np
d=np.load('cache/ml_data.npz',allow_pickle=True); model,mode=sys.argv[1],sys.argv[2]
X=d['X'].astype(np.float64); Pb=d['Pb']; trip=d['trip']; tt=d['t']; ft=d['fold_of_trip']; fold=ft[trip]; fn=list(d['feat_names'])
rng=np.random.default_rng(42)
def tripE(P,idx):  # per-trip energy kWh from 2Hz power
    out={}
    for n in np.unique(trip[idx]):
        m=idx[trip[idx]==n]; out[n]=np.trapezoid(P[m],tt[m])/3.6e6
    return out
def metrics(pred,k,tgt_idx):
    y=Pb[tgt_idx]; p=pred; e=p-y
    r=dict(rmse_kw=float(np.sqrt(np.mean(e**2))/1000),mae_kw=float(np.mean(abs(e))/1000),r2=float(1-np.sum(e**2)/np.sum((y-y.mean())**2)))
    em=tripE(np.where(np.isin(np.arange(len(Pb)),tgt_idx),np.nan,0)+0,tgt_idx) if False else None
    Pfull=np.zeros(len(Pb)); Pfull[tgt_idx]=p; Yfull=Pb
    E1=tripE(Pfull,tgt_idx); E0=tripE(Yfull,tgt_idx)
    errs=np.array([100*(E1[n]-E0[n])/E0[n] for n in E0]); r['trip_err']=errs.tolist(); r['e_model']=float(sum(E1.values())); r['e_meas']=float(sum(E0.values()))
    return r
def prep(tr,te):
    med=np.nanmedian(X[tr],axis=0); med=np.where(np.isnan(med),0,med)
    f=lambda A:np.where(np.isnan(A),med,A); mu=f(X[tr]).mean(0); sd=f(X[tr]).std(0)+1e-9
    return (f(X[tr])-mu)/sd,(f(X[te])-mu)/sd
res=[]; t_train=0; t_inf=0
for k in range(5):
    tr=np.where(fold!=k)[0]; te=np.where(fold==k)[0]; Pp=d['Pphys_%d'%k]
    target=(Pb-Pp) if mode=='resid' else Pb; base=Pp if mode=='resid' else np.zeros_like(Pb)
    t0=time.time()
    if model=='physics': pred=Pp[te]; ti=0; tr_t=0
    else:
        Xtr,Xte=prep(tr,te); ytr=target[tr]
        if mode=='resid': Xtr=np.column_stack([Xtr,(d['Pw_%d'%k][tr]-d['Pw_%d'%k][tr].mean())/ (d['Pw_%d'%k][tr].std()+1e-9)]); Xte=np.column_stack([Xte,(d['Pw_%d'%k][te]-d['Pw_%d'%k][tr].mean())/(d['Pw_%d'%k][tr].std()+1e-9)])
        if model=='lgbm':
            import lightgbm as lgb; m=lgb.LGBMRegressor(n_estimators=400,learning_rate=0.05,num_leaves=31,min_child_samples=50,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=1.0,random_state=42,n_jobs=2,verbose=-1)
            m.fit(Xtr,ytr); tr_t=time.time()-t0; t1=time.time(); out=m.predict(Xte)
        elif model=='ridge':
            from sklearn.linear_model import Ridge; m=Ridge(alpha=10.0).fit(Xtr,ytr); tr_t=time.time()-t0; t1=time.time(); out=m.predict(Xte)
        elif model=='knn':
            from sklearn.neighbors import KNeighborsRegressor; sel=rng.choice(len(Xtr),min(70000,len(Xtr)),replace=False)
            m=KNeighborsRegressor(n_neighbors=25,weights='distance',n_jobs=2).fit(Xtr[sel],ytr[sel]); tr_t=time.time()-t0; t1=time.time(); out=m.predict(Xte)
        elif model=='svr':
            from sklearn.svm import SVR; sel=rng.choice(len(Xtr),8000,replace=False); sc=np.std(ytr[sel])
            m=SVR(kernel='rbf',C=3.0,epsilon=0.05,gamma='scale').fit(Xtr[sel],ytr[sel]/sc); tr_t=time.time()-t0; t1=time.time(); out=m.predict(Xte)*sc
        elif model=='bilstm':
            import torch,torch.nn as nn; torch.manual_seed(42); torch.set_num_threads(2); W=20
            sc=ytr.std(); Xall=np.zeros((len(Pb),Xtr.shape[1])); Xall[tr]=Xtr; Xall[te]=Xte
            def win(idx):  # window of W past steps within the same trip (edge-pad)
                ii=idx[:,None]+np.arange(-W+1,1)[None,:]; ii=np.clip(ii,0,len(Pb)-1); bad=trip[ii]!=trip[idx][:,None]
                first=np.where(bad,np.inf,np.arange(W)[None,:]).min(1).astype(int); ii=np.where(bad,np.take_along_axis(ii,first[:,None],1),ii); return torch.tensor(Xall[ii],dtype=torch.float32)
            class Net(nn.Module):
                def __init__(s,nf): super().__init__(); s.l=nn.LSTM(nf,32,batch_first=True,bidirectional=True); s.h=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
                def forward(s,x): o,_=s.l(x); return s.h(o[:,-1]).squeeze(-1)
            net=Net(Xtr.shape[1]); opt=torch.optim.Adam(net.parameters(),lr=3e-3); sel=rng.choice(tr,min(60000,len(tr)),replace=False); yt=torch.tensor(target[sel]/sc,dtype=torch.float32)
            for ep in range(6):
                pm=rng.permutation(len(sel))
                for b in range(0,len(sel),512):
                    j=pm[b:b+512]; opt.zero_grad(); loss=nn.functional.mse_loss(net(win(sel[j])),yt[j]); loss.backward(); opt.step()
            tr_t=time.time()-t0; t1=time.time(); net.eval(); 
            with torch.no_grad(): out=np.concatenate([net(win(te[b:b+4096])).numpy() for b in range(0,len(te),4096)])*sc
        ti=(time.time()-t1)/len(te)*1e6; pred=(out+base[te]) if mode=='resid' else out
    r=metrics(pred,k,te); r['fold']=k; r['train_s']=tr_t; r['infer_us_per_sample']=ti; res.append(r); print(model,mode,'fold',k,'rmse %.2f kW r2 %.3f trip med|e| %.1f%%'%(r['rmse_kw'],r['r2'],np.median(abs(np.array(r['trip_err'])))),flush=True)
allerr=np.concatenate([r['trip_err'] for r in res]); em=sum(r['e_model'] for r in res); e0=sum(r['e_meas'] for r in res)
summ=dict(model=model,mode=mode,rmse_kw=float(np.mean([r['rmse_kw'] for r in res])),rmse_kw_sd=float(np.std([r['rmse_kw'] for r in res])),mae_kw=float(np.mean([r['mae_kw'] for r in res])),r2=float(np.mean([r['r2'] for r in res])),
  trip_med_abs=float(np.median(abs(allerr))),trip_mean_abs=float(np.mean(abs(allerr))),trip_p90=float(np.quantile(abs(allerr),.9)),pooled=float(100*(em-e0)/e0),within5=float(100*np.mean(abs(allerr)<5)),
  train_s=float(np.mean([r['train_s'] for r in res])),infer_us=float(np.mean([r['infer_us_per_sample'] for r in res])),folds=res)
os.makedirs('results/ml',exist_ok=True); json.dump(summ,open('results/ml/%s_%s.json'%(model,mode),'w'))
print('SUMMARY',model,mode,{k:round(v,3) for k,v in summ.items() if isinstance(v,float)})
