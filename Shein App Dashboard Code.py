from pathlib import Path
import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Shein APAC Intelligence Suite", page_icon="\U0001f6cd\ufe0f", layout="wide", initial_sidebar_state="expanded")
BASE_DIR = Path(__file__).resolve().parent
TARGET = "churn_label"

st.markdown("""
<style>
.block-container {padding-top:1rem;padding-bottom:1.2rem;padding-left:1.4rem;padding-right:1.4rem;max-width:100%;}
section[data-testid="stSidebar"] > div {background:#0d0a0f;}
.main-shell {padding:1.1rem 1.35rem;border-radius:18px;background:linear-gradient(135deg,#120a10 0%,#1a0d17 100%);border:1px solid #2d1628;margin-bottom:1rem;}
.hero-title {font-size:2.1rem;font-weight:800;color:#ffeef4;line-height:1.05;margin:0 0 0.3rem 0;}
.hero-sub {font-size:0.95rem;color:#c9a0b4;margin:0;}
.tag-row {display:flex;gap:0.4rem;flex-wrap:wrap;margin-top:0.7rem;}
.tag {padding:0.28rem 0.6rem;border-radius:999px;background:#2a0e1f;color:#ff8fb1;font-size:0.78rem;border:1px solid #4a1a32;}
.card {background:linear-gradient(180deg,#120a10 0%,#1a0d17 100%);border:1px solid #2d1628;border-radius:16px;padding:1rem 1rem 0.9rem 1rem;}
.card-label {color:#c9a0b4;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;}
.card-value {color:#ffeef4;font-size:1.85rem;font-weight:800;line-height:1;}
.card-note {color:#9a708a;font-size:0.8rem;margin-top:0.4rem;}
.section-title {font-size:1.15rem;font-weight:700;color:#ffeef4;margin:0.2rem 0 0.85rem 0;}
.insight {background:#1a0d17;border:1px solid #2d1628;border-left:4px solid #ff4d6d;padding:0.85rem 1rem;border-radius:12px;color:#f0c4d4;margin:0.4rem 0 1rem 0;font-size:0.88rem;line-height:1.6;}
.offer {background:#0f1a0a;border:1px solid #1a3a12;border-left:4px solid #22c55e;padding:0.85rem 1rem;border-radius:12px;margin-bottom:0.65rem;color:#d4f0c4;font-size:0.88rem;line-height:1.6;}
.warn {background:#1a150a;border:1px solid #3a2a12;border-left:4px solid #f59e0b;padding:0.85rem 1rem;border-radius:12px;margin-bottom:0.65rem;color:#f0dfc4;font-size:0.88rem;line-height:1.6;}
.small-muted {color:#9a708a;font-size:0.82rem;}
div[data-testid="stMetric"] {background:linear-gradient(180deg,#120a10 0%,#1a0d17 100%);border:1px solid #2d1628;padding:0.85rem 1rem;border-radius:16px;}
div[data-testid="stMetricLabel"] {color:#c9a0b4;}
div[data-testid="stMetricValue"] {color:#ffeef4;}
div[data-testid="stDataFrame"] {border-radius:12px;overflow:hidden;}
div[data-baseweb="select"] > div {background:#1a0d17;border-color:#2d1628;}
div[data-testid="stTabs"] button {color:#c9a0b4 !important;border-bottom:2px solid transparent !important;}
div[data-testid="stTabs"] button[aria-selected="true"] {color:#ff8fb1 !important;border-bottom:2px solid #ff4d6d !important;}
</style>
""", unsafe_allow_html=True)

# ── ML CORE ───────────────────────────────────────────────────────────────────
def sigmoid(x):
    x = np.clip(x,-30,30); return 1.0/(1.0+np.exp(-x))

def stratified_split(X,y,test_size=0.25,random_state=42):
    rng=np.random.default_rng(random_state)
    idx0=np.where(y==0)[0]; idx1=np.where(y==1)[0]
    rng.shuffle(idx0); rng.shuffle(idx1)
    n0=max(1,int(len(idx0)*test_size)); n1=max(1,int(len(idx1)*test_size))
    test_idx=np.concatenate([idx0[:n0],idx1[:n1]]); train_idx=np.concatenate([idx0[n0:],idx1[n1:]])
    rng.shuffle(test_idx); rng.shuffle(train_idx)
    return X[train_idx],X[test_idx],y[train_idx],y[test_idx],train_idx,test_idx

def _acc(yt,yp): return float(np.mean(yt==yp))
def _prec(yt,yp):
    tp=np.sum((yt==1)&(yp==1)); fp=np.sum((yt==0)&(yp==1))
    return float(tp/(tp+fp)) if (tp+fp) else 0.0
def _rec(yt,yp):
    tp=np.sum((yt==1)&(yp==1)); fn=np.sum((yt==1)&(yp==0))
    return float(tp/(tp+fn)) if (tp+fn) else 0.0
def _f1(yt,yp):
    p=_prec(yt,yp); r=_rec(yt,yp)
    return float(2*p*r/(p+r)) if (p+r) else 0.0
def _auc(yt,ys):
    yt=np.asarray(yt); ys=np.asarray(ys)
    pos=np.sum(yt==1); neg=np.sum(yt==0)
    if pos==0 or neg==0: return 0.5
    order=np.argsort(ys); ranks=np.empty_like(order,dtype=float)
    ranks[order]=np.arange(1,len(ys)+1)
    return float((ranks[yt==1].sum()-pos*(pos+1)/2)/(pos*neg))
def _conf(yt,yp):
    return np.array([[int(np.sum((yt==0)&(yp==0))),int(np.sum((yt==0)&(yp==1)))],
                     [int(np.sum((yt==1)&(yp==0))),int(np.sum((yt==1)&(yp==1)))]])
def _roc(yt,ys):
    th=np.unique(np.round(ys,6))[::-1]; th=np.concatenate([[1.01],th,[-0.01]]); rows=[]
    for t in th:
        pred=(ys>=t).astype(int); cm=_conf(yt,pred); tn,fp,fn,tp=cm.ravel()
        rows.append((fp/(fp+tn) if (fp+tn) else 0, tp/(tp+fn) if (tp+fn) else 0))
    return pd.DataFrame(rows,columns=["FPR","TPR"]).drop_duplicates()
def _gini(y):
    if len(y)==0: return 0.0
    p=np.mean(y); return 1.0-p*p-(1-p)*(1-p)
def _mse(y):
    if len(y)==0: return 0.0
    m=np.mean(y); return float(np.mean((y-m)**2))
def _cth(col):
    v=np.unique(col)
    if len(v)<=12: return (v[:-1]+v[1:])/2
    return np.unique(np.quantile(col,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))

class Node:
    def __init__(self,value=None,prob=None,feature=None,threshold=None,left=None,right=None):
        self.value=value;self.prob=prob;self.feature=feature;self.threshold=threshold;self.left=left;self.right=right

class DTC:
    def __init__(self,max_depth=4,min_leaf=15,max_features=None,rs=42):
        self.max_depth=max_depth;self.min_leaf=min_leaf;self.max_features=max_features;self.rs=rs
    def fit(self,X,y):
        self.nf=X.shape[1];self.fi=np.zeros(self.nf);self.rng=np.random.default_rng(self.rs)
        self.root=self._b(X,y,0);t=self.fi.sum()
        if t>0: self.fi=self.fi/t
        return self
    def _pick(self):
        idx=np.arange(self.nf)
        if self.max_features is None or self.max_features>=self.nf: return idx
        return self.rng.choice(idx,size=self.max_features,replace=False)
    def _bs(self,X,y):
        par=_gini(y);bg,bf,bt=0.0,None,None
        for f in self._pick():
            col=X[:,f]
            for t in _cth(col):
                l=col<=t;r=~l
                if l.sum()<self.min_leaf or r.sum()<self.min_leaf: continue
                g=par-(l.mean()*_gini(y[l])+r.mean()*_gini(y[r]))
                if g>bg: bg,bf,bt=g,f,float(t)
        return bg,bf,bt
    def _b(self,X,y,depth):
        prob=float(np.mean(y)) if len(y) else 0.0;pred=int(prob>=0.5)
        if depth>=self.max_depth or len(y)<self.min_leaf*2 or len(np.unique(y))==1: return Node(value=pred,prob=prob)
        g,f,t=self._bs(X,y)
        if f is None or g<=1e-10: return Node(value=pred,prob=prob)
        self.fi[f]+=g*len(y);mask=X[:,f]<=t
        return Node(value=pred,prob=prob,feature=f,threshold=t,left=self._b(X[mask],y[mask],depth+1),right=self._b(X[~mask],y[~mask],depth+1))
    def _pp(self,row,node):
        while node.feature is not None: node=node.left if row[node.feature]<=node.threshold else node.right
        return node.prob
    def predict_proba(self,X):
        p=np.array([self._pp(r,self.root) for r in X]);return np.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)

class RTC:
    def __init__(self,max_depth=2,min_leaf=15,rs=42):
        self.max_depth=max_depth;self.min_leaf=min_leaf;self.rs=rs
    def fit(self,X,y):
        self.nf=X.shape[1];self.fi=np.zeros(self.nf);self.root=self._b(X,y,0)
        t=self.fi.sum()
        if t>0: self.fi=self.fi/t
        return self
    def _bs(self,X,y):
        par=_mse(y);bg,bf,bt=0.0,None,None
        for f in range(self.nf):
            col=X[:,f]
            for t in _cth(col):
                l=col<=t;r=~l
                if l.sum()<self.min_leaf or r.sum()<self.min_leaf: continue
                g=par-(l.mean()*_mse(y[l])+r.mean()*_mse(y[r]))
                if g>bg: bg,bf,bt=g,f,float(t)
        return bg,bf,bt
    def _b(self,X,y,depth):
        val=float(np.mean(y)) if len(y) else 0.0
        if depth>=self.max_depth or len(y)<self.min_leaf*2: return Node(value=val)
        g,f,t=self._bs(X,y)
        if f is None or g<=1e-10: return Node(value=val)
        self.fi[f]+=g*len(y);mask=X[:,f]<=t
        return Node(value=val,feature=f,threshold=t,left=self._b(X[mask],y[mask],depth+1),right=self._b(X[~mask],y[~mask],depth+1))
    def _po(self,row,node):
        while node.feature is not None: node=node.left if row[node.feature]<=node.threshold else node.right
        return node.value
    def predict(self,X): return np.array([self._po(r,self.root) for r in X])

class RFC:
    def __init__(self,n=24,max_depth=5,min_leaf=15,rs=42):
        self.n=n;self.max_depth=max_depth;self.min_leaf=min_leaf;self.rs=rs
    def fit(self,X,y):
        rng=np.random.default_rng(self.rs);n,m=X.shape;self.trees=[];imps=np.zeros(m);mf=max(1,int(np.sqrt(m)))
        for i in range(self.n):
            idx=rng.choice(np.arange(n),size=n,replace=True)
            t=DTC(max_depth=self.max_depth,min_leaf=self.min_leaf,max_features=mf,rs=self.rs+i+1)
            t.fit(X[idx],y[idx]);self.trees.append(t);imps+=t.fi
        self.fi=imps/self.n;return self
    def predict_proba(self,X):
        p=np.mean([t.predict_proba(X)[:,1] for t in self.trees],axis=0);return np.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)

class GBC:
    def __init__(self,n=35,lr=0.08,max_depth=2,min_leaf=15,rs=42):
        self.n=n;self.lr=lr;self.max_depth=max_depth;self.min_leaf=min_leaf;self.rs=rs
    def fit(self,X,y):
        p=np.clip(np.mean(y),1e-5,1-1e-5);self.bs=math.log(p/(1-p));F=np.full(len(y),self.bs)
        self.trees=[];imps=np.zeros(X.shape[1])
        for i in range(self.n):
            res=y-sigmoid(F);t=RTC(max_depth=self.max_depth,min_leaf=self.min_leaf,rs=self.rs+i+1)
            t.fit(X,res);F+=self.lr*t.predict(X);self.trees.append(t);imps+=t.fi
        s=imps.sum();self.fi=imps/s if s>0 else imps;return self
    def predict_proba(self,X):
        F=np.full(X.shape[0],self.bs)
        for t in self.trees: F+=self.lr*t.predict(X)
        p=sigmoid(F);return np.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)

# ── DATA ──────────────────────────────────────────────────────────────────────
def find_csv():
    for name in ["shein_apac_synthetic.csv"]:
        p=BASE_DIR/name
        if p.exists(): return p
    csvs=list(BASE_DIR.glob("*.csv")); return csvs[0] if csvs else None

@st.cache_data
def load_data(uploaded=None):
    if uploaded: df=pd.read_csv(uploaded)
    else:
        p=find_csv()
        if p is None: raise FileNotFoundError("shein_apac_synthetic.csv not found beside app.py")
        df=pd.read_csv(p)
    df.columns=[c.strip() for c in df.columns]; return df

FEATURE_COLS = ['recency_days','frequency_30d','frequency_90d','monetary_3m_usd',
                'avg_order_value_usd','basket_size_items','browsing_sessions_30d',
                'wishlist_items','discount_dependency_pct','promo_depth_pct',
                'promo_response_rate','returns_rate','mega_sale_participant',
                'avg_delivery_days','delivery_issues_count','last_mile_cost_usd',
                'fulfillment_cost_usd','return_cost_usd','logistics_cost_pct_revenue',
                'gross_margin_pct','responded_to_campaign','clv_12m_usd']

@st.cache_resource
def train_models(df):
    fc=[c for c in FEATURE_COLS if c in df.columns]
    X=df[fc].fillna(0).to_numpy(dtype=float); y=df[TARGET].to_numpy(dtype=int)
    Xtr,Xte,ytr,yte,_,_=stratified_split(X,y,test_size=0.25,random_state=42)
    models={"Decision Tree":DTC(max_depth=4,min_leaf=15,rs=42),
            "Random Forest":RFC(n=24,max_depth=5,min_leaf=15,rs=42),
            "Gradient Boosting":GBC(n=35,lr=0.08,max_depth=2,min_leaf=15,rs=42)}
    rows=[];fitted={}
    for name,model in models.items():
        model.fit(Xtr,ytr); pred=model.predict(Xte); proba=model.predict_proba(Xte)[:,1]
        rows.append({"Model":name,"Accuracy":round(_acc(yte,pred),4),"Precision":round(_prec(yte,pred),4),
                     "Recall":round(_rec(yte,pred),4),"F1":round(_f1(yte,pred),4),"ROC AUC":round(_auc(yte,proba),4)})
        fitted[name]={"model":model,"cm":_conf(yte,pred),"roc":_roc(yte,proba),"fi":model.fi}
    metrics=pd.DataFrame(rows).sort_values("ROC AUC",ascending=False).reset_index(drop=True)
    best=metrics.iloc[0]["Model"]; bm=models[best]
    full=df.copy(); probs=bm.predict_proba(X)[:,1]
    full["Predicted Churn Prob"]=probs.round(3); full["Predicted Churn"]=(probs>=0.5).astype(int)
    return fc,metrics,fitted,best,bm,full

def seg_action(seg):
    return {"Persuadable":"Send targeted voucher + early access style drop — high-ROI segment. Focus on top 20% by CLV.",
            "Sure-Thing":"Minimal intervention. Protect with loyalty tier upgrade or VIP early access.",
            "Do-Not-Disturb":"Pause deep promos — risk of discount-conditioning. Use brand storytelling instead.",
            "Lost Cause":"Deprioritise spend. Test lightweight re-engagement email; cut campaign losses."}.get(seg,"Monitor.")

def churn_action(row):
    actions=[]
    if row.get("recency_days",0)>60: actions.append("Re-engagement campaign — 60+ days inactive")
    if row.get("discount_dependency_pct",0)>0.65: actions.append("Shift to value messaging, reduce promo depth")
    if row.get("delivery_issues_count",0)>1: actions.append("Resolve delivery SLA or send apology voucher")
    if row.get("returns_rate",0)>0.35: actions.append("Deploy in-app fit assistant to reduce returns")
    if row.get("loyalty_tier","")=="None": actions.append("Enrol in Bronze tier to build switching cost")
    if not actions: actions.append("Low risk — monitor quarterly")
    return " | ".join(actions[:2])

# ── APP ───────────────────────────────────────────────────────────────────────
uploaded=st.sidebar.file_uploader("Upload shein_apac_synthetic.csv",type=["csv"])
df=load_data(uploaded)
fc,metrics,fitted,best_name,best_model,scored=train_models(df)

st.sidebar.markdown("### Filters")
country_sel=st.sidebar.multiselect("Country",sorted(df["country"].dropna().unique()),default=sorted(df["country"].dropna().unique()))
age_sel=st.sidebar.multiselect("Age Group",sorted(df["age_group"].dropna().unique()),default=sorted(df["age_group"].dropna().unique()))
channel_sel=st.sidebar.multiselect("Acquisition Channel",sorted(df["acquisition_channel"].dropna().unique()),default=sorted(df["acquisition_channel"].dropna().unique()))
tier_sel=st.sidebar.multiselect("Loyalty Tier",sorted(df["loyalty_tier"].dropna().unique()),default=sorted(df["loyalty_tier"].dropna().unique()))
churn_sel=st.sidebar.multiselect("Churn Status",["Churned","Retained"],default=["Churned","Retained"])
churn_vals=[{"Churned":1,"Retained":0}[c] for c in churn_sel]

view=df[df["country"].isin(country_sel)&df["age_group"].isin(age_sel)&df["acquisition_channel"].isin(channel_sel)&df["loyalty_tier"].isin(tier_sel)&df[TARGET].isin(churn_vals)].copy()

total=len(view)
churn_rate=view[TARGET].mean()*100 if total else 0
avg_clv=view["clv_12m_usd"].mean() if total else 0
avg_margin=view["gross_margin_pct"].mean()*100 if total else 0
persuadable=int((view["uplift_segment"]=="Persuadable").sum())
avg_disc=view["discount_dependency_pct"].mean()*100 if total else 0

st.markdown(f"""
<div class="main-shell">
  <div class="hero-title">\U0001f6cd\ufe0f Shein APAC Intelligence Suite</div>
  <p class="hero-sub">Retention & Churn &nbsp;·&nbsp; Logistics & Margin &nbsp;·&nbsp; Sustainable Growth &nbsp;·&nbsp; APAC Market Analytics</p>
  <div class="tag-row">
    <div class="tag">PS1 · Retention & Churn</div>
    <div class="tag">PS2 · Logistics & Margin</div>
    <div class="tag">PS3 · Sustainable Growth</div>
    <div class="tag">{total} customers · {len(df["country"].unique())} markets</div>
  </div>
</div>
""",unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6=st.columns(6,gap="small")
for col,label,val,note in [
    (c1,"Total Customers",f"{total:,}","Filtered audience"),
    (c2,"Churn Rate",f"{churn_rate:.1f}%","90-day inactivity label"),
    (c3,"Avg 12M CLV",f"${avg_clv:.0f}","Predicted lifetime value"),
    (c4,"Avg Gross Margin",f"{avg_margin:.1f}%","After logistics & promos"),
    (c5,"Persuadable Leads",f"{persuadable}","High-ROI campaign targets"),
    (c6,"Discount Dependency",f"{avg_disc:.1f}%","Promo-reliant behaviour"),
]:
    col.markdown(f"<div class='card'><div class='card-label'>{label}</div><div class='card-value'>{val}</div><div class='card-note'>{note}</div></div>",unsafe_allow_html=True)

st.markdown("<div style='height:0.55rem'></div>",unsafe_allow_html=True)

t1,t2,t3,t4=st.tabs(["Descriptive","Diagnostic","Predictive","Prescriptive"])

# ════ DESCRIPTIVE ════════════════════════════════════════════════════════════
with t1:
    a,b=st.columns(2,gap="large")
    with a:
        st.markdown("<div class='section-title'>Customer Count by Country</div>",unsafe_allow_html=True)
        cdf=view["country"].value_counts().to_frame("Customers")
        st.bar_chart(cdf)
        st.dataframe(cdf.reset_index().rename(columns={"index":"Country"}),use_container_width=True,hide_index=True)
    with b:
        st.markdown("<div class='section-title'>Churn Rate by Country</div>",unsafe_allow_html=True)
        crc=view.groupby("country")[TARGET].mean().mul(100).round(1).to_frame("Churn Rate %").sort_values("Churn Rate %",ascending=False)
        st.bar_chart(crc)
        st.dataframe(crc.reset_index(),use_container_width=True,hide_index=True)

    a,b,c=st.columns(3,gap="large")
    with a:
        st.markdown("<div class='section-title'>Age Group Split</div>",unsafe_allow_html=True)
        ag=view["age_group"].value_counts().sort_index().to_frame("Customers")
        st.bar_chart(ag)
        st.dataframe(ag.reset_index().rename(columns={"index":"Age Group"}),use_container_width=True,hide_index=True)
    with b:
        st.markdown("<div class='section-title'>Acquisition Channel Mix</div>",unsafe_allow_html=True)
        ach=view["acquisition_channel"].value_counts().to_frame("Customers")
        st.bar_chart(ach)
        st.dataframe(ach.reset_index().rename(columns={"index":"Channel"}),use_container_width=True,hide_index=True)
    with c:
        st.markdown("<div class='section-title'>Style Category Breakdown</div>",unsafe_allow_html=True)
        sc=view["style_category"].value_counts().to_frame("Customers")
        st.bar_chart(sc)
        st.dataframe(sc.reset_index().rename(columns={"index":"Style"}),use_container_width=True,hide_index=True)

    a,b,c=st.columns(3,gap="large")
    with a:
        st.markdown("<div class='section-title'>Loyalty Tier Distribution</div>",unsafe_allow_html=True)
        lt=view["loyalty_tier"].value_counts().to_frame("Customers")
        st.bar_chart(lt)
        st.dataframe(lt.reset_index().rename(columns={"index":"Tier"}),use_container_width=True,hide_index=True)
    with b:
        st.markdown("<div class='section-title'>Warehouse Hub Split</div>",unsafe_allow_html=True)
        wh=view["warehouse_hub"].value_counts().to_frame("Customers")
        st.bar_chart(wh)
        st.dataframe(wh.reset_index().rename(columns={"index":"Hub"}),use_container_width=True,hide_index=True)
    with c:
        st.markdown("<div class='section-title'>Mega-Sale Participation</div>",unsafe_allow_html=True)
        ms=view["mega_sale_participant"].map({1:"Participated",0:"Did Not"}).value_counts().to_frame("Customers")
        st.bar_chart(ms)
        st.dataframe(ms.reset_index().rename(columns={"index":"Status"}),use_container_width=True,hide_index=True)

    st.markdown("<div class='section-title'>Cohort Acquisition & Churn Timeline</div>",unsafe_allow_html=True)
    coh=view.groupby("cohort_month").agg(Customers=("customer_id","count"),Churned=(TARGET,"sum")).reset_index()
    coh["Churn Rate %"]=(coh["Churned"]/coh["Customers"]*100).round(1)
    st.line_chart(coh.set_index("cohort_month")[["Customers","Churned"]])
    st.dataframe(coh,use_container_width=True,hide_index=True)

# ════ DIAGNOSTIC ═════════════════════════════════════════════════════════════
with t2:
    a,b=st.columns(2,gap="large")
    with a:
        st.markdown("<div class='section-title'>Feature Correlation with Churn</div>",unsafe_allow_html=True)
        corr_rows=[]
        for col in fc:
            if col in view.columns:
                c_val=view[[col,TARGET]].dropna().corr().iloc[0,1]
                corr_rows.append({"Feature":col,"Correlation":round(c_val,3)})
        corr_df=pd.DataFrame(corr_rows).sort_values("Correlation",ascending=False)
        st.bar_chart(corr_df.set_index("Feature")[["Correlation"]])
        st.dataframe(corr_df,use_container_width=True,hide_index=True)
    with b:
        st.markdown("<div class='section-title'>Discount Dependency by Acquisition Channel</div>",unsafe_allow_html=True)
        dd_ch=view.groupby("acquisition_channel")["discount_dependency_pct"].mean().mul(100).round(1).to_frame("Avg Disc Dep %").sort_values("Avg Disc Dep %",ascending=False)
        st.bar_chart(dd_ch)
        st.dataframe(dd_ch.reset_index(),use_container_width=True,hide_index=True)

    a,b=st.columns(2,gap="large")
    with a:
        st.markdown("<div class='section-title'>Gross Margin by Country</div>",unsafe_allow_html=True)
        gm=view.groupby("country")["gross_margin_pct"].mean().mul(100).round(2).to_frame("Avg Gross Margin %").sort_values("Avg Gross Margin %",ascending=False)
        st.bar_chart(gm)
        st.dataframe(gm.reset_index(),use_container_width=True,hide_index=True)
    with b:
        st.markdown("<div class='section-title'>Logistics Cost % by Warehouse Hub</div>",unsafe_allow_html=True)
        lc=view.groupby("warehouse_hub")["logistics_cost_pct_revenue"].mean().mul(100).round(2).to_frame("Avg Logistics Cost %").sort_values("Avg Logistics Cost %",ascending=False)
        st.bar_chart(lc)
        st.dataframe(lc.reset_index(),use_container_width=True,hide_index=True)

    a,b=st.columns(2,gap="large")
    with a:
        st.markdown("<div class='section-title'>Churn Rate by Loyalty Tier</div>",unsafe_allow_html=True)
        ct=view.groupby("loyalty_tier")[TARGET].mean().mul(100).round(1).to_frame("Churn Rate %").sort_values("Churn Rate %",ascending=False)
        st.bar_chart(ct)
        st.dataframe(ct.reset_index(),use_container_width=True,hide_index=True)
    with b:
        st.markdown("<div class='section-title'>Returns Rate by Style Category</div>",unsafe_allow_html=True)
        rr=view.groupby("style_category")["returns_rate"].mean().mul(100).round(2).to_frame("Avg Returns Rate %").sort_values("Avg Returns Rate %",ascending=False)
        st.bar_chart(rr)
        st.dataframe(rr.reset_index(),use_container_width=True,hide_index=True)

    st.markdown("<div class='section-title'>RFM Segment Summary by Loyalty Tier</div>",unsafe_allow_html=True)
    a,b=st.columns(2,gap="large")
    with a:
        rfm=view.groupby("loyalty_tier").agg(Avg_Recency=("recency_days","mean"),Avg_Freq_90d=("frequency_90d","mean"),Avg_Monetary_3M=("monetary_3m_usd","mean"),Avg_CLV=("clv_12m_usd","mean")).round(2)
        st.dataframe(rfm.reset_index(),use_container_width=True,hide_index=True)
    with b:
        rfm2=view.groupby("country").agg(Avg_Recency=("recency_days","mean"),Avg_Freq=("frequency_90d","mean"),Avg_Monetary=("monetary_3m_usd","mean"),Avg_CLV=("clv_12m_usd","mean")).round(2)
        st.dataframe(rfm2.reset_index(),use_container_width=True,hide_index=True)

    st.markdown("<div class='section-title'>Delivery Issues & Churn by Country</div>",unsafe_allow_html=True)
    di=view.groupby("country").agg(Avg_Delivery_Days=("avg_delivery_days","mean"),Avg_Issues=("delivery_issues_count","mean"),Churn_Rate=(TARGET,"mean")).round(2)
    di["Churn Rate %"]=(di["Churn_Rate"]*100).round(1)
    st.dataframe(di.drop(columns="Churn_Rate").reset_index(),use_container_width=True,hide_index=True)

    infl=view[view["acquisition_channel"]=="Influencer"]["discount_dependency_pct"].mean()*100
    none_churn=view[view["loyalty_tier"]=="None"][TARGET].mean()*100
    st.markdown(f"""<div class='insight'><b>Key Diagnostic Findings:</b><br>
    · Influencer-acquired customers show the highest discount dependency ({infl:.1f}% avg), directly compressing gross margins.<br>
    · Customers with no loyalty tier churn at {none_churn:.1f}% — loyalty enrolment is the single highest-leverage retention lever.<br>
    · Philippines and Australia hubs show the highest avg delivery days, correlating with elevated churn risk in those markets.
    </div>""",unsafe_allow_html=True)

# ════ PREDICTIVE ══════════════════════════════════════════════════════════════
with t3:
    a,b=st.columns([1.1,0.9],gap="large")
    with a:
        st.markdown("<div class='section-title'>Churn Prediction Model Scorecard</div>",unsafe_allow_html=True)
        st.dataframe(metrics,use_container_width=True,hide_index=True)
        st.markdown(f"<div class='insight'><b>{best_name}</b> leads on ROC AUC ({metrics.iloc[0]['ROC AUC']:.4f}). Use this model to score all active APAC customers for churn risk each week.</div>",unsafe_allow_html=True)
    with b:
        st.markdown("<div class='section-title'>Predictive Confusion Matrix</div>",unsafe_allow_html=True)
        cm=pd.DataFrame(fitted[best_name]["cm"],index=["Actual: Retained","Actual: Churned"],columns=["Pred: Retained","Pred: Churned"])
        st.dataframe(cm,use_container_width=True)
        st.markdown("<div class='small-muted'>Rows = actual churn · Columns = predicted churn.</div>",unsafe_allow_html=True)

    a,b=st.columns(2,gap="large")
    with a:
        st.markdown("<div class='section-title'>Predictive Feature Importance</div>",unsafe_allow_html=True)
        fi=pd.Series(best_model.fi,index=fc).sort_values(ascending=False).round(4)
        st.bar_chart(fi.head(12))
        fi_df=fi.head(12).reset_index();fi_df.columns=["Feature","Importance"]
        st.dataframe(fi_df,use_container_width=True,hide_index=True)
    with b:
        st.markdown("<div class='section-title'>Predictive ROC View</div>",unsafe_allow_html=True)
        st.line_chart(fitted[best_name]["roc"].set_index("FPR"))
        st.markdown("<div class='small-muted'>Upper-left = stronger discrimination between churned vs retained.</div>",unsafe_allow_html=True)

    a,b=st.columns(2,gap="large")
    with a:
        st.markdown("<div class='section-title'>Avg CLV by Uplift Segment</div>",unsafe_allow_html=True)
        clv_s=view.groupby("uplift_segment")["clv_12m_usd"].mean().round(2).to_frame("Avg 12M CLV ($)").sort_values("Avg 12M CLV ($)",ascending=False)
        st.bar_chart(clv_s)
        st.dataframe(clv_s.reset_index(),use_container_width=True,hide_index=True)
    with b:
        st.markdown("<div class='section-title'>Churn Rate by Uplift Segment</div>",unsafe_allow_html=True)
        ch_s=view.groupby("uplift_segment")[TARGET].mean().mul(100).round(1).to_frame("Churn Rate %").sort_values("Churn Rate %",ascending=False)
        st.bar_chart(ch_s)
        st.dataframe(ch_s.reset_index(),use_container_width=True,hide_index=True)

    st.markdown("<div class='section-title'>Churn Risk Score Distribution</div>",unsafe_allow_html=True)
    rb=pd.cut(view["churn_risk_score"],bins=[0,25,50,75,100],labels=["Low (0–25)","Medium (25–50)","High (50–75)","Critical (75–100)"],include_lowest=True)
    rb_df=rb.value_counts().sort_index().to_frame("Customers")
    st.bar_chart(rb_df)
    st.dataframe(rb_df.reset_index().rename(columns={"index":"Risk Band"}),use_container_width=True,hide_index=True)

# ════ PRESCRIPTIVE ════════════════════════════════════════════════════════════
with t4:
    a,b=st.columns(2,gap="large")
    with a:
        st.markdown("<div class='section-title'>Uplift Segment Distribution</div>",unsafe_allow_html=True)
        us=view["uplift_segment"].value_counts().to_frame("Customers")
        st.bar_chart(us)
        st.dataframe(us.reset_index().rename(columns={"index":"Segment"}),use_container_width=True,hide_index=True)
    with b:
        st.markdown("<div class='section-title'>Campaign ROI by Type</div>",unsafe_allow_html=True)
        camp=view[view["campaign_last_sent"]!="None"].groupby("campaign_last_sent").agg(
            Avg_ROI=("campaign_roi","mean"),Response_Rate=("responded_to_campaign","mean"),
            Avg_Incremental_Orders=("incremental_orders_from_campaign","mean")).round(3)
        camp["Response Rate %"]=(camp["Response_Rate"]*100).round(1)
        st.bar_chart(camp[["Avg_ROI"]])
        st.dataframe(camp.drop(columns="Response_Rate").reset_index(),use_container_width=True,hide_index=True)

    a,b=st.columns(2,gap="large")
    with a:
        st.markdown("<div class='section-title'>Persuadable Lead Board (Top CLV)</div>",unsafe_allow_html=True)
        leads=scored[scored["uplift_segment"]=="Persuadable"].sort_values("clv_12m_usd",ascending=False)
        st.dataframe(leads[["customer_id","country","loyalty_tier","clv_12m_usd","churn_risk_score","discount_dependency_pct"]].head(15),use_container_width=True,hide_index=True)
    with b:
        st.markdown("<div class='section-title'>At-Risk High-CLV Customers</div>",unsafe_allow_html=True)
        at_risk=scored[(scored["Predicted Churn"]==1)&(scored["clv_12m_usd"]>scored["clv_12m_usd"].quantile(0.6))].sort_values("clv_12m_usd",ascending=False)
        st.dataframe(at_risk[["customer_id","country","clv_12m_usd","churn_risk_score","recency_days","loyalty_tier"]].head(15),use_container_width=True,hide_index=True)

    st.markdown("<div class='section-title'>Segment-Level Prescriptive Recommendations</div>",unsafe_allow_html=True)
    for seg,cls,title in [
        ("Persuadable","offer","\U0001f3af Persuadable — Activate Now"),
        ("Sure-Thing","offer","\u2705 Sure-Thing — Protect & Upgrade"),
        ("Do-Not-Disturb","warn","\u26a0\ufe0f Do-Not-Disturb — Reduce Promo Spend"),
        ("Lost Cause","warn","\U0001f534 Lost Cause — Minimal Spend, Test Re-engagement"),
    ]:
        sd=view[view["uplift_segment"]==seg]
        if len(sd)==0: continue
        st.markdown(f"""<div class='{cls}'><b>{title}</b> &nbsp;·&nbsp; {len(sd)} customers &nbsp;·&nbsp; Avg CLV: ${sd['clv_12m_usd'].mean():.0f} &nbsp;·&nbsp; Churn Rate: {sd[TARGET].mean()*100:.1f}%<br>{seg_action(seg)}</div>""",unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Prescriptive Simulator</div>",unsafe_allow_html=True)
    s1,s2,s3=st.columns(3,gap="large")
    with s1:
        sim_rec=st.slider("Recency (days since purchase)",1,180,45)
        sim_freq=st.slider("Frequency 90d (orders)",0,15,3)
        sim_mon=st.slider("Monetary 3M (USD)",0,600,100)
    with s2:
        sim_disc=st.slider("Discount Dependency",0.10,0.95,0.50)
        sim_ret=st.slider("Returns Rate",0.0,0.6,0.15)
        sim_di=st.slider("Delivery Issues",0,5,0)
    with s3:
        sim_clv=st.slider("Est. 12M CLV (USD)",10,850,200)
        sim_tier=st.selectbox("Loyalty Tier",["None","Bronze","Silver","Gold"])
        sim_del=st.slider("Avg Delivery Days",1.0,12.0,4.5)

    sim_vec=np.array([[sim_rec,sim_freq//3,sim_freq,sim_mon,
                       sim_mon/max(sim_freq,1),3,12,5,
                       sim_disc,sim_disc*0.55,0.6-sim_disc*0.25,
                       sim_ret,0,sim_del,sim_di,
                       3.5,2.5,sim_ret*20*0.4,
                       (3.5+2.5+sim_ret*20*0.4)/max(sim_mon/max(sim_freq,1),1),
                       0.18,0,sim_clv]])
    use_vec=sim_vec[:,:len(fc)]
    cp=float(best_model.predict_proba(use_vec)[0,1])
    rl="\U0001f534 HIGH CHURN RISK" if cp>0.65 else "\U0001f7e1 MEDIUM RISK" if cp>0.4 else "\U0001f7e2 LOW RISK"
    act=churn_action({"recency_days":sim_rec,"discount_dependency_pct":sim_disc,"delivery_issues_count":sim_di,"returns_rate":sim_ret,"loyalty_tier":sim_tier})
    st.markdown(f"""<div class='insight'><b>Simulated Churn Probability: {cp:.1%} — {rl}</b><br><b>Recommended Action:</b> {act}</div>""",unsafe_allow_html=True)

st.caption(f"Shein APAC Intelligence Suite · {total} customers · 8 markets · PS1 Churn · PS2 Logistics · PS3 Sustainable Growth")
