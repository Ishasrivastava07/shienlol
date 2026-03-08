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
.block-container{padding-top:1rem;padding-bottom:1.2rem;padding-left:1.4rem;padding-right:1.4rem;max-width:100%;}
section[data-testid="stSidebar"]>div{background:#0d0a0f;}
.main-shell{padding:1.1rem 1.35rem;border-radius:18px;background:linear-gradient(135deg,#120a10 0%,#1a0d17 100%);border:1px solid #2d1628;margin-bottom:1rem;}
.hero-title{font-size:2.1rem;font-weight:800;color:#ffeef4;line-height:1.05;margin:0 0 0.3rem 0;}
.hero-sub{font-size:0.95rem;color:#c9a0b4;margin:0;}
.tag-row{display:flex;gap:0.4rem;flex-wrap:wrap;margin-top:0.7rem;}
.tag{padding:0.28rem 0.6rem;border-radius:999px;background:#2a0e1f;color:#ff8fb1;font-size:0.78rem;border:1px solid #4a1a32;}
.ps1-tag{padding:0.28rem 0.6rem;border-radius:999px;background:#0e1f2a;color:#8fb1ff;font-size:0.78rem;border:1px solid #1a324a;}
.ps2-tag{padding:0.28rem 0.6rem;border-radius:999px;background:#0e2a14;color:#8fffa0;font-size:0.78rem;border:1px solid #1a4a24;}
.ps3-tag{padding:0.28rem 0.6rem;border-radius:999px;background:#2a1f0e;color:#ffd68f;font-size:0.78rem;border:1px solid #4a341a;}
.card{background:linear-gradient(180deg,#120a10 0%,#1a0d17 100%);border:1px solid #2d1628;border-radius:16px;padding:1rem 1rem 0.9rem 1rem;}
.card-label{color:#c9a0b4;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;}
.card-value{color:#ffeef4;font-size:1.85rem;font-weight:800;line-height:1;}
.card-note{color:#9a708a;font-size:0.8rem;margin-top:0.4rem;}
.section-title{font-size:1.15rem;font-weight:700;color:#ffeef4;margin:0.2rem 0 0.85rem 0;}
.ps-header{padding:0.6rem 1rem;border-radius:10px;margin-bottom:1rem;font-size:0.9rem;font-weight:600;}
.ps1-header{background:#0e1f2a;border-left:4px solid #4d9fff;color:#a8ccff;}
.ps2-header{background:#0e2a14;border-left:4px solid #22c55e;color:#a8ffc0;}
.ps3-header{background:#2a1f0e;border-left:4px solid #f59e0b;color:#ffd6a0;}
.insight{background:#1a0d17;border:1px solid #2d1628;border-left:4px solid #ff4d6d;padding:0.85rem 1rem;border-radius:12px;color:#f0c4d4;margin:0.4rem 0 1rem 0;font-size:0.87rem;line-height:1.6;}
.offer{background:#0f1a0a;border:1px solid #1a3a12;border-left:4px solid #22c55e;padding:0.85rem 1rem;border-radius:12px;margin-bottom:0.65rem;color:#d4f0c4;font-size:0.87rem;line-height:1.6;}
.warn{background:#1a150a;border:1px solid #3a2a12;border-left:4px solid #f59e0b;padding:0.85rem 1rem;border-radius:12px;margin-bottom:0.65rem;color:#f0dfc4;font-size:0.87rem;line-height:1.6;}
.small-muted{color:#9a708a;font-size:0.82rem;}
div[data-testid="stMetric"]{background:linear-gradient(180deg,#120a10 0%,#1a0d17 100%);border:1px solid #2d1628;padding:0.85rem 1rem;border-radius:16px;}
div[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden;}
div[data-baseweb="select"]>div{background:#1a0d17;border-color:#2d1628;}
div[data-testid="stTabs"] button{color:#c9a0b4 !important;border-bottom:2px solid transparent !important;}
div[data-testid="stTabs"] button[aria-selected="true"]{color:#ff8fb1 !important;border-bottom:2px solid #ff4d6d !important;}
</style>
""", unsafe_allow_html=True)

# ── ML CORE ───────────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0/(1.0+np.exp(-np.clip(x,-30,30)))
def stratified_split(X,y,test_size=0.25,rs=42):
    rng=np.random.default_rng(rs)
    idx0=np.where(y==0)[0];idx1=np.where(y==1)[0]
    rng.shuffle(idx0);rng.shuffle(idx1)
    n0=max(1,int(len(idx0)*test_size));n1=max(1,int(len(idx1)*test_size))
    ti=np.concatenate([idx0[:n0],idx1[:n1]]);ri=np.concatenate([idx0[n0:],idx1[n1:]])
    rng.shuffle(ti);rng.shuffle(ri)
    return X[ri],X[ti],y[ri],y[ti],ri,ti
def _acc(yt,yp): return float(np.mean(yt==yp))
def _prec(yt,yp):
    tp=np.sum((yt==1)&(yp==1));fp=np.sum((yt==0)&(yp==1))
    return float(tp/(tp+fp)) if (tp+fp) else 0.0
def _rec(yt,yp):
    tp=np.sum((yt==1)&(yp==1));fn=np.sum((yt==1)&(yp==0))
    return float(tp/(tp+fn)) if (tp+fn) else 0.0
def _f1(yt,yp):
    p=_prec(yt,yp);r=_rec(yt,yp)
    return float(2*p*r/(p+r)) if (p+r) else 0.0
def _auc(yt,ys):
    yt=np.asarray(yt);ys=np.asarray(ys)
    pos=np.sum(yt==1);neg=np.sum(yt==0)
    if pos==0 or neg==0: return 0.5
    order=np.argsort(ys);ranks=np.empty_like(order,dtype=float)
    ranks[order]=np.arange(1,len(ys)+1)
    return float((ranks[yt==1].sum()-pos*(pos+1)/2)/(pos*neg))
def _conf(yt,yp):
    return np.array([[int(np.sum((yt==0)&(yp==0))),int(np.sum((yt==0)&(yp==1)))],
                     [int(np.sum((yt==1)&(yp==0))),int(np.sum((yt==1)&(yp==1)))]])
def _roc(yt,ys):
    th=np.unique(np.round(ys,6))[::-1];th=np.concatenate([[1.01],th,[-0.01]]);rows=[]
    for t in th:
        pred=(ys>=t).astype(int);cm=_conf(yt,pred);tn,fp,fn,tp=cm.ravel()
        rows.append((fp/(fp+tn) if (fp+tn) else 0,tp/(tp+fn) if (tp+fn) else 0))
    return pd.DataFrame(rows,columns=["FPR","TPR"]).drop_duplicates()
def _gini(y):
    if len(y)==0: return 0.0
    p=np.mean(y);return 1.0-p*p-(1-p)*(1-p)
def _mse(y):
    if len(y)==0: return 0.0
    m=np.mean(y);return float(np.mean((y-m)**2))
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
        if t>0: self.fi/=t
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
        return Node(value=pred,prob=prob,feature=f,threshold=t,
                    left=self._b(X[mask],y[mask],depth+1),right=self._b(X[~mask],y[~mask],depth+1))
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
        if t>0: self.fi/=t
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
        return Node(value=val,feature=f,threshold=t,
                    left=self._b(X[mask],y[mask],depth+1),right=self._b(X[~mask],y[~mask],depth+1))
    def _po(self,row,node):
        while node.feature is not None: node=node.left if row[node.feature]<=node.threshold else node.right
        return node.value
    def predict(self,X): return np.array([self._po(r,self.root) for r in X])

class RFC:
    def __init__(self,n=24,max_depth=5,min_leaf=15,rs=42):
        self.n=n;self.max_depth=max_depth;self.min_leaf=min_leaf;self.rs=rs
    def fit(self,X,y):
        rng=np.random.default_rng(self.rs);nr,m=X.shape;self.trees=[];imps=np.zeros(m);mf=max(1,int(np.sqrt(m)))
        for i in range(self.n):
            idx=rng.choice(np.arange(nr),size=nr,replace=True)
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

# ── DATA & MODELS ─────────────────────────────────────────────────────────────
def find_csv():
    p=BASE_DIR/"shein_apac_synthetic.csv"
    if p.exists(): return p
    csvs=list(BASE_DIR.glob("*.csv"));return csvs[0] if csvs else None

@st.cache_data
def load_data(up=None):
    df=pd.read_csv(up) if up else pd.read_csv(find_csv())
    df.columns=[c.strip() for c in df.columns]
    for col in ["country","age_group","gender","style_category","acquisition_channel",
                "cohort_month","loyalty_tier","warehouse_hub","campaign_last_sent","uplift_segment"]:
        if col in df.columns:
            df[col]=df[col].fillna("Unknown").astype(str)
    return df

PS1_FEATURES=['recency_days','frequency_30d','frequency_90d','monetary_3m_usd',
              'avg_order_value_usd','browsing_sessions_30d','wishlist_items',
              'discount_dependency_pct','returns_rate','mega_sale_participant',
              'delivery_issues_count','responded_to_campaign','clv_12m_usd']
PS2_FEATURES=['avg_delivery_days','delivery_issues_count','last_mile_cost_usd',
              'fulfillment_cost_usd','return_cost_usd','logistics_cost_pct_revenue',
              'basket_size_items','avg_order_value_usd','returns_rate','monetary_3m_usd']
PS3_FEATURES=['discount_dependency_pct','promo_depth_pct','promo_response_rate',
              'mega_sale_participant','campaign_roi','incremental_orders_from_campaign',
              'clv_12m_usd','gross_margin_pct','responded_to_campaign',
              'frequency_90d','monetary_3m_usd']

@st.cache_resource
def train_all(df):
    def run(fc,tgt_col,tgt_arr,label):
        fc2=[c for c in fc if c in df.columns]
        X=df[fc2].fillna(0).to_numpy(dtype=float)
        y=tgt_arr.to_numpy(dtype=int)
        Xtr,Xte,ytr,yte,_,_=stratified_split(X,y,test_size=0.25,rs=42)
        models={"Decision Tree":DTC(max_depth=4,min_leaf=12,rs=42),
                "Random Forest":RFC(n=20,max_depth=5,min_leaf=12,rs=42),
                "Gradient Boosting":GBC(n=30,lr=0.08,max_depth=2,min_leaf=12,rs=42)}
        rows=[];fitted={}
        for nm,m in models.items():
            m.fit(Xtr,ytr);pred=m.predict(Xte);proba=m.predict_proba(Xte)[:,1]
            rows.append({"Model":nm,"Accuracy":round(_acc(yte,pred),4),"Precision":round(_prec(yte,pred),4),
                         "Recall":round(_rec(yte,pred),4),"F1":round(_f1(yte,pred),4),"ROC AUC":round(_auc(yte,proba),4)})
            fitted[nm]={"cm":_conf(yte,pred),"roc":_roc(yte,proba),"fi":m.fi,"model":m}
        met=pd.DataFrame(rows).sort_values("ROC AUC",ascending=False).reset_index(drop=True)
        best=met.iloc[0]["Model"];bm=models[best]
        probs=bm.predict_proba(X)[:,1]
        return fc2,met,fitted,best,bm,probs
    # PS1: churn prediction
    ps1_fc,ps1_met,ps1_fit,ps1_best,ps1_bm,ps1_probs=run(PS1_FEATURES,"churn_label",df["churn_label"],"PS1")
    # PS2: high logistics cost (logistics_cost_pct_revenue > median)
    med_log=df["logistics_cost_pct_revenue"].median()
    ps2_target=(df["logistics_cost_pct_revenue"]>med_log).astype(int)
    ps2_fc,ps2_met,ps2_fit,ps2_best,ps2_bm,ps2_probs=run(PS2_FEATURES,"high_logistics",ps2_target,"PS2")
    # PS3: responded to campaign (campaign conversion prediction)
    ps3_fc,ps3_met,ps3_fit,ps3_best,ps3_bm,ps3_probs=run(PS3_FEATURES,"campaign_response",df["responded_to_campaign"],"PS3")
    full=df.copy()
    full["Churn Prob"]=ps1_probs.round(3)
    full["High Logistics Risk"]=(ps2_probs>=0.5).astype(int)
    full["Campaign Convert Prob"]=ps3_probs.round(3)
    return (ps1_fc,ps1_met,ps1_fit,ps1_best,ps1_bm),(ps2_fc,ps2_met,ps2_fit,ps2_best,ps2_bm),(ps3_fc,ps3_met,ps3_fit,ps3_best,ps3_bm),full

def churn_rx(row):
    a=[]
    if row.get("recency_days",0)>60: a.append("Re-engagement campaign — inactive 60+ days")
    if row.get("discount_dependency_pct",0)>0.65: a.append("Shift to value messaging, reduce promo depth")
    if row.get("delivery_issues_count",0)>1: a.append("Resolve delivery SLA or send apology voucher")
    if row.get("returns_rate",0)>0.35: a.append("Deploy in-app fit assistant")
    if row.get("loyalty_tier","")=="None": a.append("Enrol in Bronze tier to build switching cost")
    return " | ".join(a[:2]) if a else "Low risk — monitor quarterly"

def seg_rx(seg):
    return {"Persuadable":"Targeted voucher + style drop early access — highest ROI per $ spent.",
            "Sure-Thing":"No heavy spend needed. Upgrade loyalty tier to Silver/Gold to lock in.",
            "Do-Not-Disturb":"Pause deep promos. Risk of discount-conditioning. Use brand storytelling.",
            "Lost Cause":"Deprioritise. Test lightweight reactivation email only."}.get(seg,"Monitor.")

# ── LOAD ──────────────────────────────────────────────────────────────────────
up=st.sidebar.file_uploader("Upload shein_apac_synthetic.csv",type=["csv"])
df=load_data(up)
ps1,ps2,ps3,scored=train_all(df)
ps1_fc,ps1_met,ps1_fit,ps1_best,ps1_bm=ps1
ps2_fc,ps2_met,ps2_fit,ps2_best,ps2_bm=ps2
ps3_fc,ps3_met,ps3_fit,ps3_best,ps3_bm=ps3

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────────
st.sidebar.markdown("### Filters")
country_sel=st.sidebar.multiselect("Country",sorted(df["country"].dropna().astype(str).unique()),default=sorted(df["country"].dropna().astype(str).unique()))
age_sel=st.sidebar.multiselect("Age Group",sorted(df["age_group"].dropna().astype(str).unique()),default=sorted(df["age_group"].dropna().astype(str).unique()))
channel_sel=st.sidebar.multiselect("Acquisition Channel",sorted(df["acquisition_channel"].dropna().astype(str).unique()),default=sorted(df["acquisition_channel"].dropna().astype(str).unique()))
tier_sel=st.sidebar.multiselect("Loyalty Tier",sorted(df["loyalty_tier"].dropna().astype(str).unique()),default=sorted(df["loyalty_tier"].dropna().astype(str).unique()))
churn_filter=st.sidebar.multiselect("Churn Status",["Churned","Retained"],default=["Churned","Retained"])
churn_vals=[{"Churned":1,"Retained":0}[c] for c in churn_filter]

view=df[df["country"].isin(country_sel)&df["age_group"].isin(age_sel)&df["acquisition_channel"].isin(channel_sel)&df["loyalty_tier"].isin(tier_sel)&df[TARGET].isin(churn_vals)].copy()
view_scored=scored[scored["customer_id"].isin(view["customer_id"])].copy()

total=len(view)
churn_rate=view[TARGET].mean()*100 if total else 0
avg_clv=view["clv_12m_usd"].mean() if total else 0
avg_margin=view["gross_margin_pct"].mean()*100 if total else 0
persuadable=int((view["uplift_segment"]=="Persuadable").sum())
avg_disc=view["discount_dependency_pct"].mean()*100 if total else 0
avg_logistics=view["logistics_cost_pct_revenue"].mean()*100 if total else 0

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-shell">
  <div class="hero-title">\U0001f6cd\ufe0f Shein APAC Intelligence Suite</div>
  <p class="hero-sub">Three APAC problem statements · Each tab broken down by PS1 · PS2 · PS3</p>
  <div class="tag-row">
    <div class="ps1-tag">PS1 · Retention & Churn</div>
    <div class="ps2-tag">PS2 · Logistics & Margin</div>
    <div class="ps3-tag">PS3 · Sustainable Growth</div>
    <div class="tag">{total} customers · {len(df["country"].unique())} markets</div>
  </div>
</div>""",unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6=st.columns(6,gap="small")
for col,label,val,note in [
    (c1,"Total Customers",f"{total:,}","Filtered audience"),
    (c2,"Churn Rate",f"{churn_rate:.1f}%","PS1 · 90-day inactivity"),
    (c3,"Avg 12M CLV",f"${avg_clv:.0f}","PS1 · Lifetime value"),
    (c4,"Avg Gross Margin",f"{avg_margin:.1f}%","PS2 · After costs & promos"),
    (c5,"Avg Logistics Cost",f"{avg_logistics:.1f}%","PS2 · % of revenue"),
    (c6,"Discount Dependency",f"{avg_disc:.1f}%","PS3 · Promo-reliant share"),
]:
    col.markdown(f"<div class='card'><div class='card-label'>{label}</div><div class='card-value'>{val}</div><div class='card-note'>{note}</div></div>",unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>",unsafe_allow_html=True)

# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab_desc,tab_diag,tab_pred,tab_presc=st.tabs(["Descriptive","Diagnostic","Predictive","Prescriptive"])

# ══════════════════════════════════════════════════════════════════════════════
# DESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_desc:
    d1,d2,d3=st.tabs(["PS1 · Retention & Churn","PS2 · Logistics & Margin","PS3 · Sustainable Growth"])

    with d1:
        st.markdown("<div class='ps-header ps1-header'>Who are the customers, and what does churn look like across APAC?</div>",unsafe_allow_html=True)
        a,b=st.columns(2,gap="large")
        with a:
            st.markdown("<div class='section-title'>Customer Count by Country</div>",unsafe_allow_html=True)
            cdf=view["country"].value_counts().to_frame("Customers")
            st.bar_chart(cdf)
            st.dataframe(cdf.reset_index().rename(columns={"index":"Country"}),use_container_width=True,hide_index=True)
        with b:
            st.markdown("<div class='section-title'>Churn Rate by Country</div>",unsafe_allow_html=True)
            crc=view.dropna(subset=["country"]).groupby("country")[TARGET].mean().mul(100).round(1).to_frame("Churn Rate %").sort_values("Churn Rate %",ascending=False)
            st.bar_chart(crc)
            st.dataframe(crc.reset_index(),use_container_width=True,hide_index=True)
        a,b,c=st.columns(3,gap="large")
        with a:
            st.markdown("<div class='section-title'>Loyalty Tier Distribution</div>",unsafe_allow_html=True)
            lt=view["loyalty_tier"].value_counts().to_frame("Customers")
            st.bar_chart(lt);st.dataframe(lt.reset_index().rename(columns={"index":"Tier"}),use_container_width=True,hide_index=True)
        with b:
            st.markdown("<div class='section-title'>Acquisition Channel Mix</div>",unsafe_allow_html=True)
            ach=view["acquisition_channel"].value_counts().to_frame("Customers")
            st.bar_chart(ach);st.dataframe(ach.reset_index().rename(columns={"index":"Channel"}),use_container_width=True,hide_index=True)
        with c:
            st.markdown("<div class='section-title'>Age Group Split</div>",unsafe_allow_html=True)
            ag=view["age_group"].value_counts().to_frame("Customers")
            st.bar_chart(ag);st.dataframe(ag.reset_index().rename(columns={"index":"Age Group"}),use_container_width=True,hide_index=True)
        st.markdown("<div class='section-title'>Cohort Acquisition & Churn Over Time</div>",unsafe_allow_html=True)
        coh=view.dropna(subset=["cohort_month"]).groupby("cohort_month").agg(Customers=("customer_id","count"),Churned=(TARGET,"sum")).reset_index()
        coh["Churn Rate %"]=(coh["Churned"]/coh["Customers"]*100).round(1)
        st.line_chart(coh.set_index("cohort_month")[["Customers","Churned"]])
        st.dataframe(coh,use_container_width=True,hide_index=True)

    with d2:
        st.markdown("<div class='ps-header ps2-header'>How is the logistics network structured, and what does cost look like across hubs?</div>",unsafe_allow_html=True)
        a,b=st.columns(2,gap="large")
        with a:
            st.markdown("<div class='section-title'>Customer Volume by Warehouse Hub</div>",unsafe_allow_html=True)
            wh=view["warehouse_hub"].value_counts().to_frame("Customers")
            st.bar_chart(wh);st.dataframe(wh.reset_index().rename(columns={"index":"Hub"}),use_container_width=True,hide_index=True)
        with b:
            st.markdown("<div class='section-title'>Avg Delivery Days by Country</div>",unsafe_allow_html=True)
            del_c=view.dropna(subset=["country"]).groupby("country")["avg_delivery_days"].mean().round(2).to_frame("Avg Delivery Days").sort_values("Avg Delivery Days",ascending=False)
            st.bar_chart(del_c);st.dataframe(del_c.reset_index(),use_container_width=True,hide_index=True)
        a,b,c=st.columns(3,gap="large")
        with a:
            st.markdown("<div class='section-title'>Avg Basket Size by Country</div>",unsafe_allow_html=True)
            bs=view.dropna(subset=["country"]).groupby("country")["basket_size_items"].mean().round(2).to_frame("Avg Basket Items").sort_values("Avg Basket Items",ascending=False)
            st.bar_chart(bs);st.dataframe(bs.reset_index(),use_container_width=True,hide_index=True)
        with b:
            st.markdown("<div class='section-title'>Returns Rate by Country</div>",unsafe_allow_html=True)
            rr=view.dropna(subset=["country"]).groupby("country")["returns_rate"].mean().mul(100).round(2).to_frame("Avg Returns Rate %").sort_values("Avg Returns Rate %",ascending=False)
            st.bar_chart(rr);st.dataframe(rr.reset_index(),use_container_width=True,hide_index=True)
        with c:
            st.markdown("<div class='section-title'>Delivery Issues Count by Hub</div>",unsafe_allow_html=True)
            di=view.dropna(subset=["warehouse_hub"]).groupby("warehouse_hub")["delivery_issues_count"].mean().round(2).to_frame("Avg Delivery Issues")
            st.bar_chart(di);st.dataframe(di.reset_index(),use_container_width=True,hide_index=True)
        st.markdown("<div class='section-title'>Full Logistics Cost Breakdown by Country</div>",unsafe_allow_html=True)
        lc=view.dropna(subset=["country"]).groupby("country").agg(Avg_Last_Mile=("last_mile_cost_usd","mean"),Avg_Fulfillment=("fulfillment_cost_usd","mean"),Avg_Return_Cost=("return_cost_usd","mean"),Avg_Logistics_Pct=("logistics_cost_pct_revenue","mean")).mul({"Avg_Last_Mile":1,"Avg_Fulfillment":1,"Avg_Return_Cost":1,"Avg_Logistics_Pct":100}).round(2)
        st.dataframe(lc.reset_index(),use_container_width=True,hide_index=True)

    with d3:
        st.markdown("<div class='ps-header ps3-header'>How dependent is growth on discounts, influencers, and mega-sale events?</div>",unsafe_allow_html=True)
        a,b=st.columns(2,gap="large")
        with a:
            st.markdown("<div class='section-title'>Avg Discount Dependency by Channel</div>",unsafe_allow_html=True)
            dd=view.dropna(subset=["acquisition_channel"]).groupby("acquisition_channel")["discount_dependency_pct"].mean().mul(100).round(1).to_frame("Disc Dep %").sort_values("Disc Dep %",ascending=False)
            st.bar_chart(dd);st.dataframe(dd.reset_index(),use_container_width=True,hide_index=True)
        with b:
            st.markdown("<div class='section-title'>Mega-Sale Participation by Country</div>",unsafe_allow_html=True)
            ms=view.dropna(subset=["country"]).groupby("country")["mega_sale_participant"].mean().mul(100).round(1).to_frame("Mega-Sale Participation %").sort_values("Mega-Sale Participation %",ascending=False)
            st.bar_chart(ms);st.dataframe(ms.reset_index(),use_container_width=True,hide_index=True)
        a,b,c=st.columns(3,gap="large")
        with a:
            st.markdown("<div class='section-title'>Avg Promo Depth by Country</div>",unsafe_allow_html=True)
            pd_c=view.dropna(subset=["country"]).groupby("country")["promo_depth_pct"].mean().mul(100).round(1).to_frame("Avg Promo Depth %").sort_values("Avg Promo Depth %",ascending=False)
            st.bar_chart(pd_c);st.dataframe(pd_c.reset_index(),use_container_width=True,hide_index=True)
        with b:
            st.markdown("<div class='section-title'>Campaign Type Mix</div>",unsafe_allow_html=True)
            camp=view["campaign_last_sent"].value_counts().to_frame("Count")
            st.bar_chart(camp);st.dataframe(camp.reset_index().rename(columns={"index":"Campaign"}),use_container_width=True,hide_index=True)
        with c:
            st.markdown("<div class='section-title'>Uplift Segment Distribution</div>",unsafe_allow_html=True)
            us=view["uplift_segment"].value_counts().to_frame("Customers")
            st.bar_chart(us);st.dataframe(us.reset_index().rename(columns={"index":"Segment"}),use_container_width=True,hide_index=True)
        st.markdown("<div class='section-title'>Avg 12M CLV by Acquisition Channel</div>",unsafe_allow_html=True)
        clv_ch=view.dropna(subset=["acquisition_channel"]).groupby("acquisition_channel")["clv_12m_usd"].mean().round(2).to_frame("Avg 12M CLV ($)").sort_values("Avg 12M CLV ($)",ascending=False)
        st.bar_chart(clv_ch);st.dataframe(clv_ch.reset_index(),use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════
with tab_diag:
    g1,g2,g3=st.tabs(["PS1 · Retention & Churn","PS2 · Logistics & Margin","PS3 · Sustainable Growth"])

    with g1:
        st.markdown("<div class='ps-header ps1-header'>What is driving churn across APAC customer segments?</div>",unsafe_allow_html=True)
        a,b=st.columns(2,gap="large")
        with a:
            st.markdown("<div class='section-title'>Feature Correlation with Churn</div>",unsafe_allow_html=True)
            cr=[]
            for col in PS1_FEATURES:
                if col in view.columns:
                    v=view[[col,TARGET]].dropna().corr().iloc[0,1]
                    cr.append({"Feature":col,"Correlation":round(v,3)})
            cr_df=pd.DataFrame(cr).sort_values("Correlation",ascending=False)
            st.bar_chart(cr_df.set_index("Feature")[["Correlation"]])
            st.dataframe(cr_df,use_container_width=True,hide_index=True)
        with b:
            st.markdown("<div class='section-title'>Churn Rate by Loyalty Tier</div>",unsafe_allow_html=True)
            ct=view.dropna(subset=["loyalty_tier"]).groupby("loyalty_tier")[TARGET].mean().mul(100).round(1).to_frame("Churn Rate %").sort_values("Churn Rate %",ascending=False)
            st.bar_chart(ct);st.dataframe(ct.reset_index(),use_container_width=True,hide_index=True)
        a,b,c=st.columns(3,gap="large")
        with a:
            st.markdown("<div class='section-title'>Churn Rate by Age Group</div>",unsafe_allow_html=True)
            ca=view.dropna(subset=["age_group"]).groupby("age_group")[TARGET].mean().mul(100).round(1).to_frame("Churn Rate %")
            st.bar_chart(ca);st.dataframe(ca.reset_index(),use_container_width=True,hide_index=True)
        with b:
            st.markdown("<div class='section-title'>Churn Rate by Acquisition Channel</div>",unsafe_allow_html=True)
            cc=view.dropna(subset=["acquisition_channel"]).groupby("acquisition_channel")[TARGET].mean().mul(100).round(1).to_frame("Churn Rate %").sort_values("Churn Rate %",ascending=False)
            st.bar_chart(cc);st.dataframe(cc.reset_index(),use_container_width=True,hide_index=True)
        with c:
            st.markdown("<div class='section-title'>Returns Rate by Style Category</div>",unsafe_allow_html=True)
            rsc=view.dropna(subset=["style_category"]).groupby("style_category")["returns_rate"].mean().mul(100).round(2).to_frame("Returns Rate %").sort_values("Returns Rate %",ascending=False)
            st.bar_chart(rsc);st.dataframe(rsc.reset_index(),use_container_width=True,hide_index=True)
        st.markdown("<div class='section-title'>RFM Summary by Loyalty Tier</div>",unsafe_allow_html=True)
        rfm=view.dropna(subset=["loyalty_tier"]).groupby("loyalty_tier").agg(Avg_Recency=("recency_days","mean"),Avg_Freq_90d=("frequency_90d","mean"),Avg_Monetary=("monetary_3m_usd","mean"),Avg_CLV=("clv_12m_usd","mean"),Churn_Rate=(TARGET,"mean")).round(2)
        rfm["Churn Rate %"]=(rfm["Churn_Rate"]*100).round(1)
        st.dataframe(rfm.drop(columns="Churn_Rate").reset_index(),use_container_width=True,hide_index=True)
        none_c=view[view["loyalty_tier"]=="None"][TARGET].mean()*100
        infl_c=view[view["acquisition_channel"]=="Influencer"][TARGET].mean()*100
        st.markdown(f"<div class='insight'><b>Key finding:</b> No-tier customers churn at {none_c:.1f}%. Influencer-acquired customers churn at {infl_c:.1f}%. Recency and discount dependency are the two strongest churn correlates — high recency + high discount dep = highest churn risk profile.</div>",unsafe_allow_html=True)

    with g2:
        st.markdown("<div class='ps-header ps2-header'>What is causing margin pressure across logistics and fulfilment?</div>",unsafe_allow_html=True)
        a,b=st.columns(2,gap="large")
        with a:
            st.markdown("<div class='section-title'>Gross Margin % by Country</div>",unsafe_allow_html=True)
            gm=view.dropna(subset=["country"]).groupby("country")["gross_margin_pct"].mean().mul(100).round(2).to_frame("Gross Margin %").sort_values("Gross Margin %",ascending=False)
            st.bar_chart(gm);st.dataframe(gm.reset_index(),use_container_width=True,hide_index=True)
        with b:
            st.markdown("<div class='section-title'>Logistics Cost % by Warehouse Hub</div>",unsafe_allow_html=True)
            lch=view.dropna(subset=["warehouse_hub"]).groupby("warehouse_hub")["logistics_cost_pct_revenue"].mean().mul(100).round(2).to_frame("Logistics Cost %").sort_values("Logistics Cost %",ascending=False)
            st.bar_chart(lch);st.dataframe(lch.reset_index(),use_container_width=True,hide_index=True)
        a,b=st.columns(2,gap="large")
        with a:
            st.markdown("<div class='section-title'>Last-Mile Cost vs Avg Order Value by Country</div>",unsafe_allow_html=True)
            lmv=view.dropna(subset=["country"]).groupby("country").agg(Avg_AOV=("avg_order_value_usd","mean"),Avg_Last_Mile=("last_mile_cost_usd","mean")).round(2)
            lmv["Last Mile as % AOV"]=(lmv["Avg_Last_Mile"]/lmv["Avg_AOV"]*100).round(1)
            st.dataframe(lmv.reset_index(),use_container_width=True,hide_index=True)
            st.bar_chart(lmv[["Last Mile as % AOV"]])
        with b:
            st.markdown("<div class='section-title'>Delivery Issues vs Churn Rate by Country</div>",unsafe_allow_html=True)
            di2=view.dropna(subset=["country"]).groupby("country").agg(Avg_Issues=("delivery_issues_count","mean"),Churn_Rate=(TARGET,"mean")).round(3)
            di2["Churn Rate %"]=(di2["Churn_Rate"]*100).round(1)
            st.dataframe(di2.drop(columns="Churn_Rate").reset_index(),use_container_width=True,hide_index=True)
        st.markdown("<div class='section-title'>Full Per-Order Cost Breakdown</div>",unsafe_allow_html=True)
        cost=view.dropna(subset=["warehouse_hub"]).groupby("warehouse_hub").agg(Avg_Last_Mile=("last_mile_cost_usd","mean"),Avg_Fulfillment=("fulfillment_cost_usd","mean"),Avg_Return_Cost=("return_cost_usd","mean"),Avg_AOV=("avg_order_value_usd","mean")).round(2)
        cost["Total Cost Est"]=cost["Avg_Last_Mile"]+cost["Avg_Fulfillment"]+cost["Avg_Return_Cost"]
        cost["Cost as % AOV"]=(cost["Total Cost Est"]/cost["Avg_AOV"]*100).round(1)
        st.dataframe(cost.reset_index(),use_container_width=True,hide_index=True)
        au_hub=view[view["warehouse_hub"]=="AU Hub"]["logistics_cost_pct_revenue"].mean()*100
        sg_hub=view[view["warehouse_hub"]=="SG Hub"]["logistics_cost_pct_revenue"].mean()*100
        st.markdown(f"<div class='insight'><b>Key finding:</b> AU Hub carries the highest logistics cost at {au_hub:.1f}% of revenue vs SG Hub at {sg_hub:.1f}%. Philippines-routed orders show the longest delivery times and highest per-order cost exposure. Return costs compound margin pressure in high-discount markets.</div>",unsafe_allow_html=True)

    with g3:
        st.markdown("<div class='ps-header ps3-header'>Is growth driven by sustainable behaviour or discount & influencer dependency?</div>",unsafe_allow_html=True)
        a,b=st.columns(2,gap="large")
        with a:
            st.markdown("<div class='section-title'>Discount Dependency vs CLV by Channel</div>",unsafe_allow_html=True)
            dd2=view.dropna(subset=["acquisition_channel"]).groupby("acquisition_channel").agg(Avg_Disc_Dep=("discount_dependency_pct","mean"),Avg_CLV=("clv_12m_usd","mean"),Avg_Margin=("gross_margin_pct","mean")).mul({"Avg_Disc_Dep":100,"Avg_CLV":1,"Avg_Margin":100}).round(2)
            st.dataframe(dd2.reset_index(),use_container_width=True,hide_index=True)
            st.bar_chart(dd2[["Avg_Disc_Dep","Avg_Margin"]])
        with b:
            st.markdown("<div class='section-title'>Campaign ROI by Campaign Type</div>",unsafe_allow_html=True)
            camp2=view[view["campaign_last_sent"].notna() & (view["campaign_last_sent"]!="None")].groupby("campaign_last_sent").agg(Avg_ROI=("campaign_roi","mean"),Response_Rate=("responded_to_campaign","mean"),Avg_Incremental=("incremental_orders_from_campaign","mean")).round(3)
            camp2["Response %"]=(camp2["Response_Rate"]*100).round(1)
            st.bar_chart(camp2[["Avg_ROI"]])
            st.dataframe(camp2.drop(columns="Response_Rate").reset_index(),use_container_width=True,hide_index=True)
        a,b=st.columns(2,gap="large")
        with a:
            st.markdown("<div class='section-title'>Gross Margin by Uplift Segment</div>",unsafe_allow_html=True)
            gms=view.dropna(subset=["uplift_segment"]).groupby("uplift_segment")["gross_margin_pct"].mean().mul(100).round(2).to_frame("Gross Margin %").sort_values("Gross Margin %",ascending=False)
            st.bar_chart(gms);st.dataframe(gms.reset_index(),use_container_width=True,hide_index=True)
        with b:
            st.markdown("<div class='section-title'>CLV by Uplift Segment</div>",unsafe_allow_html=True)
            clvs=view.dropna(subset=["uplift_segment"]).groupby("uplift_segment")["clv_12m_usd"].mean().round(2).to_frame("Avg 12M CLV ($)").sort_values("Avg 12M CLV ($)",ascending=False)
            st.bar_chart(clvs);st.dataframe(clvs.reset_index(),use_container_width=True,hide_index=True)
        infl_dd=view[view["acquisition_channel"]=="Influencer"]["discount_dependency_pct"].mean()*100
        org_dd=view[view["acquisition_channel"]=="Organic"]["discount_dependency_pct"].mean()*100
        st.markdown(f"<div class='insight'><b>Key finding:</b> Influencer-acquired customers show {infl_dd:.1f}% avg discount dependency vs {org_dd:.1f}% for organic. Do-Not-Disturb and Lost Cause segments drag down avg CLV and gross margin significantly — disproportionate campaign spend on these cohorts is the core profitability risk.</div>",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    p1,p2,p3=st.tabs(["PS1 · Churn Prediction","PS2 · Logistics Risk Prediction","PS3 · Campaign Response Prediction"])

    with p1:
        st.markdown("<div class='ps-header ps1-header'>Which customers are at risk of churning in the next 90 days?</div>",unsafe_allow_html=True)
        a,b=st.columns([1.1,0.9],gap="large")
        with a:
            st.markdown("<div class='section-title'>Churn Prediction Model Scorecard</div>",unsafe_allow_html=True)
            st.dataframe(ps1_met,use_container_width=True,hide_index=True)
            st.markdown(f"<div class='insight'><b>{ps1_best}</b> leads with ROC AUC = {ps1_met.iloc[0]['ROC AUC']:.4f}. Score all active customers weekly to flag at-risk cohorts before 90-day lapse.</div>",unsafe_allow_html=True)
        with b:
            st.markdown("<div class='section-title'>Confusion Matrix</div>",unsafe_allow_html=True)
            cm1=pd.DataFrame(ps1_fit[ps1_best]["cm"],index=["Actual: Retained","Actual: Churned"],columns=["Pred: Retained","Pred: Churned"])
            st.dataframe(cm1,use_container_width=True)
            st.markdown("<div class='small-muted'>Rows = actual · Columns = predicted churn.</div>",unsafe_allow_html=True)
        a,b=st.columns(2,gap="large")
        with a:
            st.markdown("<div class='section-title'>Feature Importance — Churn Drivers</div>",unsafe_allow_html=True)
            fi1=pd.Series(ps1_bm.fi,index=ps1_fc).sort_values(ascending=False).round(4)
            st.bar_chart(fi1.head(10))
            fi1_df=fi1.head(10).reset_index();fi1_df.columns=["Feature","Importance"]
            st.dataframe(fi1_df,use_container_width=True,hide_index=True)
        with b:
            st.markdown("<div class='section-title'>ROC Curve</div>",unsafe_allow_html=True)
            st.line_chart(ps1_fit[ps1_best]["roc"].set_index("FPR"))
            st.markdown("<div class='small-muted'>Upper-left = stronger discrimination.</div>",unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Churn Risk Score Distribution</div>",unsafe_allow_html=True)
        rb=pd.cut(view["churn_risk_score"].fillna(0),bins=[0,25,50,75,100],labels=["Low (0-25)","Medium (25-50)","High (50-75)","Critical (75-100)"],include_lowest=True)
        rb_df=rb.value_counts().to_frame("Customers")
        st.bar_chart(rb_df);st.dataframe(rb_df.reset_index().rename(columns={"index":"Risk Band"}),use_container_width=True,hide_index=True)

    with p2:
        st.markdown("<div class='ps-header ps2-header'>Which orders and routes are at risk of high logistics cost overrun?</div>",unsafe_allow_html=True)
        a,b=st.columns([1.1,0.9],gap="large")
        with a:
            st.markdown("<div class='section-title'>High Logistics Risk Model Scorecard</div>",unsafe_allow_html=True)
            st.dataframe(ps2_met,use_container_width=True,hide_index=True)
            st.markdown(f"<div class='insight'><b>{ps2_best}</b> predicts whether an order will fall in the high-cost logistics bucket (above median logistics cost %). Use to pre-flag risky routes and optimise warehouse allocation.</div>",unsafe_allow_html=True)
        with b:
            st.markdown("<div class='section-title'>Confusion Matrix</div>",unsafe_allow_html=True)
            cm2=pd.DataFrame(ps2_fit[ps2_best]["cm"],index=["Actual: Low Cost","Actual: High Cost"],columns=["Pred: Low Cost","Pred: High Cost"])
            st.dataframe(cm2,use_container_width=True)
        a,b=st.columns(2,gap="large")
        with a:
            st.markdown("<div class='section-title'>Feature Importance — Logistics Cost Drivers</div>",unsafe_allow_html=True)
            fi2=pd.Series(ps2_bm.fi,index=ps2_fc).sort_values(ascending=False).round(4)
            st.bar_chart(fi2.head(10))
            fi2_df=fi2.head(10).reset_index();fi2_df.columns=["Feature","Importance"]
            st.dataframe(fi2_df,use_container_width=True,hide_index=True)
        with b:
            st.markdown("<div class='section-title'>ROC Curve</div>",unsafe_allow_html=True)
            st.line_chart(ps2_fit[ps2_best]["roc"].set_index("FPR"))
        st.markdown("<div class='section-title'>High Logistics Risk Customers by Country</div>",unsafe_allow_html=True)
        hl=view_scored.dropna(subset=["country"]).groupby("country")["High Logistics Risk"].mean().mul(100).round(1).to_frame("High Cost Risk %").sort_values("High Cost Risk %",ascending=False)
        st.bar_chart(hl);st.dataframe(hl.reset_index(),use_container_width=True,hide_index=True)

    with p3:
        st.markdown("<div class='ps-header ps3-header'>Which customers will respond to campaigns and generate incremental orders?</div>",unsafe_allow_html=True)
        a,b=st.columns([1.1,0.9],gap="large")
        with a:
            st.markdown("<div class='section-title'>Campaign Response Prediction Scorecard</div>",unsafe_allow_html=True)
            st.dataframe(ps3_met,use_container_width=True,hide_index=True)
            st.markdown(f"<div class='insight'><b>{ps3_best}</b> predicts whether a customer will respond to a campaign. Prioritise Persuadable customers with high response probability to maximise incremental ROI.</div>",unsafe_allow_html=True)
        with b:
            st.markdown("<div class='section-title'>Confusion Matrix</div>",unsafe_allow_html=True)
            cm3=pd.DataFrame(ps3_fit[ps3_best]["cm"],index=["Actual: No Response","Actual: Responded"],columns=["Pred: No Response","Pred: Responded"])
            st.dataframe(cm3,use_container_width=True)
        a,b=st.columns(2,gap="large")
        with a:
            st.markdown("<div class='section-title'>Feature Importance — Campaign Response Drivers</div>",unsafe_allow_html=True)
            fi3=pd.Series(ps3_bm.fi,index=ps3_fc).sort_values(ascending=False).round(4)
            st.bar_chart(fi3.head(10))
            fi3_df=fi3.head(10).reset_index();fi3_df.columns=["Feature","Importance"]
            st.dataframe(fi3_df,use_container_width=True,hide_index=True)
        with b:
            st.markdown("<div class='section-title'>ROC Curve</div>",unsafe_allow_html=True)
            st.line_chart(ps3_fit[ps3_best]["roc"].set_index("FPR"))
        st.markdown("<div class='section-title'>Predicted Campaign Conversion by Uplift Segment</div>",unsafe_allow_html=True)
        if "Campaign Convert Prob" in view_scored.columns:
            cv=view_scored.dropna(subset=["uplift_segment"]).groupby("uplift_segment")["Campaign Convert Prob"].mean().round(3).to_frame("Avg Convert Prob").sort_values("Avg Convert Prob",ascending=False)
            st.bar_chart(cv);st.dataframe(cv.reset_index(),use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PRESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_presc:
    r1,r2,r3=st.tabs(["PS1 · Retention Actions","PS2 · Logistics Optimisation","PS3 · Growth Spend Strategy"])

    with r1:
        st.markdown("<div class='ps-header ps1-header'>What specific actions should be taken to retain high-value customers?</div>",unsafe_allow_html=True)
        a,b=st.columns(2,gap="large")
        with a:
            st.markdown("<div class='section-title'>Persuadable Lead Board (Top CLV)</div>",unsafe_allow_html=True)
            leads=view_scored[view_scored["uplift_segment"]=="Persuadable"].sort_values("clv_12m_usd",ascending=False)
            st.dataframe(leads[["customer_id","country","loyalty_tier","clv_12m_usd","churn_risk_score","discount_dependency_pct"]].head(15),use_container_width=True,hide_index=True)
        with b:
            st.markdown("<div class='section-title'>At-Risk High-CLV Customers</div>",unsafe_allow_html=True)
            q60=scored["clv_12m_usd"].quantile(0.6)
            at_risk=view_scored[(view_scored["Predicted Churn"]==1)&(view_scored["clv_12m_usd"]>q60)].sort_values("clv_12m_usd",ascending=False)
            st.dataframe(at_risk[["customer_id","country","clv_12m_usd","churn_risk_score","recency_days","loyalty_tier"]].head(15),use_container_width=True,hide_index=True)
        st.markdown("<div class='section-title'>Segment-Level Retention Playbook</div>",unsafe_allow_html=True)
        for seg,cls,title in [
            ("Persuadable","offer","\U0001f3af Persuadable — Activate Now"),
            ("Sure-Thing","offer","\u2705 Sure-Thing — Lock In & Upgrade"),
            ("Do-Not-Disturb","warn","\u26a0\ufe0f Do-Not-Disturb — Reduce Promo Spend"),
            ("Lost Cause","warn","\U0001f534 Lost Cause — Minimal Spend Only"),
        ]:
            sd=view[view["uplift_segment"]==seg]
            if len(sd)==0: continue
            st.markdown(f"<div class='{cls}'><b>{title}</b> &nbsp;·&nbsp; {len(sd)} customers &nbsp;·&nbsp; Avg CLV: ${sd['clv_12m_usd'].mean():.0f} &nbsp;·&nbsp; Churn Rate: {sd[TARGET].mean()*100:.1f}%<br>{seg_rx(seg)}</div>",unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Churn Intervention Simulator</div>",unsafe_allow_html=True)
        s1,s2,s3=st.columns(3,gap="large")
        with s1:
            sim_rec=st.slider("Recency (days)",1,180,45)
            sim_freq=st.slider("Frequency 90d",0,15,3)
            sim_mon=st.slider("Monetary 3M ($)",0,600,100)
        with s2:
            sim_disc=st.slider("Discount Dependency",0.10,0.95,0.50)
            sim_ret=st.slider("Returns Rate",0.0,0.6,0.15)
            sim_di=st.slider("Delivery Issues",0,5,0)
        with s3:
            sim_clv=st.slider("Est. 12M CLV ($)",10,850,200)
            sim_tier=st.selectbox("Loyalty Tier",["None","Bronze","Silver","Gold"])
            sim_del=st.slider("Avg Delivery Days",1.0,12.0,4.5)
        sv=np.array([[sim_rec,sim_freq//3,sim_freq,sim_mon,
                      sim_mon/max(sim_freq,1),12,5,sim_disc,sim_ret,0,sim_di,0,sim_clv]])[:,:len(ps1_fc)]
        cp=float(ps1_bm.predict_proba(sv)[0,1])
        rl="\U0001f534 HIGH CHURN RISK" if cp>0.65 else "\U0001f7e1 MEDIUM RISK" if cp>0.4 else "\U0001f7e2 LOW RISK"
        act=churn_rx({"recency_days":sim_rec,"discount_dependency_pct":sim_disc,"delivery_issues_count":sim_di,"returns_rate":sim_ret,"loyalty_tier":sim_tier})
        st.markdown(f"<div class='insight'><b>Simulated Churn Probability: {cp:.1%} — {rl}</b><br><b>Recommended Action:</b> {act}</div>",unsafe_allow_html=True)

    with r2:
        st.markdown("<div class='ps-header ps2-header'>How should logistics be optimised to protect margin across APAC hubs?</div>",unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Hub Optimisation Recommendations</div>",unsafe_allow_html=True)
        hub_df=view.dropna(subset=["warehouse_hub"]).groupby("warehouse_hub").agg(
            Customers=("customer_id","count"),Avg_Logistics_Cost_Pct=("logistics_cost_pct_revenue","mean"),
            Avg_Delivery_Days=("avg_delivery_days","mean"),Avg_Return_Cost=("return_cost_usd","mean"),
            Avg_Gross_Margin=("gross_margin_pct","mean"),Delivery_Issues_Rate=("delivery_issues_count","mean")
        ).mul({"Customers":1,"Avg_Logistics_Cost_Pct":100,"Avg_Delivery_Days":1,"Avg_Return_Cost":1,"Avg_Gross_Margin":100,"Delivery_Issues_Rate":1}).round(2)
        st.dataframe(hub_df.reset_index(),use_container_width=True,hide_index=True)
        for hub,action,cls in [
            ("AU Hub","Highest logistics cost hub. Prioritise local supplier partnerships to reduce last-mile cost. Test free-shipping threshold increase to $60+ to protect margin.","warn"),
            ("SG Hub","Most efficient hub. Expand SKU depth for SEA markets routed here (Indonesia, Malaysia, Philippines). Leverage speed advantage in marketing.","offer"),
            ("VN Hub","High delivery volume with moderate cost. Focus on reducing return rates via better product imagery and size guides for Thailand/Vietnam markets.","offer"),
            ("Dubai Hub","Smallest volume but highest AOV market. Protect with premium service SLA and dedicated customer success for UAE high-CLV customers.","offer"),
        ]:
            if hub in view["warehouse_hub"].values:
                st.markdown(f"<div class='{cls}'><b>{hub}</b><br>{action}</div>",unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Returns Reduction Priorities by Style Category</div>",unsafe_allow_html=True)
        a,b=st.columns(2,gap="large")
        with a:
            rr2=view.dropna(subset=["style_category"]).groupby("style_category").agg(Avg_Returns=("returns_rate","mean"),Avg_Return_Cost=("return_cost_usd","mean"),Customers=("customer_id","count")).round(2)
            rr2["Avg Returns %"]=(rr2["Avg_Returns"]*100).round(1)
            st.dataframe(rr2.drop(columns="Avg_Returns").reset_index(),use_container_width=True,hide_index=True)
        with b:
            st.bar_chart(rr2[["Avg Returns %"]])

    with r3:
        st.markdown("<div class='ps-header ps3-header'>Where should marketing spend go to build sustainable profitable growth?</div>",unsafe_allow_html=True)
        a,b=st.columns(2,gap="large")
        with a:
            st.markdown("<div class='section-title'>Campaign ROI Ranking</div>",unsafe_allow_html=True)
            cr2=view[view["campaign_last_sent"].notna() & (view["campaign_last_sent"]!="None")].groupby("campaign_last_sent").agg(
                Avg_ROI=("campaign_roi","mean"),Response_Rate=("responded_to_campaign","mean"),
                Avg_CLV_Impact=("clv_12m_usd","mean"),Avg_Disc_Depth=("promo_depth_pct","mean")).round(3)
            cr2["Response %"]=(cr2["Response_Rate"]*100).round(1);cr2["Disc Depth %"]=(cr2["Avg_Disc_Depth"]*100).round(1)
            st.bar_chart(cr2[["Avg_ROI"]])
            st.dataframe(cr2.drop(columns=["Response_Rate","Avg_Disc_Depth"]).reset_index(),use_container_width=True,hide_index=True)
        with b:
            st.markdown("<div class='section-title'>Predicted Converters by Country</div>",unsafe_allow_html=True)
            if "Campaign Convert Prob" in view_scored.columns:
                pconv=view_scored.dropna(subset=["country"]).groupby("country")["Campaign Convert Prob"].mean().round(3).to_frame("Avg Convert Prob").sort_values("Avg Convert Prob",ascending=False)
                st.bar_chart(pconv);st.dataframe(pconv.reset_index(),use_container_width=True,hide_index=True)
        st.markdown("<div class='section-title'>Growth Strategy Recommendations</div>",unsafe_allow_html=True)
        for rec,cls in [
            ("\U0001f4b0 <b>Shift campaign evaluation to 12M CLV impact, not one-shot revenue.</b> Customers acquired via Mega-Sale events show high churn and low CLV — measure cohort retention at 3M, 6M, 12M to surface this.","warn"),
            ("\U0001f4f1 <b>Invest in Style Drop and loyalty-first campaigns over voucher-only.</b> Style Drop campaigns show stronger CLV-preserving engagement without training pure discount behaviour.","offer"),
            ("\U0001f6ab <b>Cap influencer promo depth for new customer acquisition.</b> Influencer codes are the primary driver of high discount dependency. Shift influencer partnerships to brand/style content rather than pure price codes.","warn"),
            ("\U0001f3af <b>Focus A/B testing budget on Persuadable segment only.</b> Run campaign variant tests within the {persuadable} Persuadable customers identified — they have the highest incremental lift potential per $ spent.","offer"),
            ("\U0001f4ca <b>Deploy uplift model scoring monthly.</b> Re-score all customers monthly to migrate them from Lost Cause / Do-Not-Disturb into Persuadable as their behaviour changes, enabling dynamic campaign targeting.","offer"),
        ]:
            st.markdown(f"<div class='{cls}'>{rec.format(persuadable=persuadable)}</div>",unsafe_allow_html=True)

st.caption(f"Shein APAC Intelligence Suite · {total} customers · 8 markets · PS1 Churn · PS2 Logistics · PS3 Sustainable Growth")
