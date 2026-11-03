from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).parent
DATA = ROOT / "data"

st.set_page_config(page_title="AuditFlux | Research Dashboard", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
.stApp { background: #0b1020; color: #e8edf7; }
section[data-testid="stSidebar"] { background: #10182b; border-right: 1px solid #26324b; }
section[data-testid="stSidebar"] * { color: #dce5f5; }
.block-container { padding-top: 2.0rem; max-width: 1500px; }
.hero { padding: 1.35rem 1.6rem; border: 1px solid #293754; border-radius: 18px; background: linear-gradient(120deg,#17243c,#111a2d 65%,#162b37); margin-bottom: 1.2rem; }
.eyebrow { color:#70d6c4; font-family:'DM Mono',monospace; font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; }
.hero h1 { font-size:2.25rem; margin:.3rem 0 .25rem; letter-spacing:-.05em; color:#f3f6fc; }
.hero p { color:#aab8ce; margin:0; font-size:.98rem; }
.kpi { background:#131e33; border:1px solid #293754; border-radius:14px; padding:1rem 1.1rem; min-height:110px; }
.kpi-label { color:#9eacc3; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
.kpi-value { color:#f3f6fc; font-size:1.8rem; font-weight:800; margin-top:.3rem; }
.kpi-note { color:#70d6c4; font-size:.75rem; margin-top:.1rem; }
.section-note { background:#121d31; border-left:3px solid #70d6c4; padding:.8rem 1rem; border-radius:5px; color:#c3cee0; margin:.5rem 0 1rem; }
div[data-testid="stDataFrame"] { border:1px solid #293754; border-radius:10px; }
h1,h2,h3 { letter-spacing:-.03em; }
[data-testid="stMetric"] { background:#131e33; border:1px solid #293754; padding:14px 16px; border-radius:12px; }
[data-testid="stMetricLabel"] { color:#aab8ce !important; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

FILES = {
    "Baseline audit": "baseline_audit_all_models(1).csv",
    "Threshold sweep": "threshold_sweep_all_models(2).csv",
    "Metric comparison": "metric_comparison_all_models(1).csv",
    "Sample instability": "sample_instability_metrics_all_models_revised.csv",
    "Threshold instability": "threshold_instability_metrics_all_models_revised.csv",
    "Representation associations": "21_cross_representation_associations.csv",
    "Adjusted regressions": "21_cross_representation_adjusted_regressions.csv",
    "Prediction metrics": "21_cross_representation_prediction_metrics.csv",
    "Huber robustness": "21B_huber_all_thresholds.csv",
    "Influence diagnostics": "21B_influence_primary_threshold.csv",
    "Identity influence refits": "21B_top10_leave_one_identity_out.csv",
    "Group compactness": "20_representation_space_group_compactness_summary.csv",
    "Held-out representation audit": "20_representation_space_heldout_verification_audit.csv",
}

@st.cache_data(show_spinner=False)
def load(name):
    path = DATA / FILES[name]
    if not path.exists(): return pd.DataFrame()
    return pd.read_csv(path)

def kpi(label, value, note=""):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)

def chart_theme(fig, height=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
                      margin=dict(l=15,r=15,t=45,b=15), font=dict(family="Manrope",color="#dce5f5"),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#26334b", zerolinecolor="#26334b")
    fig.update_yaxes(gridcolor="#26334b", zerolinecolor="#26334b")
    return fig

with st.sidebar:
    st.markdown("### ◈ AUDITFLUX")
    st.caption("Research & experimental analysis")
    page = st.radio("NAVIGATION", ["Overview", "Fairness audit", "Instability analysis", "Representation study", "Robustness validation", "Data explorer"], label_visibility="visible")
    st.divider()
    st.caption("DATA SNAPSHOT")
    st.write(f"{len(FILES)} curated result files")
    st.caption("Results are read from the bundled CSVs. No experiment is rerun by this dashboard.")

st.markdown('<div class="hero"><div class="eyebrow">Predictive instability quantification</div><h1>AuditFlux</h1><p>Exploring the reliability, sensitivity and robustness of fairness conclusions in deep face verification.</p></div>', unsafe_allow_html=True)

if page == "Overview":
    st.subheader("Research at a glance")
    c = st.columns(4)
    with c[0]: kpi("Experiments", "21+", "Research programme")
    with c[1]: kpi("Face groups", "8", "Intersectional demographic groups")
    with c[2]: kpi("Verification models", "3", "VGG16 · ResNet50 · SENet50")
    with c[3]: kpi("Representation", "512-D", "FaceNet-CASIA embeddings")
    st.markdown("### The research workflow")
    stages = st.columns(5)
    for col, num, title, desc in zip(stages, ["01","02","03","04","05"], ["Measure","Stress-test","Calibrate","Predict","Investigate"], ["Baseline fairness metrics", "Thresholds and sampling", "Null and uncertainty", "Pilot-to-target models", "Representation correlates"]):
        with col:
            st.markdown(f'<div class="kpi"><div class="eyebrow">{num}</div><h3>{title}</h3><div style="color:#9eacc3;font-size:.83rem">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown("### Flagship evidence · Programs 21 and 21B")
    a,b,c,d = st.columns(4)
    with a: kpi("Holm-significant tests", "74 / 108", "Adjusted association tests")
    with b: kpi("MAE improvements", "23 / 27", "Group + geometry vs group only")
    with c: kpi("Signs preserved", "80 / 81", "OLS vs Huber across thresholds")
    with d: kpi("Influence sign reversals", "0 / 90", "Top-10 identity leave-one-out refits")
    st.markdown('<div class="section-note"><b>Interpretation:</b> Representation-space characteristics provide additional predictive information about identity-level verification error. The observed coefficient directions also show substantial robustness to robust-regression fitting and influential-identity removal. These are observational findings, not causal claims.</div>', unsafe_allow_html=True)
    st.markdown("### Project scope")
    st.write("AuditFlux studies how fairness-audit conclusions change under reasonable evaluation choices, whether that instability can be anticipated from pilot information, and whether representation-space properties help explain variation in verification errors.")
    st.caption("The headline counts above are project-reported results encoded from the current experiment outputs. Explore the underlying tables in the relevant sections.")

elif page == "Fairness audit":
    st.subheader("Fairness audit")
    df=load("Baseline audit")
    if df.empty: st.error("Baseline results file is missing.")
    else:
        models=sorted(df.model.unique()); model=st.selectbox("Verification model",models)
        metric=st.selectbox("Metric",["FNR","FPR","TPR"])
        d=df[df.model==model].sort_values("group")
        fig=px.bar(d,x="group",y=metric,title=f"{metric} by demographic group · {model}",labels={metric:metric,"group":"Demographic group"},text_auto=".3f")
        st.plotly_chart(chart_theme(fig,400),use_container_width=True)
        st.markdown("#### Baseline group-level results")
        st.dataframe(d[["group","TPR","FPR","FNR","samples","positive_pairs","negative_pairs","threshold"]].style.format({"TPR":"{:.4f}","FPR":"{:.4f}","FNR":"{:.4f}"}),use_container_width=True,hide_index=True)
        st.caption("Baseline threshold = 0.50. Model score scales differ; the same numerical threshold does not imply the same operating point across architectures.")
        ts=load("Threshold sweep")
        if not ts.empty:
            st.markdown("#### Threshold sensitivity")
            met=st.selectbox("Threshold-sweep metric",["FNR","FPR","TPR"],key="thmetric")
            groups=st.multiselect("Groups",sorted(ts.group.unique()),default=sorted(ts.group.unique())[:4])
            td=ts[(ts.model==model)&(ts.group.isin(groups))]
            fig=px.line(td,x="threshold",y=met,color="group",markers=True,title=f"{met} across thresholds · {model}")
            st.plotly_chart(chart_theme(fig),use_container_width=True)

elif page == "Instability analysis":
    st.subheader("Instability analysis")
    st.markdown("Compare how sampling size and threshold transitions affect group gaps, worst-group identity and ranking stability.")
    sample=load("Sample instability")
    if not sample.empty:
        c1,c2=st.columns(2)
        with c1: model=st.selectbox("Model",sorted(sample.model.unique()))
        with c2: metric=st.selectbox("Metric",sorted(sample.metric.unique()))
        sd=sample[(sample.model==model)&(sample.metric==metric)]
        fig=px.box(sd,x="sample_size",y="group_metric_gap",points=False,title="Distribution of group metric gaps by sample size",labels={"sample_size":"Sample size","group_metric_gap":"Group metric gap"})
        st.plotly_chart(chart_theme(fig),use_container_width=True)
        st.caption("Each point in the source repetitions represents a sampling repetition; this view summarises their gap distribution.")
    ti=load("Threshold instability")
    if not ti.empty:
        st.markdown("#### Threshold-transition instability")
        tm=st.selectbox("Transition metric",sorted(ti.metric.unique()),key="tim")
        model2=st.selectbox("Transition model",sorted(ti.model.unique()),key="timod")
        x=ti[(ti.metric==tm)&(ti.model==model2)].dropna(subset=["threshold"])
        fig=px.line(x,x="threshold",y="ranking_instability",markers=True,title="Ranking instability across adjacent thresholds",labels={"threshold":"Threshold","ranking_instability":"Normalised ranking instability"})
        st.plotly_chart(chart_theme(fig),use_container_width=True)
        st.dataframe(x[["threshold","worst_group","worst_group_reversal","kendall_tau_b_previous","ranking_instability","absolute_gap_change"]],use_container_width=True,hide_index=True)

elif page == "Representation study":
    st.subheader("Representation-space analysis · Program 21")
    st.markdown("FaceNet-CASIA identity-level geometry is examined alongside identity-level false-negative rates from the original verification models.")
    assoc=load("Representation associations")
    pred=load("Prediction metrics")
    reg=load("Adjusted regressions")
    if not assoc.empty:
        c1,c2=st.columns(2)
        with c1: model=st.selectbox("Model",sorted(assoc.model.unique()))
        with c2: threshold=st.selectbox("Threshold",sorted(assoc.threshold.unique()),index=list(sorted(assoc.threshold.unique())).index(0.5) if 0.5 in sorted(assoc.threshold.unique()) else 0)
        ad=assoc[(assoc.model==model)&(assoc.threshold==threshold)]
        fig=px.bar(ad,x="feature",y="spearman_rho",color="scope",barmode="group",title="Representation feature association with identity-level FNR",labels={"feature":"Representation feature","spearman_rho":"Spearman correlation"})
        st.plotly_chart(chart_theme(fig,430),use_container_width=True)
        st.dataframe(ad,use_container_width=True,hide_index=True)
    if not pred.empty:
        st.markdown("#### Predictive value of representation geometry")
        pm=pred[(pred.model==model)&(pred.threshold==threshold)]
        if not pm.empty:
            fig=go.Figure()
            for metric in ["mae","rmse"]:
                fig.add_trace(go.Bar(name=metric.upper(),x=pm.feature_set,y=pm[metric]))
            fig.update_layout(barmode="group",title="Prediction error by feature set",xaxis_title="Feature set",yaxis_title="Error")
            st.plotly_chart(chart_theme(fig),use_container_width=True)
            st.dataframe(pm,use_container_width=True,hide_index=True)
    if not reg.empty:
        st.markdown("#### Adjusted regression results")
        rd=reg[(reg.model==model)&(reg.threshold==threshold)].copy()
        if "p_value_holm_global" in rd:
            rd["holm_significant"] = rd.p_value_holm_global < .05
        st.dataframe(rd,use_container_width=True,hide_index=True)
    compact=load("Group compactness")
    if not compact.empty:
        st.markdown("#### FaceNet identity compactness by group")
        fig=px.bar(compact,x="group",y="mean_identity_compactness",error_y=compact["p97_5_stratified_identity_bootstrap"]-compact["mean_identity_compactness"],error_y_minus=compact["mean_identity_compactness"]-compact["p2_5_stratified_identity_bootstrap"],title="Mean identity compactness with stratified identity-bootstrap interval")
        st.plotly_chart(chart_theme(fig),use_container_width=True)

elif page == "Robustness validation":
    st.subheader("Robustness validation · Program 21B")
    st.markdown("The analysis checks whether coefficient directions persist under Huber robust regression and influential-identity removal.")
    hub=load("Huber robustness"); inf=load("Influence diagnostics"); loo=load("Identity influence refits")
    if not hub.empty:
        same=int(hub.same_sign.sum()); total=len(hub)
        c1,c2=st.columns(2)
        with c1: kpi("Coefficient signs preserved",f"{same} / {total}","OLS vs Huber across all evaluated thresholds")
        with c2:
            reversals=int(inf.top10_loo_sign_reversals.sum()) if not inf.empty else 0
            kpi("Influence sign reversals",f"{reversals} / 90","Top-10 leave-one-identity-out refits")
        hm=hub.groupby("threshold",as_index=False).agg(preserved=("same_sign","sum"),tests=("same_sign","count"))
        hm["not_preserved"]=hm.tests-hm.preserved
        fig=px.bar(hm,x="threshold",y=["preserved","not_preserved"],title="Coefficient sign consistency by threshold",labels={"value":"Number of tests","threshold":"Threshold","variable":"Sign comparison"})
        st.plotly_chart(chart_theme(fig),use_container_width=True)
        st.dataframe(hub,use_container_width=True,hide_index=True)
    if not inf.empty:
        st.markdown("#### Primary-threshold influence diagnostics")
        st.dataframe(inf,use_container_width=True,hide_index=True)
        st.caption("The 90 refits are 9 model-feature regressions × 10 individually removed influential identities. The maximum relative coefficient change is reported in the source table.")
    if not loo.empty:
        st.markdown("#### Individual identity refits")
        st.dataframe(loo.sort_values("relative_coefficient_change",ascending=False),use_container_width=True,hide_index=True)

elif page == "Data explorer":
    st.subheader("Experiment output explorer")
    name=st.selectbox("Select result file",list(FILES.keys()))
    df=load(name)
    if df.empty: st.warning("This result file is not present in the dashboard bundle.")
    else:
        st.caption(f"Source: {FILES[name]} · {len(df):,} rows × {len(df.columns)} columns")
        search=st.text_input("Filter rows (search all visible text)")
        if search:
            mask=df.astype(str).apply(lambda col: col.str.contains(search,case=False,na=False)).any(axis=1)
            df=df[mask]
        st.dataframe(df,use_container_width=True,hide_index=True,height=520)
        st.download_button("Download this CSV",df.to_csv(index=False).encode("utf-8"),file_name=FILES[name],mime="text/csv")

st.divider()
st.caption("AUDITFLUX · Research prototype · Results are descriptive/observational where indicated. Consult experiment manifests and scripts for full methodology, assumptions and limitations.")
