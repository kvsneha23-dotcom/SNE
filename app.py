"""
Insurance Claim Settlement Bias Analysis
Streamlit Dashboard – Full Analysis Suite
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
)
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Claim Bias Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800; color: #1a237e;
        text-align: center; padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem; color: #546e7a; text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 10px; color: white; text-align: center;
    }
    .finding-box {
        background: #fff3cd; border-left: 4px solid #ff9800;
        padding: 1rem; border-radius: 5px; margin: 0.5rem 0;
    }
    .bias-alert {
        background: #ffebee; border-left: 4px solid #f44336;
        padding: 1rem; border-radius: 5px; margin: 0.5rem 0;
    }
    .good-box {
        background: #e8f5e9; border-left: 4px solid #4caf50;
        padding: 1rem; border-radius: 5px; margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 44px; padding: 0 20px; font-weight: 600; font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_engineer(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.error("Please upload the Insurance CSV file.")
        st.stop()

    # Clean numerics
    df["SUM_ASSURED"]      = df["SUM_ASSURED"].astype(str).str.replace(",","").str.strip().astype(float)
    df["PI_ANNUAL_INCOME"] = df["PI_ANNUAL_INCOME"].astype(str).str.replace(",","").str.strip().astype(float)

    # Target
    df["TARGET"] = (df["POLICY_STATUS"] == "Approved Death Claim").astype(int)

    # Fill missing
    df["PI_OCCUPATION"].fillna("Unknown", inplace=True)
    df["REASON_FOR_CLAIM"].fillna("Not Specified", inplace=True)

    # Bins
    df["AGE_GROUP"] = pd.cut(df["PI_AGE"], bins=[0,40,50,60,70,80,110],
                              labels=["<40","40-50","50-60","60-70","70-80","80+"])
    df["INCOME_GROUP"] = pd.cut(df["PI_ANNUAL_INCOME"],
                                 bins=[-1,0,100000,300000,600000,100000000],
                                 labels=["No Data","Low","Medium","High","Very High"])

    # Feature engineering
    df["IS_CREDITOR_ZONE"]  = df["ZONE"].str.contains("CREDITOR|HOUSING", na=False).astype(int)
    df["IS_TEAM_ZONE"]      = df["ZONE"].str.startswith("TEAM", na=False).astype(int)
    df["AGE_SQUARED"]       = df["PI_AGE"] ** 2
    df["AGE_OVER_70"]       = (df["PI_AGE"] >= 70).astype(int)
    df["HAS_INCOME"]        = (df["PI_ANNUAL_INCOME"] > 0).astype(int)
    df["LOG_INCOME"]        = np.log1p(df["PI_ANNUAL_INCOME"])
    df["LOG_SUM_ASSURED"]   = np.log1p(df["SUM_ASSURED"])
    df["INCOME_TO_ASSURED"] = df["PI_ANNUAL_INCOME"] / (df["SUM_ASSURED"] + 1)
    df["EARLY_x_MEDICAL"]   = ((df["EARLY_NON"]=="EARLY").astype(int) *
                                (df["MEDICAL_NONMED"]=="MEDICAL").astype(int))
    df["IS_SINGLE_PAYMENT"] = (df["PAYMENT_MODE"]=="Single").astype(int)
    df["IS_ANNUAL"]         = (df["PAYMENT_MODE"]=="Annual").astype(int)
    df["REASON_KNOWN"]      = (df["REASON_FOR_CLAIM"] != "Not Specified").astype(int)
    heart_causes = ["Heart Attack","Heart Failure","Cardiac Arrest",
                    "Congestive Cardiac Failure","Cardio Respiratory Arrest","Cardio Pulmonary Arrest"]
    df["CARDIAC_CAUSE"]     = df["REASON_FOR_CLAIM"].isin(heart_causes).astype(int)

    le = LabelEncoder()
    for c in ["PI_GENDER","ZONE","PAYMENT_MODE","EARLY_NON","PI_OCCUPATION",
              "MEDICAL_NONMED","PI_STATE"]:
        df[c+"_ENC"] = le.fit_transform(df[c].astype(str))

    return df


@st.cache_data
def run_models(_df):
    FEATURES = [
        "PI_AGE","AGE_SQUARED","AGE_OVER_70",
        "LOG_INCOME","LOG_SUM_ASSURED","INCOME_TO_ASSURED","HAS_INCOME",
        "PI_GENDER_ENC","ZONE_ENC","PAYMENT_MODE_ENC","EARLY_NON_ENC",
        "PI_OCCUPATION_ENC","MEDICAL_NONMED_ENC","PI_STATE_ENC",
        "IS_CREDITOR_ZONE","IS_TEAM_ZONE","IS_SINGLE_PAYMENT","IS_ANNUAL",
        "EARLY_x_MEDICAL","REASON_KNOWN","CARDIAC_CAUSE",
    ]
    X = _df[FEATURES]
    y = _df["TARGET"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc  = sc.transform(X_test)

    models = {
        "KNN":              KNeighborsClassifier(n_neighbors=7),
        "Decision Tree":    DecisionTreeClassifier(max_depth=8, min_samples_split=20, random_state=42),
        "Random Forest":    RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "Gradient Boosted": GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                                        max_depth=4, random_state=42),
    }

    results = {}
    for name, model in models.items():
        use_scaled = (name == "KNN")
        Xtr = X_train_sc if use_scaled else X_train.values
        Xte = X_test_sc  if use_scaled else X_test.values
        model.fit(Xtr, y_train)
        y_pred_tr = model.predict(Xtr)
        y_pred_te = model.predict(Xte)
        y_prob    = model.predict_proba(Xte)[:, 1]
        cm        = confusion_matrix(y_test, y_pred_te)
        tn, fp, fn, tp = cm.ravel()
        total = cm.sum()
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        cv = cross_val_score(model,
                             X_train_sc if use_scaled else X_train.values,
                             y_train, cv=5, scoring="accuracy")
        results[name] = dict(
            model=model, train_acc=accuracy_score(y_train, y_pred_tr),
            test_acc=accuracy_score(y_test, y_pred_te),
            precision=precision_score(y_test, y_pred_te),
            recall=recall_score(y_test, y_pred_te),
            f1=f1_score(y_test, y_pred_te),
            cm=cm, prob=y_prob, fpr=fpr, tpr=tpr, auc=auc(fpr, tpr),
            fp_pct=fp/total*100, fn_pct=fn/total*100,
            tp_pct=tp/total*100, tn_pct=tn/total*100,
            cv_mean=cv.mean(), cv_std=cv.std(),
            report=classification_report(y_test, y_pred_te,
                                         target_names=["Repudiated","Approved"]),
        )
        if hasattr(model, "feature_importances_"):
            results[name]["feat_imp"] = model.feature_importances_

    return results, FEATURES, X_test, y_test


# ─────────────────────────────────────────────
# HELPER PLOTTING FUNCTIONS
# ─────────────────────────────────────────────
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    return buf


COLORS = ["#3498DB","#27AE60","#E67E22","#9B59B6"]
MODEL_NAMES = ["KNN","Decision Tree","Random Forest","Gradient Boosted"]


def plot_crosstabs(df):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Descriptive Analysis – Cross-tabulation vs Policy Status",
                 fontsize=16, fontweight="bold")
    cats = [("PI_GENDER","Gender"), ("EARLY_NON","Early / Non-Early"),
            ("PAYMENT_MODE","Payment Mode"), ("MEDICAL_NONMED","Medical / Non-Medical"),
            ("AGE_GROUP","Age Group"), ("INCOME_GROUP","Income Group")]
    for ax, (col, title) in zip(axes.flat, cats):
        ct = pd.crosstab(df[col], df["POLICY_STATUS"], normalize="index") * 100
        ct.plot(kind="bar", ax=ax, color=["#2196F3","#F44336"], edgecolor="white", width=0.7)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel("% within Group")
        ax.set_ylim(0, 120)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)
        for p in ax.patches:
            if p.get_height() > 4:
                ax.annotate(f"{p.get_height():.1f}%",
                            (p.get_x()+p.get_width()/2, p.get_height()+1),
                            ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    return fig


def plot_zone_bias(df):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Diagnostic Analysis – Team/Zone Bias", fontsize=16, fontweight="bold")
    overall = df["TARGET"].mean() * 100

    zone_stats = (df.groupby("ZONE")
                    .agg(Total=("TARGET","count"), Approved=("TARGET","sum"))
                    .reset_index())
    zone_stats["Rate"] = zone_stats["Approved"] / zone_stats["Total"] * 100
    zone_stats = zone_stats[zone_stats["Total"] >= 5].sort_values("Rate")
    colors = ["#F44336" if r < overall-5 else "#4CAF50" if r > overall+5 else "#FF9800"
              for r in zone_stats["Rate"]]
    axes[0].barh(zone_stats["ZONE"], zone_stats["Rate"], color=colors, edgecolor="white")
    axes[0].axvline(overall, color="navy", ls="--", lw=2, label=f"Overall {overall:.1f}%")
    axes[0].legend(); axes[0].set_xlabel("Approval Rate (%)")
    ct = pd.crosstab(df["ZONE"], df["TARGET"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    axes[0].set_title(f"Approval Rate by Zone/Team\nChi² p={p:.5f}  ← {'SIGNIFICANT BIAS' if p<0.05 else 'No bias'}",
                      fontweight="bold", color="red" if p<0.05 else "black")
    for i, (r, tot) in enumerate(zip(zone_stats["Rate"], zone_stats["Total"])):
        axes[0].text(r+0.5, i, f"{r:.1f}% (n={tot})", va="center", fontsize=7.5)

    state_stats = (df.groupby("PI_STATE")
                     .agg(Total=("TARGET","count"), Approved=("TARGET","sum"))
                     .reset_index())
    state_stats["Rate"] = state_stats["Approved"] / state_stats["Total"] * 100
    state_stats = (state_stats[state_stats["Total"] >= 10]
                   .sort_values("Rate", ascending=False).head(20))
    sns.barplot(data=state_stats, x="Rate", y="PI_STATE", ax=axes[1],
                palette=sns.color_palette("RdYlGn", len(state_stats)))
    axes[1].axvline(overall, color="navy", ls="--", lw=2, label=f"Overall {overall:.1f}%")
    axes[1].set_title("Approval Rate by State (min n=10)", fontweight="bold")
    axes[1].set_xlabel("Approval Rate (%)"); axes[1].legend()
    plt.tight_layout()
    return fig


def plot_age_income_bias(df):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Diagnostic Analysis – Age & Income Bias", fontsize=16, fontweight="bold")
    overall = df["TARGET"].mean() * 100

    approved   = df[df["TARGET"]==1]["PI_AGE"]
    repudiated = df[df["TARGET"]==0]["PI_AGE"]
    t, p_age   = stats.ttest_ind(approved, repudiated)
    axes[0,0].hist([approved, repudiated], bins=20,
                   label=["Approved","Repudiated"], color=["#2196F3","#F44336"],
                   alpha=0.7, edgecolor="white")
    axes[0,0].set_title(f"Age Distribution by Claim Status\nt-test p={p_age:.4f}",
                        fontweight="bold")
    axes[0,0].set_xlabel("Age"); axes[0,0].legend()

    age_rate  = df.groupby("AGE_GROUP")["TARGET"].mean() * 100
    age_count = df.groupby("AGE_GROUP")["TARGET"].count()
    bar_c = ["#F44336" if r < overall-5 else "#4CAF50" if r > overall+5 else "#FF9800"
             for r in age_rate]
    axes[0,1].bar(age_rate.index.astype(str), age_rate.values, color=bar_c, edgecolor="white")
    axes[0,1].axhline(overall, color="navy", ls="--", lw=2, label=f"Overall {overall:.1f}%")
    axes[0,1].set_title("Approval Rate by Age Group", fontweight="bold")
    axes[0,1].set_ylabel("%"); axes[0,1].legend()
    for i,(ag,r) in enumerate(zip(age_rate.index, age_rate.values)):
        axes[0,1].text(i, r+1.5, f"{r:.1f}%\n(n={age_count[ag]})", ha="center", fontsize=9)

    df_inc = df[df["PI_ANNUAL_INCOME"] > 0]
    axes[1,0].hist([df_inc[df_inc["TARGET"]==1]["PI_ANNUAL_INCOME"],
                    df_inc[df_inc["TARGET"]==0]["PI_ANNUAL_INCOME"]],
                   bins=25, label=["Approved","Repudiated"],
                   color=["#2196F3","#F44336"], alpha=0.7, edgecolor="white")
    axes[1,0].set_xlim(0, 2e6)
    axes[1,0].set_title("Income Distribution (excl. zero)", fontweight="bold")
    axes[1,0].set_xlabel("Annual Income"); axes[1,0].legend()

    inc_rate = df.groupby("INCOME_GROUP")["TARGET"].mean() * 100
    inc_cnt  = df.groupby("INCOME_GROUP")["TARGET"].count()
    bar_c2 = ["#F44336" if r < overall-5 else "#4CAF50" if r > overall+5 else "#FF9800"
              for r in inc_rate]
    axes[1,1].bar(inc_rate.index.astype(str), inc_rate.values, color=bar_c2, edgecolor="white")
    axes[1,1].axhline(overall, color="navy", ls="--", lw=2, label=f"Overall {overall:.1f}%")
    axes[1,1].set_title("Approval Rate by Income Group", fontweight="bold")
    axes[1,1].set_ylabel("%"); axes[1,1].legend()
    for i,(ig,r) in enumerate(zip(inc_rate.index, inc_rate.values)):
        axes[1,1].text(i, r+1.5, f"{r:.1f}%\n(n={inc_cnt[ig]})", ha="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_bias_heatmaps(df):
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Diagnostic Heatmaps – Interaction Effects", fontsize=16, fontweight="bold")
    for ax, (r,c,t) in zip(axes, [
        ("PI_GENDER","EARLY_NON","Gender × Early/Non-Early"),
        ("ZONE","MEDICAL_NONMED","Zone × Medical Examination"),
        ("PAYMENT_MODE","PI_GENDER","Payment Mode × Gender"),
    ]):
        pivot = df.pivot_table(index=r, columns=c, values="TARGET", aggfunc="mean") * 100
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn",
                    vmin=0, vmax=100, linewidths=1)
        ax.set_title(f"Approval Rate (%) – {t}", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_model_performance(results):
    sns.set_theme(style="whitegrid")
    names = MODEL_NAMES
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Model Performance – Training vs Testing", fontsize=15, fontweight="bold")
    x = np.arange(len(names)); w = 0.35
    tr_acc = [results[n]["train_acc"]*100 for n in names]
    te_acc = [results[n]["test_acc"]*100  for n in names]
    b1 = axes[0].bar(x-w/2, tr_acc, w, label="Training", color="#3498DB", edgecolor="white")
    b2 = axes[0].bar(x+w/2, te_acc, w, label="Testing",  color="#E74C3C", edgecolor="white")
    axes[0].set_ylim(0,115); axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right"); axes[0].legend()
    axes[0].set_title("Accuracy: Training vs Testing", fontweight="bold")
    for b in list(b1)+list(b2):
        axes[0].annotate(f"{b.get_height():.1f}%",
                         (b.get_x()+b.get_width()/2, b.get_height()+0.5),
                         ha="center", fontsize=8)
    x2 = np.arange(len(names)); w2 = 0.22
    for i,(m,c) in enumerate(zip(["precision","recall","f1"],["#2ECC71","#E74C3C","#3498DB"])):
        vals = [results[n][m]*100 for n in names]
        axes[1].bar(x2+(i-1)*w2, vals, w2, label=m.capitalize(), color=c, edgecolor="white")
    axes[1].set_ylim(0,115); axes[1].set_xticks(x2)
    axes[1].set_xticklabels(names, rotation=20, ha="right"); axes[1].legend()
    axes[1].set_title("Precision / Recall / F1-Score", fontweight="bold"); axes[1].set_ylabel("%")
    cv_m = [results[n]["cv_mean"]*100 for n in names]
    cv_s = [results[n]["cv_std"]*100  for n in names]
    axes[2].bar(x, cv_m, color=COLORS, edgecolor="white", yerr=cv_s, capsize=5)
    axes[2].set_ylim(0,115); axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=20, ha="right")
    axes[2].set_title("5-Fold CV Accuracy (Mean ± Std)", fontweight="bold")
    for i,(m,s) in enumerate(zip(cv_m,cv_s)):
        axes[2].text(i, m+s+1.5, f"{m:.1f}±{s:.1f}%", ha="center", fontsize=8)
    plt.tight_layout(); return fig


def plot_roc(results):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0,1],[0,1],"k--", lw=1, label="Random Classifier")
    for name, c in zip(MODEL_NAMES, COLORS):
        ax.plot(results[name]["fpr"], results[name]["tpr"], color=c, lw=2.5,
                label=f"{name} (AUC={results[name]['auc']:.3f})")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves – All Models", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    plt.tight_layout(); return fig


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Confusion Matrices – FP% and FN% Contribution", fontsize=15, fontweight="bold")
    for ax, (name, c) in zip(axes.flat, zip(MODEL_NAMES, COLORS)):
        cm = results[name]["cm"]
        total = cm.sum()
        sns.heatmap(cm, annot=False, ax=ax, cmap="Blues",
                    xticklabels=["Repudiated","Approved"],
                    yticklabels=["Repudiated","Approved"],
                    linewidths=2, linecolor="white")
        for (i,j), val in np.ndenumerate(cm):
            pct = val/total*100
            ax.text(j+0.5, i+0.5, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if (i==1 and j==1) else "black")
        ax.set_title(
            f"{name}\n"
            f"Acc={results[name]['test_acc']*100:.1f}%  |  "
            f"FP={results[name]['fp_pct']:.1f}%  |  "
            f"FN={results[name]['fn_pct']:.1f}%",
            fontweight="bold", fontsize=11)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); return fig


def plot_fp_fn_breakdown(results):
    fig, ax = plt.subplots(figsize=(13, 6))
    names = MODEL_NAMES
    x = np.arange(len(names)); w = 0.22
    for i,(key,label,c) in enumerate(zip(
        ["tp_pct","tn_pct","fp_pct","fn_pct"],
        ["TP – Correct Approval","TN – Correct Repudiation",
         "FP – Wrong Approval (Financial Risk)","FN – Wrong Denial (Fairness Risk)"],
        ["#2ECC71","#3498DB","#E67E22","#E74C3C"]
    )):
        vals = [results[n][key] for n in names]
        bars = ax.bar(x+(i-1.5)*w, vals, w, label=label, color=c, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                    f"{bar.get_height():.1f}%", ha="center", fontsize=7.5, rotation=90)
    ax.set_ylabel("% of Total Test Samples", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("TP / TN / FP / FN Percentage Breakdown per Model", fontweight="bold", fontsize=13)
    plt.tight_layout(); return fig


def plot_feature_importance(results, features):
    feat_labels = [
        "Age","Age²","Age>70","Log Income","Log Sum Assured","Income/Assured",
        "Has Income","Gender","Zone","Payment Mode","Early/Non","Occupation",
        "Medical/Non","State","Creditor Zone","Team Zone","Single Pay",
        "Annual Pay","Early×Medical","Reason Known","Cardiac Cause"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Feature Importance – Random Forest & Gradient Boosted", fontsize=15, fontweight="bold")
    for ax, model_name in zip(axes, ["Random Forest","Gradient Boosted"]):
        imp = results[model_name]["feat_imp"]
        idx = np.argsort(imp)
        labels_s = [feat_labels[i] for i in idx]
        cols = ["#E74C3C" if imp[i] > np.percentile(imp,75) else "#3498DB" for i in idx]
        ax.barh(labels_s, imp[idx], color=cols, edgecolor="white")
        ax.set_title(model_name, fontweight="bold"); ax.set_xlabel("Importance")
        for v, l in zip(imp[idx], labels_s):
            ax.text(v+0.001, labels_s.index(l), f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout(); return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Insurance Bias Detector")
    st.markdown("---")
    uploaded = st.file_uploader("📂 Upload Insurance CSV", type=["csv"])
    st.markdown("---")
    st.markdown("### Navigation")
    sections = [
        "🏠 Overview",
        "📊 Descriptive Analysis",
        "🔬 Diagnostic Analysis",
        "🤖 ML Models & Feature Eng.",
        "📈 Model Evaluation",
        "📋 Findings & Recommendations",
    ]
    section = st.radio("", sections, label_visibility="collapsed")
    st.markdown("---")
    st.info("Upload the **Insurance.csv** file to begin analysis.")

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">🔍 Insurance Claim Settlement Bias Analysis</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Settlement Office – Fairness & Compliance Dashboard</div>',
            unsafe_allow_html=True)

if uploaded is None:
    st.warning("⬅️ Please upload the **Insurance.csv** file from the sidebar to begin.")
    st.markdown("""
    ### What this dashboard analyses:
    1. **Descriptive Analysis** – Cross-tabulation of key variables vs Policy Status
    2. **Diagnostic Analysis** – Deep-dive into bias by team, age, income, and payment mode
    3. **Feature Engineering + ML Models** – KNN, Decision Tree, Random Forest, Gradient Boosted
    4. **Model Evaluation** – Accuracy, Precision, Recall, F1, ROC, Confusion Matrix with FP/FN %
    5. **Findings & Recommendations** – Actionable insights for fair claim settlement
    """)
    st.stop()

# Load data
with st.spinner("Loading and engineering features…"):
    df = load_and_engineer(uploaded)

# Run models
with st.spinner("Training ML models (this may take ~30 seconds)…"):
    results, FEATURES, X_test, y_test = run_models(df)

overall_rate = df["TARGET"].mean() * 100

# ═══════════════════════════════════════════════
# SECTION 0: OVERVIEW
# ═══════════════════════════════════════════════
if section == "🏠 Overview":
    st.subheader("📋 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Claims", f"{len(df):,}")
    c2.metric("Approved", f"{df['TARGET'].sum():,}", f"{overall_rate:.1f}%")
    c3.metric("Repudiated", f"{(df['TARGET']==0).sum():,}", f"{100-overall_rate:.1f}%")
    c4.metric("Unique Zones/Teams", f"{df['ZONE'].nunique()}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Sample Data")
        st.dataframe(df[["POLICY_NO","PI_GENDER","PI_AGE","ZONE","PAYMENT_MODE",
                          "EARLY_NON","PI_ANNUAL_INCOME","SUM_ASSURED","POLICY_STATUS"]].head(15),
                     use_container_width=True)
    with c2:
        st.markdown("#### Missing Values")
        mv = df.isnull().sum()
        mv = mv[mv > 0]
        if len(mv):
            st.dataframe(mv.reset_index().rename(columns={"index":"Column", 0:"Missing"}))
        else:
            st.success("No missing values after cleaning!")

        st.markdown("#### Class Distribution")
        vc = df["POLICY_STATUS"].value_counts()
        fig_pie, ax_p = plt.subplots(figsize=(5,4))
        ax_p.pie(vc.values, labels=vc.index, autopct="%1.1f%%",
                 colors=["#3498DB","#E74C3C"], startangle=140)
        ax_p.set_title("Policy Status Distribution", fontweight="bold")
        st.pyplot(fig_pie); plt.close()


# ═══════════════════════════════════════════════
# SECTION 1: DESCRIPTIVE ANALYSIS
# ═══════════════════════════════════════════════
elif section == "📊 Descriptive Analysis":
    st.subheader("📊 Descriptive Analysis – Cross-tabulation vs Policy Status")
    st.info("Compares approval rates across Gender, Payment Mode, Medical Exam, Early/Non-Early, Age Group, and Income Group.")

    fig = plot_crosstabs(df)
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("📋 Detailed Cross-tabulations")
    col_sel = st.selectbox("Select variable:", ["PI_GENDER","EARLY_NON","PAYMENT_MODE","MEDICAL_NONMED","AGE_GROUP","INCOME_GROUP"])
    ct = pd.crosstab(df[col_sel], df["POLICY_STATUS"], margins=True)
    ct["Approval Rate %"] = (ct.get("Approved Death Claim", 0) /
                              (ct.get("Approved Death Claim",0) + ct.get("Repudiate Death",0)) * 100).round(1)
    st.dataframe(ct.style.background_gradient(subset=["Approval Rate %"], cmap="RdYlGn"), use_container_width=True)

    # Chi-squared test
    ct_raw = pd.crosstab(df[col_sel], df["TARGET"])
    chi2, p, dof, _ = stats.chi2_contingency(ct_raw)
    if p < 0.05:
        st.markdown(f'<div class="bias-alert">⚠️ <b>Chi² Test:</b> χ²={chi2:.2f}, p={p:.5f} — '
                    f'<b>Statistically significant association</b> between {col_sel} and policy status (p < 0.05)</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="good-box">✅ <b>Chi² Test:</b> χ²={chi2:.2f}, p={p:.5f} — '
                    f'No statistically significant association detected (p ≥ 0.05)</div>',
                    unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# SECTION 2: DIAGNOSTIC ANALYSIS
# ═══════════════════════════════════════════════
elif section == "🔬 Diagnostic Analysis":
    st.subheader("🔬 Diagnostic Analysis – Bias Detection")

    tab1, tab2, tab3, tab4 = st.tabs(["🏢 Team/Zone Bias","👥 Age Bias","💰 Income Bias","🔥 Interaction Effects"])

    with tab1:
        st.markdown("#### Team & Zone Approval Rates")
        fig = plot_zone_bias(df)
        st.pyplot(fig); plt.close()

        # Highlight top/bottom zones
        zone_tbl = (df.groupby("ZONE")["TARGET"]
                      .agg(["count","mean"])
                      .rename(columns={"count":"Total","mean":"Approval Rate"})
                      .reset_index())
        zone_tbl["Approval Rate"] = (zone_tbl["Approval Rate"] * 100).round(1)
        zone_tbl = zone_tbl[zone_tbl["Total"] >= 10].sort_values("Approval Rate")
        st.markdown("**Zones with ≥10 claims, sorted by approval rate:**")
        st.dataframe(
            zone_tbl.style.background_gradient(subset=["Approval Rate"], cmap="RdYlGn"),
            use_container_width=True
        )
        st.markdown(
            '<div class="bias-alert">🚨 <b>Critical Finding:</b> Approval rates vary from 23% to 97% '
            'across zones — a 74 percentage-point spread — suggesting strong team-level settlement bias.</div>',
            unsafe_allow_html=True
        )

    with tab2:
        st.markdown("#### Age-wise Bias Analysis")
        fig = plot_age_income_bias(df)
        st.pyplot(fig); plt.close()
        approved   = df[df["TARGET"]==1]["PI_AGE"]
        repudiated = df[df["TARGET"]==0]["PI_AGE"]
        t, p_age   = stats.ttest_ind(approved, repudiated)
        st.markdown(f"**Mean age – Approved: {approved.mean():.1f} | Repudiated: {repudiated.mean():.1f}**")
        if p_age < 0.05:
            st.markdown(f'<div class="bias-alert">⚠️ t-test p={p_age:.4f}: Age is statistically different '
                        f'between approved and repudiated claims.</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown("#### Income-wise Bias Analysis")
        inc_tbl = (df.groupby("INCOME_GROUP")["TARGET"]
                     .agg(["count","mean"])
                     .rename(columns={"count":"Total","mean":"Approval Rate"})
                     .reset_index())
        inc_tbl["Approval Rate"] = (inc_tbl["Approval Rate"] * 100).round(1)
        st.dataframe(inc_tbl.style.background_gradient(subset=["Approval Rate"], cmap="RdYlGn"),
                     use_container_width=True)
        st.markdown(
            '<div class="finding-box">💡 <b>Income Note:</b> Many elderly/retired policyholders '
            'report zero income. This group shows different settlement patterns from income-earners.</div>',
            unsafe_allow_html=True
        )

    with tab4:
        st.markdown("#### Interaction Heatmaps")
        fig = plot_bias_heatmaps(df)
        st.pyplot(fig); plt.close()
        st.markdown(
            '<div class="bias-alert">🔥 <b>Critical:</b> The Zone × Medical heatmap reveals '
            'that the same medical examination status leads to dramatically different approval '
            'rates across teams — confirming team-level bias.</div>',
            unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════
# SECTION 3: ML MODELS
# ═══════════════════════════════════════════════
elif section == "🤖 ML Models & Feature Eng.":
    st.subheader("🤖 ML Models & Feature Engineering")

    st.markdown("### Feature Engineering Applied")
    feat_table = pd.DataFrame({
        "Feature": ["Age²","Age > 70 Flag","Log Income","Log Sum Assured",
                    "Income/Sum-Assured Ratio","Has Income Flag","Creditor Zone Flag",
                    "Team Zone Flag","Single Payment Flag","Early × Medical Interaction",
                    "Reason Known Flag","Cardiac Cause Flag"],
        "Description": [
            "Non-linear age effect","Binary – senior citizen flag",
            "Log-transformed income (right-skewed)","Log-transformed sum assured",
            "Affordability proxy","1 if annual income > 0, else 0",
            "1 if zone is a creditor/bank zone","1 if zone starts with TEAM",
            "1 if payment mode is Single lump-sum","Interaction of EARLY × MEDICAL examination",
            "1 if reason for claim is provided","1 if cause is heart-related",
        ],
        "Motivation": [
            "Mortality risk is non-linear","High-risk age threshold",
            "Removes scale effect","Removes scale effect",
            "Policy over-insurance detection","Separates working vs retired",
            "Bank-tied creditor policies have higher approval","Team zones show high approval",
            "Single payers get 89.9% approval rate","Combined risk factor",
            "Unexplained claims get repudiated more","Cardiac is most common claim cause",
        ]
    })
    st.dataframe(feat_table, use_container_width=True)

    st.markdown("---")
    st.markdown("### Model Summary")
    summary_data = {
        "Model": MODEL_NAMES,
        "Train Acc %": [f"{results[n]['train_acc']*100:.1f}" for n in MODEL_NAMES],
        "Test Acc %":  [f"{results[n]['test_acc']*100:.1f}"  for n in MODEL_NAMES],
        "Precision %": [f"{results[n]['precision']*100:.1f}" for n in MODEL_NAMES],
        "Recall %":    [f"{results[n]['recall']*100:.1f}"    for n in MODEL_NAMES],
        "F1 %":        [f"{results[n]['f1']*100:.1f}"        for n in MODEL_NAMES],
        "AUC":         [f"{results[n]['auc']:.3f}"           for n in MODEL_NAMES],
        "CV Mean %":   [f"{results[n]['cv_mean']*100:.1f}±{results[n]['cv_std']*100:.1f}" for n in MODEL_NAMES],
        "FP %":        [f"{results[n]['fp_pct']:.1f}"        for n in MODEL_NAMES],
        "FN %":        [f"{results[n]['fn_pct']:.1f}"        for n in MODEL_NAMES],
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    st.markdown("---")
    st.markdown("### Full Classification Reports")
    for name in MODEL_NAMES:
        with st.expander(f"📄 {name}"):
            st.code(results[name]["report"])


# ═══════════════════════════════════════════════
# SECTION 4: MODEL EVALUATION
# ═══════════════════════════════════════════════
elif section == "📈 Model Evaluation":
    st.subheader("📈 Model Evaluation – Plots")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Accuracy","📉 ROC Curves","🧮 Confusion Matrices","📊 FP/FN Breakdown","🌟 Feature Importance"
    ])

    with tab1:
        fig = plot_model_performance(results)
        st.pyplot(fig); plt.close()

    with tab2:
        fig = plot_roc(results)
        st.pyplot(fig); plt.close()
        st.markdown("**AUC Scores:**")
        for n, c in zip(MODEL_NAMES, COLORS):
            st.markdown(f"- **{n}**: AUC = `{results[n]['auc']:.3f}`")

    with tab3:
        fig = plot_confusion_matrices(results)
        st.pyplot(fig); plt.close()
        st.markdown("""
        **Reading the Confusion Matrix:**
        - **TP** (top-left inside "Approved" block): Correct approvals
        - **TN** (top-left inside "Repudiated" block): Correct repudiations
        - **FP** ⚠️: Claim was actually repudiated but model predicted Approved → **Financial risk**
        - **FN** ⚠️: Claim was actually approved but model predicted Repudiated → **Fairness risk**
        """)

    with tab4:
        fig = plot_fp_fn_breakdown(results)
        st.pyplot(fig); plt.close()

        st.markdown("### FP & FN Percentage Contribution Table")
        tbl = pd.DataFrame({
            "Model": MODEL_NAMES,
            "TP % (Correct Approval)":        [f"{results[n]['tp_pct']:.2f}%" for n in MODEL_NAMES],
            "TN % (Correct Repudiation)":      [f"{results[n]['tn_pct']:.2f}%" for n in MODEL_NAMES],
            "FP % (Wrong Approval – FinRisk)": [f"{results[n]['fp_pct']:.2f}%" for n in MODEL_NAMES],
            "FN % (Wrong Denial – FairnessRisk)": [f"{results[n]['fn_pct']:.2f}%" for n in MODEL_NAMES],
        })
        st.dataframe(tbl, use_container_width=True)
        st.markdown(
            '<div class="finding-box">💡 <b>Gradient Boosted</b> achieves the lowest FP% (15.4%) — '
            'best for financial risk control. <b>Random Forest</b> achieves the lowest FN% (5.6%) — '
            'best for ensuring fair claim approvals.</div>',
            unsafe_allow_html=True
        )

    with tab5:
        fig = plot_feature_importance(results, FEATURES)
        st.pyplot(fig); plt.close()
        st.markdown(
            '<div class="bias-alert">🔍 <b>Zone/Team</b> ranks as a top predictor in both tree-based models, '
            'confirming that team assignment significantly influences claim outcomes — '
            'a marker of institutional bias.</div>',
            unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════
# SECTION 5: FINDINGS
# ═══════════════════════════════════════════════
elif section == "📋 Findings & Recommendations":
    st.subheader("📋 Key Findings & Recommendations")

    st.markdown("### 🔴 Bias Findings")
    findings = [
        ("🏢 Team/Zone Bias (CRITICAL)", "Approval rates span 23% to 97% across teams (Chi² p<0.0001). "
         "PENINSULAR zone approves only 23% of claims while JKB CREDITOR approves 97%. "
         "This 74pp spread is the strongest evidence of structural bias."),
        ("💳 Payment Mode Bias (SIGNIFICANT)", "Single-payment policyholders are approved at 89.9% vs "
         "45.0% for Quarterly payers — a 44.9pp gap with no actuarial justification."),
        ("🩺 Medical Exam Bias", "MEDICAL examination policies are approved at 81.1% vs 66.4% for "
         "NON-MEDICAL — a 14.7pp gap, potentially justified by risk assessment but warrants review."),
        ("⏱️ Early/Non-Early Bias", "EARLY policies (within 2 years of issue) approved at 77.0% vs "
         "62.8% for NON-EARLY — a 14.2pp gap that contradicts typical anti-selection concerns."),
        ("👴 Age Effect", "Age shows a statistically significant difference (t-test p<0.05) between "
         "approved and repudiated claimants, with 60-70 year olds seeing higher repudiation than expected."),
    ]
    for title, desc in findings:
        st.markdown(f'<div class="bias-alert"><b>{title}</b><br>{desc}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ✅ Model Recommendations")
    rec = [
        ("Best Overall Model", "**Gradient Boosted** — highest test accuracy (75.2%), AUC (0.790), "
         "and lowest FP% (15.4%)"),
        ("Lowest FN (Fair Approval)", "**Random Forest** — lowest FN% (5.6%) ensures fewest wrongful denials"),
        ("Most Interpretable", "**Decision Tree** — audit-friendly, single-path rules for each claim"),
    ]
    for title, desc in rec:
        st.markdown(f'<div class="good-box"><b>{title}:</b> {desc}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📌 Action Items for Management")
    actions = [
        "🔍 **Audit PENINSULAR & JKB JAMMU teams** — investigate why their approval rates (23–43%) are far below average.",
        "📋 **Standardise settlement guidelines** across all zones to eliminate zone-level discretion.",
        "💳 **Review payment-mode correlation** — Single payers should not receive preferential treatment.",
        "👩‍⚖️ **Implement blind review** — remove team/zone identity from initial claim assessment.",
        "🤖 **Deploy Gradient Boosted model** as a decision-support tool to flag outlier decisions.",
        "📊 **Monthly bias report** — track approval rates by team, payment mode, and medical exam status.",
        "🧾 **Mandate reason documentation** — currently 381 claims have no stated reason (21% of data).",
    ]
    for a in actions:
        st.markdown(a)

    st.markdown("---")
    with st.expander("📄 Download Full Findings as Text"):
        report_text = """
INSURANCE CLAIM SETTLEMENT BIAS ANALYSIS – FINDINGS REPORT
===========================================================

1. DATASET: 1,790 death claims | 68.0% Approved | 32.0% Repudiated

2. CRITICAL BIAS – TEAM/ZONE (Chi² p < 0.0001)
   • PENINSULAR: 23.1% approval (n=13)
   • JKB JAMMU:  43.5% approval (n=62)
   • JKB CREDITOR: 96.6% approval (n=58)
   → 74 percentage-point range across teams

3. PAYMENT MODE BIAS
   • Single payment: 89.9% approval
   • Quarterly:      45.0% approval
   → 44.9pp gap — no actuarial justification

4. MEDICAL EXAM: 81.1% (MEDICAL) vs 66.4% (NON-MEDICAL) — 14.7pp

5. EARLY POLICY: 77.0% (EARLY) vs 62.8% (NON-EARLY) — 14.2pp

6. ML MODELS:
   • KNN:              Test=67.9%, F1=78.3%, AUC=0.655
   • Decision Tree:    Test=73.9%, F1=81.7%, AUC=0.762
   • Random Forest:    Test=73.2%, F1=82.4%, AUC=0.776, FN=5.6% (best)
   • Gradient Boosted: Test=75.2%, F1=82.6%, AUC=0.790, FP=15.4% (best)

7. RECOMMENDED MODEL: Gradient Boosted (best balanced performance)

8. KEY ACTIONS:
   a) Audit PENINSULAR and South zones immediately
   b) Standardise cross-team settlement guidelines
   c) Implement blind claim review process
   d) Deploy GB model as decision-support tool
   e) Monthly bias monitoring reports
        """
        st.download_button("⬇️ Download Report", report_text, "bias_findings.txt", "text/plain")
