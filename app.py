"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║     INTERACTIVE MACHINE LEARNING CLASSIFICATION DASHBOARD                 ║
║     Version: 3.0 - Enhanced Visualizations                                ║
║     Author: ML Dashboard Development Team                                 ║
║     Date: January 2026                                                    ║
║                                                                            ║
║  A professional-grade end-to-end ML pipeline with:                        ║
║  • Dataset Management & Exploration                                       ║
║  • Multi-Algorithm Configuration with Training                            ║
║  • Advanced Visualizations & Metrics                                      ║
║  • Interactive Dashboard                                                  ║
║  • Single & Batch Predictions                                             ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ML Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stMetric { background-color: #f9fafb; padding: 1.5rem; border-radius: 0.75rem; border-left: 4px solid #2563eb; }
    .model-card { background-color: #f0f9ff; padding: 1rem; border-radius: 0.5rem; border: 2px solid #2563eb; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
for key, value in {
    'df': None, 'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
    'models': {}, 'predictions': {}, 'metrics': {}, 'feature_importance': {},
    'scaler': None, 'features': [], 'target': None,
    'knn_config': {'k': 5, 'metric': 'euclidean', 'weights': 'uniform'},
    'rf_config': {'max_depth': 10, 'criterion': 'gini', 'n_estimators': 100},
    'nb_config': {'type': 'Gaussian', 'alpha': 1.0},
    'dt_config': {'max_depth': 5, 'criterion': 'gini', 'min_samples': 2},
    'selected_algorithms': {'kNN': True, 'Random Forest': True, 'Naive Bayes': True, 'Decision Tree': True},
    'cross_val_scores': {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_sample_dataset(dataset_name: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        if dataset_name == 'Iris':
            from sklearn.datasets import load_iris
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df['target'] = iris.target
            return df, 'target'
        elif dataset_name == 'Breast Cancer':
            from sklearn.datasets import load_breast_cancer
            cancer = load_breast_cancer()
            df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
            df['target'] = cancer.target
            return df, 'target'
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler

def calculate_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                   color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_pred_proba, model_name):
    if len(np.unique(y_true)) != 2:
        return None, None
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#2563eb', lw=2.5, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#9ca3af', lw=1.5, linestyle='--')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#2563eb')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, roc_auc

def plot_feature_importance(model, feature_names, model_name):
    if not hasattr(model, 'feature_importances_'):
        return None, None
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(importance_df)))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors, edgecolor='black')
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title(f'Top 10 Features - {model_name}', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig, importance_df

def train_models():
    """Train selected models"""
    progress = st.progress(0)
    status = st.empty()
    
    try:
        y = st.session_state.df[st.session_state.target]
        if len(y.unique()) > 20:
            st.error("❌ Target must be classification (≤20 classes)")
            st.stop()
        if y.dtype == 'float':
            y = y.astype(int)
        
        status.text("📊 Preparing data...")
        X = st.session_state.df[st.session_state.features].fillna(
            st.session_state.df[st.session_state.features].mean(numeric_only=True))
        
        progress.progress(10)
        status.text("✂️ Splitting data...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42)
            st.warning("⚠️ Random split used")
        
        status.text("📈 Normalizing...")
        X_train, X_test, scaler = scale_features(X_train, X_test)
        st.session_state.scaler = scaler
        st.session_state.X_train, st.session_state.X_test = X_train, X_test
        st.session_state.y_train, st.session_state.y_test = y_train, y_test
        
        models, predictions, metrics, cv_scores = {}, {}, {}, {}
        model_idx = 0
        steps = [25, 40, 55, 70]
        
        if st.session_state.selected_algorithms['kNN']:
            progress.progress(steps[model_idx])
            status.text("🔷 Training kNN...")
            knn = KNeighborsClassifier(n_neighbors=st.session_state.knn_config['k'], 
                                      metric=st.session_state.knn_config['metric'])
            knn.fit(X_train, y_train)
            models['kNN'] = knn
            y_pred = knn.predict(X_test)
            predictions['kNN'] = y_pred
            metrics['kNN'] = calculate_metrics(y_test, y_pred)
            cv_scores['kNN'] = cross_val_score(knn, X_train, y_train, cv=5).mean()
            model_idx += 1
        
        if st.session_state.selected_algorithms['Random Forest']:
            progress.progress(steps[model_idx])
            status.text("🌳 Training RF...")
            rf = RandomForestClassifier(max_depth=st.session_state.rf_config['max_depth'],
                                       criterion=st.session_state.rf_config['criterion'],
                                       n_estimators=st.session_state.rf_config['n_estimators'],
                                       random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            models['Random Forest'] = rf
            y_pred = rf.predict(X_test)
            predictions['Random Forest'] = y_pred
            metrics['Random Forest'] = calculate_metrics(y_test, y_pred)
            cv_scores['Random Forest'] = cross_val_score(rf, X_train, y_train, cv=5).mean()
            st.session_state.feature_importance['Random Forest'] = dict(
                zip(st.session_state.features, rf.feature_importances_))
            model_idx += 1
        
        if st.session_state.selected_algorithms['Naive Bayes']:
            progress.progress(steps[model_idx])
            status.text("🎲 Training NB...")
            if st.session_state.nb_config['type'] == 'Gaussian':
                nb = GaussianNB(var_smoothing=st.session_state.nb_config['alpha'])
            elif st.session_state.nb_config['type'] == 'Multinomial':
                nb = MultinomialNB(alpha=st.session_state.nb_config['alpha'])
            else:
                nb = BernoulliNB(alpha=st.session_state.nb_config['alpha'])
            nb.fit(X_train, y_train)
            models['Naive Bayes'] = nb
            y_pred = nb.predict(X_test)
            predictions['Naive Bayes'] = y_pred
            metrics['Naive Bayes'] = calculate_metrics(y_test, y_pred)
            cv_scores['Naive Bayes'] = cross_val_score(nb, X_train, y_train, cv=5).mean()
            model_idx += 1
        
        if st.session_state.selected_algorithms['Decision Tree']:
            progress.progress(steps[model_idx])
            status.text("🌲 Training DT...")
            dt = DecisionTreeClassifier(max_depth=st.session_state.dt_config['max_depth'],
                                       criterion=st.session_state.dt_config['criterion'],
                                       random_state=42)
            dt.fit(X_train, y_train)
            models['Decision Tree'] = dt
            y_pred = dt.predict(X_test)
            predictions['Decision Tree'] = y_pred
            metrics['Decision Tree'] = calculate_metrics(y_test, y_pred)
            cv_scores['Decision Tree'] = cross_val_score(dt, X_train, y_train, cv=5).mean()
            st.session_state.feature_importance['Decision Tree'] = dict(
                zip(st.session_state.features, dt.feature_importances_))
            model_idx += 1
        
        st.session_state.models = models
        st.session_state.predictions = predictions
        st.session_state.metrics = metrics
        st.session_state.cross_val_scores = cv_scores
        
        progress.progress(100)
        st.success(f"✅ Trained {len(models)} model(s)!")
        return True
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return False

# ============================================================================
# MAIN APP - HEADER
# ============================================================================
st.markdown("<div style='text-align: center; margin-bottom: 2rem;'><h1 style='color: #2563eb;'>📊 ML Classification Dashboard</h1></div>", 
            unsafe_allow_html=True)
st.markdown("---")

# SIDEBAR
with st.sidebar:
    st.markdown("<h2 style='color: #2563eb;'>🧭 Navigation</h2>", unsafe_allow_html=True)
    page = st.radio("Select Module:", ["📁 Dataset", "⚙️ Algorithms", "📈 Visualize", "🔮 Predict"], 
                   label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<h3 style='color: #2563eb;'>📊 Status</h3>", unsafe_allow_html=True)
    if st.session_state.df is not None:
        st.metric("Dataset", f"{len(st.session_state.df)} rows")
    if st.session_state.features:
        st.metric("Features", len(st.session_state.features))
    if st.session_state.models:
        st.metric("Models", len(st.session_state.models))
        if st.session_state.metrics:
            best = max([m.get('Accuracy', 0) for m in st.session_state.metrics.values()])
            st.metric("Best Acc", f"{best:.1%}")

# ============================================================================
# PAGE 1: DATASET
# ============================================================================
if page == "📁 Dataset":
    st.header("📁 Dataset Management")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Data Input")
        input_type = st.radio("Method:", ["📤 Upload", "📚 Sample"], key="input_method")
        
        if input_type == "📤 Upload":
            file = st.file_uploader("Select CSV/Excel:", type=["csv", "xlsx", "xls"])
            if file:
                try:
                    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
                    st.session_state.df = df
                    st.success("✅ Loaded!")
                except Exception as e:
                    st.error(f"❌ {e}")
        else:
            sample = st.selectbox("Dataset:", ['Iris', 'Breast Cancer'])
            if st.button("Load", key="load_btn"):
                df, target = load_sample_dataset(sample)
                if df is not None:
                    st.session_state.df = df
                    st.session_state.target = target
                    st.success(f"✅ {sample}!")
    
    with col2:
        if st.session_state.df is not None:
            st.subheader("📊 Overview")
            st.metric("Rows", len(st.session_state.df))
            st.metric("Columns", st.session_state.df.shape[1])
            st.metric("Missing", st.session_state.df.isnull().sum().sum())
    
    if st.session_state.df is not None:
        st.markdown("---")
        st.subheader("📋 Data Exploration")
        
        tab1, tab2, tab3 = st.tabs(["Preview", "Statistics", "Info"])
        with tab1:
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
        with tab2:
            st.dataframe(st.session_state.df.describe(), use_container_width=True)
        with tab3:
            info = pd.DataFrame({'Column': st.session_state.df.columns, 'Type': st.session_state.df.dtypes,
                                'Non-Null': st.session_state.df.notna().sum()})
            st.dataframe(info, use_container_width=True)
        
        st.markdown("---")
        st.subheader("⚙️ Configuration")
        col_f, col_t = st.columns([1, 1])
        
        with col_f:
            features = st.multiselect("Features:", st.session_state.df.columns.tolist(),
                                     default=st.session_state.df.columns.tolist()[:-1] if len(st.session_state.df.columns) > 1 else [],
                                     key="feature_select")
            st.session_state.features = features
        
        with col_t:
            target = st.selectbox("Target:", st.session_state.df.columns.tolist(), key="target_select")
            st.session_state.target = target
        
        if features and target and target not in features:
            st.success(f"✅ Ready: {len(features)} features | Target: {target}")

# ============================================================================
# PAGE 2: ALGORITHMS
# ============================================================================
elif page == "⚙️ Algorithms":
    st.header("⚙️ Algorithm Configuration & Training")
    st.markdown("---")
    
    if st.session_state.df is None:
        st.warning("⚠️ Load dataset first")
    else:
        st.subheader("📋 Select Algorithms")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.session_state.selected_algorithms['kNN'] = st.checkbox("🔷 kNN", 
                value=st.session_state.selected_algorithms['kNN'], key="select_knn")
        with col2:
            st.session_state.selected_algorithms['Random Forest'] = st.checkbox("🌳 RF", 
                value=st.session_state.selected_algorithms['Random Forest'], key="select_rf")
        with col3:
            st.session_state.selected_algorithms['Naive Bayes'] = st.checkbox("🎲 NB", 
                value=st.session_state.selected_algorithms['Naive Bayes'], key="select_nb")
        with col4:
            st.session_state.selected_algorithms['Decision Tree'] = st.checkbox("🌲 DT", 
                value=st.session_state.selected_algorithms['Decision Tree'], key="select_dt")
        
        selected_count = sum(st.session_state.selected_algorithms.values())
        if selected_count == 0:
            st.error("❌ Select at least one algorithm!")
        else:
            st.success(f"✅ {selected_count} algorithm(s) selected")
        
        st.markdown("---")
        st.subheader("⚙️ Algorithm Parameters")
        
        tab1, tab2, tab3, tab4 = st.tabs(["kNN", "RF", "NB", "DT"])
        
        with tab1:
            if st.session_state.selected_algorithms['kNN']:
                st.subheader("k-Nearest Neighbors")
                k = st.slider("k", 1, 20, 5, key="knn_k")
                metric = st.selectbox("Distance", ["euclidean", "manhattan", "minkowski"], key="knn_metric")
                st.session_state.knn_config = {'k': k, 'metric': metric, 'weights': 'uniform'}
                st.info(f"✓ k={k} | {metric}")
            else:
                st.info("Not selected")
        
        with tab2:
            if st.session_state.selected_algorithms['Random Forest']:
                st.subheader("Random Forest")
                depth = st.slider("Max Depth", 1, 20, 10, key="rf_depth")
                criterion = st.selectbox("Criterion", ["gini", "entropy"], key="rf_criterion")
                estimators = st.slider("Trees", 10, 200, 100, 10, key="rf_estimators")
                st.session_state.rf_config = {'max_depth': depth, 'criterion': criterion, 'n_estimators': estimators}
                st.info(f"✓ depth={depth} | {criterion} | trees={estimators}")
            else:
                st.info("Not selected")
        
        with tab3:
            if st.session_state.selected_algorithms['Naive Bayes']:
                st.subheader("Naive Bayes")
                nb_type = st.selectbox("Type", ["Gaussian", "Multinomial", "Bernoulli"], key="nb_type")
                alpha = st.number_input("Alpha", 0.0, 10.0, 1.0, 0.1, key="nb_alpha")
                st.session_state.nb_config = {'type': nb_type, 'alpha': alpha}
                st.info(f"✓ {nb_type} | alpha={alpha}")
            else:
                st.info("Not selected")
        
        with tab4:
            if st.session_state.selected_algorithms['Decision Tree']:
                st.subheader("Decision Tree")
                dt_depth = st.slider("Max Depth", 1, 20, 5, key="dt_depth")
                dt_crit = st.selectbox("Criterion", ["gini", "entropy"], key="dt_criterion")
                st.session_state.dt_config = {'max_depth': dt_depth, 'criterion': dt_crit, 'min_samples': 2}
                st.info(f"✓ depth={dt_depth} | {dt_crit}")
            else:
                st.info("Not selected")
        
        st.markdown("---")
        st.subheader("🚀 Train Model")
        
        if st.button("🎯 Train Selected Models", use_container_width=True, type="primary", key="train_algo_btn"):
            if st.session_state.df is None or not st.session_state.features or not st.session_state.target:
                st.error("❌ Configure dataset first!")
            elif selected_count == 0:
                st.error("❌ Select at least one algorithm!")
            else:
                if train_models():
                    st.balloons()

# ============================================================================
# PAGE 3: VISUALIZE
# ============================================================================
elif page == "📈 Visualize":
    st.header("📈 Advanced Visualizations & Metrics")
    st.markdown("---")
    
    if not st.session_state.models:
        st.warning("⚠️ Train models first from Algorithms page")
    else:
        # Metrics Display
        st.subheader("📊 Model Performance Metrics")
        cols = st.columns(len(st.session_state.models))
        
        for col, (name, metrics) in zip(cols, st.session_state.metrics.items()):
            with col:
                st.metric(name, f"{metrics.get('Accuracy', 0):.1%}")
                st.caption(f"P: {metrics.get('Precision', 0):.1%}\nR: {metrics.get('Recall', 0):.1%}\nF1: {metrics.get('F1-Score', 0):.1%}")
        
        st.markdown("---")
        
        # Accuracy Comparison
        st.subheader("📊 Model Accuracy Comparison")
        comp_data = [{'Algorithm': name, 'Accuracy': m.get('Accuracy', 0)} 
                    for name, m in st.session_state.metrics.items()]
        comp_df = pd.DataFrame(comp_data)
        
        fig = px.bar(comp_df, x='Algorithm', y='Accuracy', color='Algorithm', 
                    title='Accuracy Comparison', height=400,
                    color_discrete_sequence=['#2563eb', '#10b981', '#f59e0b', '#ef4444'])
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        
        # All Metrics Comparison
        st.subheader("📈 Comprehensive Metrics")
        metrics_df = pd.DataFrame([
            {'Algorithm': name, **metrics} 
            for name, metrics in st.session_state.metrics.items()
        ])
        
        fig = px.bar(metrics_df, x='Algorithm', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    barmode='group', height=400, title='All Metrics Comparison')
        st.plotly_chart(fig, use_container_width=True)
        
        # Cross-Validation Scores
        if st.session_state.cross_val_scores:
            st.subheader("🔄 Cross-Validation Scores (5-Fold)")
            cv_data = [{'Algorithm': name, 'CV Score': score} 
                      for name, score in st.session_state.cross_val_scores.items()]
            cv_df = pd.DataFrame(cv_data)
            
            fig = px.bar(cv_df, x='Algorithm', y='CV Score', title='Cross-Validation Performance',
                        height=350, color='Algorithm',
                        color_discrete_sequence=['#2563eb', '#10b981', '#f59e0b', '#ef4444'])
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Confusion Matrices
        st.subheader("🔲 Confusion Matrices")
        cols = st.columns(len(st.session_state.models))
        for col, (name, model) in zip(cols, st.session_state.models.items()):
            with col:
                if name in st.session_state.predictions:
                    y_pred = st.session_state.predictions[name]
                    fig = plot_confusion_matrix(st.session_state.y_test, y_pred, name)
                    st.pyplot(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Feature Importance
        if st.session_state.feature_importance:
            st.subheader("🌟 Feature Importance (Top 10)")
            cols = st.columns(len(st.session_state.feature_importance))
            for col, (name, model) in zip(cols, st.session_state.models.items()):
                if name in st.session_state.feature_importance:
                    with col:
                        fig, _ = plot_feature_importance(model, st.session_state.features, name)
                        if fig:
                            st.pyplot(fig, use_container_width=True)
        
        st.markdown("---")
        
        # ROC Curves
        if len(np.unique(st.session_state.y_test)) == 2:
            st.subheader("📈 ROC Curves")
            cols = st.columns(len(st.session_state.models))
            for col, (name, model) in zip(cols, st.session_state.models.items()):
                with col:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(st.session_state.X_test)
                        fig, auc_score = plot_roc_curve(st.session_state.y_test, proba, name)
                        if fig:
                            st.pyplot(fig, use_container_width=True)
                            st.metric("AUC", f"{auc_score:.4f}")

# ============================================================================
# PAGE 4: PREDICT
# ============================================================================
elif page == "🔮 Predict":
    st.header("🔮 Predictions")
    st.markdown("---")
    
    if not st.session_state.models:
        st.warning("⚠️ Train models first")
    else:
        tab1, tab2 = st.tabs(["Manual", "Batch"])
        
        with tab1:
            st.subheader("Single Prediction")
            input_dict = {}
            cols = st.columns(2)
            for idx, feat in enumerate(st.session_state.features):
                with cols[idx % 2]:
                    min_v = st.session_state.df[feat].min()
                    max_v = st.session_state.df[feat].max()
                    input_dict[feat] = st.number_input(feat, float(min_v), float(max_v), 
                                                       float((min_v + max_v) / 2), key=f"pred_{feat}")
            
            if st.button("Predict", use_container_width=True, type="primary", key="predict_btn"):
                inp_df = pd.DataFrame([input_dict])
                inp_scaled = st.session_state.scaler.transform(inp_df) if st.session_state.scaler else inp_df.values
                
                st.markdown("---")
                st.subheader("Results")
                
                cols = st.columns(len(st.session_state.models))
                for col, (name, model) in zip(cols, st.session_state.models.items()):
                    with col:
                        pred = model.predict(inp_scaled)[0]
                        st.metric(name, pred)
        
        with tab2:
            st.subheader("Batch Prediction")
            file = st.file_uploader("CSV:", type=["csv"])
            
            if file:
                try:
                    df = pd.read_csv(file)
                    missing = [f for f in st.session_state.features if f not in df.columns]
                    if missing:
                        st.error(f"Missing: {missing}")
                    else:
                        st.dataframe(df.head(), use_container_width=True)
                        
                        if st.button("Predict All", use_container_width=True, type="primary", key="batch_predict_btn"):
                            X_batch = df[st.session_state.features]
                            X_scaled = st.session_state.scaler.transform(X_batch) if st.session_state.scaler else X_batch.values
                            res_df = df.copy()
                            
                            for name, model in st.session_state.models.items():
                                res_df[f'{name}'] = model.predict(X_scaled)
                            
                            st.success("✅ Done!")
                            st.dataframe(res_df, use_container_width=True)
                            
                            csv = res_df.to_csv(index=False)
                            st.download_button("Download", csv, "predictions.csv", "text/csv", use_container_width=True)
                except Exception as e:
                    st.error(f"❌ {e}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #6b7280; padding: 1rem;'><p>ML Dashboard v3.0 | Streamlit • Scikit-learn • Pandas</p></div>", 
            unsafe_allow_html=True)
