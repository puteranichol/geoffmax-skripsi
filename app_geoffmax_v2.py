import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, roc_auc_score, roc_curve)

st.set_page_config(
    page_title="Dashboard GEOFFMAX — Klasifikasi Kepuasan",
    page_icon="👟",
    layout="wide"
)

# ══════════════════════════════════════════════════════════
# PALET WARNA (mengikuti tema referensi: biru + putih + aksen pastel)
# ══════════════════════════════════════════════════════════
BLUE          = "#3B82F6"
BLUE_DARK     = "#1D4ED8"
BLUE_LIGHT    = "#EAF2FF"
TEAL          = "#2ED8B6"
TEAL_LIGHT    = "#E3FBF6"
CORAL         = "#FF6B81"
CORAL_LIGHT   = "#FFE9EC"
ORANGE        = "#FFA26B"
ORANGE_LIGHT  = "#FFF1E6"
TEXT_DARK     = "#241E42"
TEXT_MUTED    = "#8E8AA6"
BG_APP        = "#F6F5FB"
CARD_BORDER   = "#EFEDFB"

LABELS_TEXT  = ['Tidak Puas', 'Puas']
LABELS_ORDER = [0, 1]
LABEL_MAP    = {0: 'Tidak Puas', 1: 'Puas'}
WARNA        = {'Tidak Puas': CORAL, 'Puas': TEAL}

# ══════════════════════════════════════════════════════════
# CSS GLOBAL — meniru gaya kartu + sidebar pada referensi
# ══════════════════════════════════════════════════════════
CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {BG_APP};
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid {CARD_BORDER};
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.5rem;
    }}

    /* Judul sidebar */
    section[data-testid="stSidebar"] h1 {{
        color: {TEXT_DARK};
        font-size: 1.3rem;
        font-weight: 800;
    }}
    section[data-testid="stSidebar"] .stCaption, 
    section[data-testid="stSidebar"] p {{
        color: {TEXT_MUTED};
    }}

    /* Navigasi radio -> jadi list menu ala sidebar referensi */
    div[role="radiogroup"] {{
        gap: 4px;
    }}
    div[role="radiogroup"] > label {{
        background-color: transparent;
        border-radius: 12px;
        padding: 10px 14px !important;
        margin-bottom: 2px;
        transition: all 0.15s ease-in-out;
        border: 1px solid transparent;
    }}
    div[role="radiogroup"] > label:hover {{
        background-color: {BLUE_LIGHT};
    }}
    div[role="radiogroup"] > label[data-checked="true"],
    div[role="radiogroup"] > label:has(input:checked) {{
        background-color: {BLUE};
        border: 1px solid {BLUE};
    }}
    div[role="radiogroup"] > label:has(input:checked) p {{
        color: white !important;
        font-weight: 700 !important;
    }}
    div[role="radiogroup"] input {{
        display: none;
    }}

    /* Tombol utama (upload/proses) - pill biru ala "Register patient" */
    .stButton > button, .stDownloadButton > button {{
        background-color: {BLUE};
        color: #FFFFFF !important;
        border-radius: 12px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: background-color 0.15s ease-in-out;
    }}
    .stButton > button *, .stDownloadButton > button * {{
        color: #FFFFFF !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: {BLUE_DARK};
        color: #FFFFFF !important;
    }}
    .stButton > button:hover *, .stDownloadButton > button:hover * {{
        color: #FFFFFF !important;
    }}
    .stButton > button:focus, .stButton > button:active,
    .stDownloadButton > button:focus, .stDownloadButton > button:active {{
        color: #FFFFFF !important;
        box-shadow: none;
    }}
    .stButton > button:focus *, .stButton > button:active * {{
        color: #FFFFFF !important;
    }}

    /* File uploader dibuat lebih rapi/rounded */
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] {{
        background-color: {BLUE_LIGHT};
        border: 1px dashed {BLUE};
        border-radius: 14px;
    }}

    /* ---------- Judul halaman utama ---------- */
    h1 {{
        color: {TEXT_DARK};
        font-weight: 800;
    }}
    h2, h3 {{
        color: {TEXT_DARK};
        font-weight: 700;
    }}
    .stCaption, p {{
        color: {TEXT_MUTED};
    }}

    /* ---------- Kartu KPI custom ---------- */
    .kpi-card {{
        background: #FFFFFF;
        border: 1px solid {CARD_BORDER};
        border-radius: 18px;
        padding: 18px 20px;
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0 4px 14px rgba(36, 30, 66, 0.05);
        height: 92px;
    }}
    .kpi-icon {{
        min-width: 46px;
        height: 46px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
    }}
    .kpi-value {{
        font-size: 1.5rem;
        font-weight: 800;
        color: {TEXT_DARK};
        line-height: 1.1;
    }}
    .kpi-label {{
        font-size: 0.82rem;
        color: {TEXT_MUTED};
        font-weight: 500;
    }}

    /* ---------- Kartu pembungkus section (chart, tabel) ---------- */
    .section-card {{
        background: #FFFFFF;
        border: 1px solid {CARD_BORDER};
        border-radius: 18px;
        padding: 18px 20px 6px 20px;
        box-shadow: 0 4px 14px rgba(36, 30, 66, 0.05);
        margin-bottom: 18px;
    }}
    .section-title {{
        font-size: 1rem;
        font-weight: 700;
        color: {TEXT_DARK};
        margin-bottom: 6px;
    }}

    /* Dataframe */
    div[data-testid="stDataFrame"] {{
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid {CARD_BORDER};
    }}

    /* Info/warning/success boxes lebih rounded */
    div[data-testid="stAlert"] {{
        border-radius: 14px;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Tema matplotlib/seaborn senada dengan dashboard
plt.rcParams.update({
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": CARD_BORDER,
    "axes.labelcolor": TEXT_DARK,
    "text.color": TEXT_DARK,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#F0EEF9",
    "font.family": "sans-serif",
})

def kpi_card(icon_bg, icon, value, label):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon" style="background:{icon_bg};">{icon}</div>
        <div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def section_start(title):
    st.markdown(f"""<div class="section-card"><div class="section-title">{title}</div>""",
                unsafe_allow_html=True)

def section_end():
    st.markdown("</div>", unsafe_allow_html=True)

# ── STOPWORDS ─────────────────────────────────────────────
@st.cache_resource
def load_stopwords():
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        sw = set(StopWordRemoverFactory().get_stop_words())
    except:
        sw = set()
    slang = {'yg','dgn','utk','krn','sdh','blm','ga','gak','nggak',
             'udah','udh','bgt','aja','sih','deh','nih','loh','dong',
             'emg','emang','tp','sy','gw','lo','lu','ok','oke','ya',
             'yah','lah','kah','nya','si','wkwk','haha'}
    return sw | slang

def preprocess(teks, sw):
    teks = str(teks).lower()
    teks = re.sub(r'[^a-z\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return ' '.join([w for w in teks.split() if w not in sw and len(w) > 1])

def buat_label(r):
    return 0 if r <= 2 else 1

# ── LOAD MODEL PKL ────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model_geoffmax.pkl')
        tfidf = joblib.load('tfidf_geoffmax.pkl')
        return model, tfidf, True
    except:
        return None, None, False

# ── LATIH MODEL DARI DATA ─────────────────────────────────
@st.cache_resource
def latih(_df):
    from imblearn.over_sampling import SMOTE
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,1),
                             min_df=1, sublinear_tf=True)
    X = tfidf.fit_transform(_df['ulasan_bersih'])
    y = _df['Y_Kode'].values
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X.toarray(), y)
    skf   = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    model = MultinomialNB(alpha=0.01)
    y_pred  = cross_val_predict(model, X_sm, y_sm, cv=skf)
    y_proba = cross_val_predict(model, X_sm, y_sm, cv=skf, method='predict_proba')
    model.fit(X_sm, y_sm)
    return model, tfidf, X_sm, y_sm, y_pred, y_proba

# ── SIDEBAR ───────────────────────────────────────────────
st.sidebar.markdown(
    f"""<h1>👟 GEOFFMAX</h1>
    <p style="margin-top:-10px;">Klasifikasi Kepuasan Pelanggan</p>""",
    unsafe_allow_html=True
)
halaman = st.sidebar.radio("Navigasi", [
    "📊 Beranda",
    "📁 Data & Preprocessing",
    "🧠 Evaluasi Model",
    "🔮 Prediksi Ulasan Baru"
], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.subheader("Upload Data")
uploaded = st.sidebar.file_uploader(
    "Upload CSV ulasan GEOFFMAX",
    type=["csv"],
    help="Kolom yang dibutuhkan: 'ulasan' dan 'rating'"
)

stopwords = load_stopwords()
model_pkl, tfidf_pkl, pkl_ada = load_model()

if uploaded:
    df = pd.read_csv(uploaded)
    if 'ulasan' not in df.columns or 'rating' not in df.columns:
        st.sidebar.error("❌ Kolom 'ulasan' dan 'rating' tidak ditemukan!")
        st.stop()
    df['Y_Kode']       = df['rating'].apply(buat_label)
    df['Y_Label']      = df['Y_Kode'].map(LABEL_MAP)
    df['ulasan_bersih'] = df['ulasan'].apply(lambda t: preprocess(t, stopwords))
    sumber = f"Data Upload ({len(df)} ulasan)"

    if pkl_ada:
        model = model_pkl
        tfidf = tfidf_pkl
        from imblearn.over_sampling import SMOTE
        X_raw = tfidf.transform(df['ulasan_bersih'])
        y_raw = df['Y_Kode'].values
        sm    = SMOTE(random_state=42)
        X_sm, y_sm = sm.fit_resample(X_raw.toarray(), y_raw)
        skf   = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        y_pred  = cross_val_predict(model, X_sm, y_sm, cv=skf)
        y_proba = cross_val_predict(model, X_sm, y_sm, cv=skf, method='predict_proba')
    else:
        model, tfidf, X_sm, y_sm, y_pred, y_proba = latih(df)

    st.sidebar.success(f"✅ {sumber}")
    data_ada = True
else:
    df = None
    data_ada = False
    if pkl_ada:
        model = model_pkl
        tfidf = tfidf_pkl
        st.sidebar.success("✅ Model pkl dimuat. Upload CSV untuk analisis lengkap.")
    else:
        st.sidebar.info("Upload CSV untuk memulai analisis.")

# ══════════════════════════════════════════════════════════
# HALAMAN 1: BERANDA
# ══════════════════════════════════════════════════════════
if halaman == "📊 Beranda":
    st.title("Dashboard Klasifikasi Kepuasan Pelanggan GEOFFMAX")
    st.caption("Produk Sepatu di Shopee · Multinomial Naïve Bayes + SMOTE · CRISP-DM")

    if data_ada:
        acc = accuracy_score(y_sm, y_pred)
        auc = roc_auc_score(y_sm, y_proba[:,1])

        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi_card(BLUE_LIGHT, "📝", f"{len(df)}", "Total Ulasan")
        with c2: kpi_card(TEAL_LIGHT,   "🎯", f"{acc*100:.2f}%", "Akurasi Model")
        with c3: kpi_card(ORANGE_LIGHT, "📈", f"{auc:.3f}", "AUC")
        with c4: kpi_card(CORAL_LIGHT,  "⚖️", "SMOTE", "Metode Balancing")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            section_start("Distribusi Kelas Kepuasan")
            dist = [df[df['Y_Label']==l].shape[0] for l in LABELS_TEXT]
            fig, ax = plt.subplots(figsize=(5,4))
            bars = ax.bar(LABELS_TEXT, dist,
                          color=[WARNA[l] for l in LABELS_TEXT],
                          edgecolor='white', width=0.5)
            for bar, v in zip(bars, dist):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.5,
                        str(v), ha='center', fontweight='bold')
            ax.set_ylabel("Jumlah Ulasan")
            ax.grid(axis='y', alpha=0.4)
            plt.tight_layout(); st.pyplot(fig)
            section_end()

        with col2:
            section_start("Distribusi Rating Bintang")
            rcnt = df['rating'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(5,4))
            warna_bar = [CORAL, CORAL, TEAL, TEAL, TEAL]
            ax.bar(rcnt.index, rcnt.values,
                   color=warna_bar[:len(rcnt)],
                   edgecolor='white', width=0.6)
            for x, v in zip(rcnt.index, rcnt.values):
                ax.text(x, v+0.3, str(v), ha='center', fontweight='bold')
            ax.set_xticks([1,2,3,4,5])
            ax.set_xlabel("Rating"); ax.set_ylabel("Jumlah")
            ax.grid(axis='y', alpha=0.4)
            plt.tight_layout(); st.pyplot(fig)
            section_end()

        section_start("Cuplikan Data")
        st.dataframe(df[['ulasan','rating','Y_Label']].head(10),
                     use_container_width=True)
        section_end()
    else:
        st.info("Upload data CSV di sidebar kiri untuk melihat analisis lengkap.")
        section_start("Cara menggunakan dashboard ini")
        st.markdown("""
        1. Siapkan file CSV dengan kolom `ulasan` dan `rating`
        2. Upload di sidebar kiri
        3. Dashboard otomatis memproses dan menampilkan hasil

        **Hasil model yang sudah dilatih:**
        - Akurasi: **88.64%**
        - AUC: **0.949**
        - Algoritma: Multinomial Naïve Bayes + SMOTE
        - Validasi: 10-Fold Cross Validation
        """)
        section_end()

# ══════════════════════════════════════════════════════════
# HALAMAN 2: DATA & PREPROCESSING
# ══════════════════════════════════════════════════════════
elif halaman == "📁 Data & Preprocessing":
    st.title("Data & Preprocessing Teks")
    if not data_ada:
        st.warning("Upload data CSV terlebih dahulu melalui sidebar kiri.")
        st.stop()

    section_start("Statistik Deskriptif")
    df['panjang'] = df['ulasan'].str.split().str.len()
    st.dataframe(df[['rating','panjang']].describe().round(2),
                 use_container_width=True)
    section_end()

    section_start("Sebelum vs Sesudah Preprocessing")
    preview = df[['ulasan','ulasan_bersih','Y_Label']].head(10).copy()
    preview.columns = ['Ulasan Asli','Setelah Preprocessing','Label Y']
    st.dataframe(preview, use_container_width=True)
    section_end()

    section_start("Top 20 Kata — Bobot TF-IDF Tertinggi")
    fitur_names = tfidf.get_feature_names_out()
    X_all       = tfidf.transform(df['ulasan_bersih'])
    mean_tfidf  = np.asarray(X_all.mean(axis=0)).flatten()
    top20_idx   = mean_tfidf.argsort()[-20:][::-1]
    top20 = pd.DataFrame({
        'Kata': fitur_names[top20_idx],
        'Bobot TF-IDF': mean_tfidf[top20_idx]
    })
    fig, ax = plt.subplots(figsize=(8,6))
    biru_grad = sns.light_palette(BLUE, n_colors=20, reverse=True)
    sns.barplot(data=top20, x='Bobot TF-IDF', y='Kata',
                palette=biru_grad, ax=ax)
    ax.set_title('Top 20 Kata dengan Bobot TF-IDF Tertinggi')
    ax.grid(axis='x', alpha=0.4)
    plt.tight_layout(); st.pyplot(fig)
    section_end()

    section_start("Distribusi Panjang Ulasan")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.hist(df['panjang'], bins=30, color=BLUE, edgecolor='white')
    ax.set_xlabel('Jumlah Kata per Ulasan')
    ax.set_ylabel('Frekuensi')
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout(); st.pyplot(fig)
    section_end()

# ══════════════════════════════════════════════════════════
# HALAMAN 3: EVALUASI MODEL
# ══════════════════════════════════════════════════════════
elif halaman == "🧠 Evaluasi Model":
    st.title("Evaluasi Model Multinomial Naïve Bayes")
    st.caption("Parameter: alpha=0.01 · TF-IDF 1000 fitur unigram · SMOTE · 10-Fold CV")

    if not data_ada:
        c1,c2,c3 = st.columns(3)
        with c1: kpi_card(TEAL_LIGHT,   "🎯", "88.64%", "Akurasi")
        with c2: kpi_card(BLUE_LIGHT, "📈", "0.949",  "AUC")
        with c3: kpi_card(ORANGE_LIGHT, "🏷️", "0.89",   "F1 Macro")

        st.info("Upload data CSV di sidebar untuk melihat evaluasi interaktif.")
        section_start("Ringkasan metrik per kelas")
        st.markdown("""
        | Kelas | Precision | Recall | F1-Score | Support |
        |---|---|---|---|---|
        | Tidak Puas | 0.85 | 0.94 | 0.89 | 264 |
        | Puas | 0.93 | 0.83 | 0.88 | 264 |
        | **Macro avg** | **0.89** | **0.89** | **0.89** | **528** |
        """)
        section_end()
        st.stop()

    acc = accuracy_score(y_sm, y_pred)
    auc = roc_auc_score(y_sm, y_proba[:,1])

    c1, c2 = st.columns(2)
    with c1: kpi_card(TEAL_LIGHT, "🎯", f"{acc*100:.2f}%", "Akurasi Keseluruhan")
    with c2: kpi_card(BLUE_LIGHT, "📈", f"{auc:.3f}", "AUC")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        section_start("Confusion Matrix")
        cm = confusion_matrix(y_sm, y_pred, labels=LABELS_ORDER)
        fig, ax = plt.subplots(figsize=(5,4))
        biru_cmap = sns.light_palette(BLUE, as_cmap=True)
        sns.heatmap(cm, annot=True, fmt='d', cmap=biru_cmap,
                    xticklabels=LABELS_TEXT, yticklabels=LABELS_TEXT,
                    ax=ax, linewidths=0.5, annot_kws={'size':14})
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
        plt.tight_layout(); st.pyplot(fig)
        section_end()

    with col2:
        section_start("Metrik per Kelas")
        report = classification_report(y_sm, y_pred,
                     target_names=LABELS_TEXT, output_dict=True)
        ring = pd.DataFrame({
            'Kelas':     LABELS_TEXT,
            'Precision': [report[l]['precision'] for l in LABELS_TEXT],
            'Recall':    [report[l]['recall']    for l in LABELS_TEXT],
            'F1-Score':  [report[l]['f1-score']  for l in LABELS_TEXT],
            'Support':   [int(report[l]['support']) for l in LABELS_TEXT],
        })
        st.dataframe(ring.round(4), use_container_width=True, hide_index=True)
        section_end()

    section_start("ROC Curve")
    fpr, tpr, _ = roc_curve(y_sm, y_proba[:,1])
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(fpr, tpr, color=BLUE, lw=2,
            label=f'Naive Bayes (AUC = {auc:.3f})')
    ax.fill_between(fpr, tpr, alpha=0.12, color=BLUE)
    ax.plot([0,1],[0,1],'--', color='#C9C6DE', label='Random Guess')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.4)
    plt.tight_layout(); st.pyplot(fig)
    section_end()

    section_start("Kata Paling Berpengaruh per Kelas")
    fitur_names = tfidf.get_feature_names_out()
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    for cls_idx, (cls_name, color) in enumerate(zip(LABELS_TEXT,
                                               [CORAL, TEAL])):
        log_prob   = model.feature_log_prob_[cls_idx]
        top10_idx  = log_prob.argsort()[-10:][::-1]
        top10_kata = [fitur_names[i] for i in top10_idx]
        top10_skor = [log_prob[i] for i in top10_idx]
        axes[cls_idx].barh(top10_kata[::-1], top10_skor[::-1], color=color)
        axes[cls_idx].set_title(f'{cls_name}', fontsize=12)
        axes[cls_idx].set_xlabel('Log Probability')
        axes[cls_idx].grid(axis='x', alpha=0.4)
    plt.tight_layout(); st.pyplot(fig)
    section_end()

    st.info("💡 Screenshot halaman ini untuk Tabel & Grafik Bab IV skripsi.")

# ══════════════════════════════════════════════════════════
# HALAMAN 4: PREDIKSI ULASAN BARU
# ══════════════════════════════════════════════════════════
elif halaman == "🔮 Prediksi Ulasan Baru":
    st.title("Prediksi Kepuasan dari Ulasan Baru")
    st.caption("Masukkan teks ulasan produk sepatu GEOFFMAX dari Shopee")

    if not (pkl_ada or data_ada):
        st.error("Model belum tersedia. Upload CSV atau pastikan file model.pkl ada di repository.")
        st.stop()

    section_start("Input Ulasan")
    teks_input = st.text_area(
        "Ketik atau paste ulasan di sini:",
        height=120,
        placeholder="Contoh: sepatunya bagus banget, nyaman dipakai seharian, ukuran pas!",
        label_visibility="collapsed"
    )
    klik = st.button("🔍 Klasifikasikan Ulasan", type="primary")
    section_end()

    if klik:
        if teks_input.strip():
            teks_bersih = preprocess(teks_input, stopwords)
            if not teks_bersih.strip():
                st.warning("Teks terlalu pendek setelah preprocessing. Coba ulasan yang lebih panjang.")
            else:
                inp         = tfidf.transform([teks_bersih])
                pred_kelas  = model.predict(inp)[0]
                pred_proba  = model.predict_proba(inp)[0]
                label_hasil = LABEL_MAP[pred_kelas]

                section_start("Hasil Klasifikasi")
                if label_hasil == "Puas":
                    kpi_card(TEAL_LIGHT, "😊", label_hasil, "Prediksi Kepuasan")
                else:
                    kpi_card(CORAL_LIGHT, "😞", label_hasil, "Prediksi Kepuasan")

                st.caption(f"Teks setelah preprocessing: *{teks_bersih}*")

                st.markdown("**Probabilitas Tiap Kelas**")
                fig, ax = plt.subplots(figsize=(6,2.5))
                bars = ax.barh(LABELS_TEXT, pred_proba,
                               color=[WARNA[l] for l in LABELS_TEXT])
                ax.set_xlim(0,1)
                for bar, v in zip(bars, pred_proba):
                    ax.text(v+0.01, bar.get_y()+bar.get_height()/2,
                            f"{v*100:.1f}%", va='center', fontweight='bold')
                ax.set_xlabel("Probabilitas")
                ax.grid(axis='x', alpha=0.4)
                plt.tight_layout(); st.pyplot(fig)
                section_end()
        else:
            st.warning("Mohon isi teks ulasan terlebih dahulu.")

    st.info(
        "💡 Fitur ini adalah **prototipe sistem klasifikasi otomatis** — "
        "ulasan baru dari Shopee bisa langsung diklasifikasikan "
        "tingkat kepuasannya secara real-time."
    )
