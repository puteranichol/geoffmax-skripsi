import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
import os

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

st.set_page_config(
    page_title="Dashboard GEOFFMAX — Klasifikasi Kepuasan",
    page_icon="👟",
    layout="wide"
)

# ══════════════════════════════════════════════════════════
# PALET WARNA
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
PURPLE        = "#8B5CF6"
PURPLE_LIGHT  = "#F1EBFF"
TEXT_DARK     = "#241E42"
TEXT_MUTED    = "#8E8AA6"
BG_APP        = "#F6F5FB"
CARD_BORDER   = "#EFEDFB"

# ══════════════════════════════════════════════════════════
# SKEMA 5 KELAS (REVISI: mengikuti rating asli 1-5)
# ══════════════════════════════════════════════════════════
LABELS_TEXT  = ['Sangat Tidak Puas', 'Tidak Puas', 'Cukup Puas', 'Puas', 'Sangat Puas']
LABELS_ORDER = [0, 1, 2, 3, 4]
LABEL_MAP    = {0: 'Sangat Tidak Puas', 1: 'Tidak Puas', 2: 'Cukup Puas', 3: 'Puas', 4: 'Sangat Puas'}
WARNA        = {
    'Sangat Tidak Puas': "#C0392B",
    'Tidak Puas':        CORAL,
    'Cukup Puas':        ORANGE,
    'Puas':              "#82C91E",
    'Sangat Puas':       TEAL,
}
WARNA_KPI_BG = {
    'Sangat Tidak Puas': CORAL_LIGHT,
    'Tidak Puas':        CORAL_LIGHT,
    'Cukup Puas':        ORANGE_LIGHT,
    'Puas':              TEAL_LIGHT,
    'Sangat Puas':       TEAL_LIGHT,
}
EMOJI = {0: '😡', 1: '😞', 2: '😐', 3: '🙂', 4: '😍'}

# ══════════════════════════════════════════════════════════
# CSS GLOBAL (tidak diubah dari versi sebelumnya)
# ══════════════════════════════════════════════════════════
CUSTOM_CSS = f"""
<style>
    .stApp {{ background-color: {BG_APP}; }}
    section[data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid {CARD_BORDER};
    }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}
    section[data-testid="stSidebar"] h1 {{ color: {TEXT_DARK}; font-size: 1.3rem; font-weight: 800; }}
    section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] p {{ color: {TEXT_MUTED}; }}
    div[role="radiogroup"] {{ gap: 4px; }}
    div[role="radiogroup"] > label {{
        background-color: transparent; border-radius: 12px; padding: 10px 14px !important;
        margin-bottom: 2px; transition: all 0.15s ease-in-out; border: 1px solid transparent;
    }}
    div[role="radiogroup"] > label:hover {{ background-color: {BLUE_LIGHT}; }}
    div[role="radiogroup"] > label[data-checked="true"],
    div[role="radiogroup"] > label:has(input:checked) {{ background-color: transparent; border: 1px solid transparent; }}
    div[role="radiogroup"] > label:has(input:checked) p {{ color: {BLUE} !important; font-weight: 700 !important; }}
    div[role="radiogroup"] label {{ display: flex; align-items: center; gap: 0; }}
    div[role="radiogroup"] label div[data-baseweb="radio"] > div {{ border-color: {BLUE} !important; }}
    div[role="radiogroup"] label div[data-baseweb="radio"] input:checked ~ div {{
        border-color: {BLUE} !important; background-color: transparent !important;
    }}
    div[role="radiogroup"] label div[data-baseweb="radio"] input:checked ~ div > div,
    div[role="radiogroup"] label div[data-baseweb="radio"] input:checked ~ div::after {{ background-color: {BLUE} !important; }}
    div[role="radiogroup"] label div[data-baseweb="radio"] svg {{ fill: {BLUE} !important; }}
    .stButton > button {{
        background-color: {BLUE}; color: #FFFFFF !important; border-radius: 12px; border: none;
        padding: 0.6rem 1.2rem; font-weight: 600; transition: background-color 0.15s ease-in-out;
    }}
    .stButton > button * {{ color: #FFFFFF !important; }}
    .stButton > button:hover {{ background-color: {BLUE_DARK}; color: #FFFFFF !important; }}
    .stButton > button:hover * {{ color: #FFFFFF !important; }}
    .stButton > button:focus, .stButton > button:active {{ color: #FFFFFF !important; box-shadow: none; }}
    .stButton > button:focus *, .stButton > button:active * {{ color: #FFFFFF !important; }}
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] {{
        background-color: {BLUE_LIGHT}; border: 1px dashed {BLUE}; border-radius: 14px;
    }}
    h1 {{ color: {TEXT_DARK}; font-weight: 800; }}
    h2, h3 {{ color: {TEXT_DARK}; font-weight: 700; }}
    .stCaption, p {{ color: {TEXT_MUTED}; }}
    .kpi-card {{
        background: #FFFFFF; border: 1px solid {CARD_BORDER}; border-radius: 18px; padding: 18px 20px;
        display: flex; align-items: center; gap: 14px; box-shadow: 0 4px 14px rgba(36, 30, 66, 0.05); height: 92px;
    }}
    .kpi-icon {{
        min-width: 46px; height: 46px; border-radius: 14px; display: flex;
        align-items: center; justify-content: center; font-size: 22px;
    }}
    .kpi-value {{ font-size: 1.5rem; font-weight: 800; color: {TEXT_DARK}; line-height: 1.1; }}
    .kpi-label {{ font-size: 0.82rem; color: {TEXT_MUTED}; font-weight: 500; }}
    .section-card {{
        background: #FFFFFF; border: 1px solid {CARD_BORDER}; border-radius: 18px;
        padding: 18px 20px 6px 20px; box-shadow: 0 4px 14px rgba(36, 30, 66, 0.05); margin-bottom: 18px;
    }}
    .section-title {{ font-size: 1rem; font-weight: 700; color: {TEXT_DARK}; margin-bottom: 6px; }}
    div[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; border: 1px solid {CARD_BORDER}; }}
    div[data-testid="stAlert"] {{ border-radius: 14px; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

plt.rcParams.update({
    "axes.facecolor": "white", "figure.facecolor": "white", "axes.edgecolor": CARD_BORDER,
    "axes.labelcolor": TEXT_DARK, "text.color": TEXT_DARK, "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#F0EEF9", "font.family": "sans-serif",
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

# ══════════════════════════════════════════════════════════
# PREPROCESSING — HARUS PERSIS SAMA DENGAN NOTEBOOK REVISI9
# ══════════════════════════════════════════════════════════
@st.cache_resource
def load_stopwords():
    try:
        return joblib.load('stopwords_geoffmax.pkl')
    except Exception:
        pass
    # Fallback kalau file .pkl belum ada di repo — replikasi persis Sel 3 notebook REVISI9
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    sw = set(StopWordRemoverFactory().get_stop_words())
    slang = {'yg','dgn','utk','krn','sdh','blm','udah','udh','aja','sih','deh','nih','loh','dong',
             'emg','emang','sy','gw','lo','lu','wkwk','haha','lah','kah','nya','si'}
    # Kata negasi informal, kontras, intensitas, dan sentimen ringan TIDAK boleh dihapus
    kata_penting_sentimen = {
        'tidak','belum','tapi','tetapi','namun','tp',
        'ga','gak','nggak','enggak','kaga',
        'bgt','banget',
        'ok','oke','ya','yah'
    }
    return (sw | slang) - kata_penting_sentimen

@st.cache_resource
def load_stemmer():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    return StemmerFactory().create_stemmer()

@st.cache_resource
def load_leksikon():
    try:
        lek = joblib.load('leksikon_geoffmax.pkl')
        return lek['positif'], lek['negatif']
    except Exception:
        pass
    try:
        lex_pos = pd.read_csv('https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv', sep='\t')
        lex_neg = pd.read_csv('https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv', sep='\t')
        return set(lex_pos['word'].str.lower()), set(lex_neg['word'].str.lower())
    except Exception:
        return set(), set()

@st.cache_resource
def load_meta():
    try:
        return joblib.load('meta_geoffmax.pkl')
    except Exception:
        return None

def preprocess(teks, sw, stemmer):
    teks = str(teks).lower()
    teks = re.sub(r'[^a-z\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    kata = [w for w in teks.split() if w not in sw and len(w) > 1]
    return ' '.join([stemmer.stem(w) for w in kata])

def hitung_fitur_leksikon(teks_bersih, kamus_positif, kamus_negatif):
    """REVISI: proporsi (dibagi jumlah kata), BUKAN hitungan mentah —
    harus sama persis dengan yang dipakai saat training (notebook Sel 3c),
    karena MinMaxScaler di dalam pipeline di-fit dengan skala proporsi ini."""
    kata = str(teks_bersih).split()
    n_total = max(len(kata), 1)
    n_pos = sum(1 for k in kata if k in kamus_positif) / n_total
    n_neg = sum(1 for k in kata if k in kamus_negatif) / n_total
    campuran = 1 if (n_pos > 0 and n_neg > 0) else 0
    return n_pos, n_neg, campuran

def buat_label(r):
    """5 kelas mengikuti rating asli 1-5 (REVISI hasil sidang)."""
    return int(r) - 1

def bangun_fitur(df, tfidf, lex_scaler, selector):
    """Replika manual dari apa yang tadinya dilakukan ColumnTransformer + SelectKBest
    di dalam pipeline_final, tapi dipecah jadi komponen dasar supaya tidak rawan
    error kompatibilitas versi scikit-learn (mis. _RemainderColsList).
    Urutan kolom HARUS SAMA seperti saat training: [tfidf..., lex_scaled...]."""
    X_tfidf = tfidf.transform(df['ulasan_bersih'])
    X_lex_raw = df[['n_kata_positif', 'n_kata_negatif', 'fitur_campuran']].values
    X_lex_scaled = lex_scaler.transform(X_lex_raw)
    X_combined = hstack([X_tfidf, csr_matrix(X_lex_scaled)]).tocsr()
    X_selected = selector.transform(X_combined)
    return X_selected

def prediksi(df, tfidf, lex_scaler, selector, model):
    X_selected = bangun_fitur(df, tfidf, lex_scaler, selector)
    pred = model.predict(X_selected)
    proba = model.predict_proba(X_selected)
    return pred, proba

# ══════════════════════════════════════════════════════════
# LOAD KOMPONEN TERPISAH (hasil Sel Ekspor v2 di notebook REVISI9)
# ══════════════════════════════════════════════════════════
FILES_MODEL = ['tfidf_geoffmax.pkl', 'lex_scaler_geoffmax.pkl',
               'selector_geoffmax.pkl', 'model_geoffmax.pkl']

@st.cache_resource
def load_model():
    diag = {f: ('ada di folder' if os.path.exists(f) else 'TIDAK DITEMUKAN') for f in FILES_MODEL}
    try:
        tfidf      = joblib.load('tfidf_geoffmax.pkl')
        lex_scaler = joblib.load('lex_scaler_geoffmax.pkl')
        selector   = joblib.load('selector_geoffmax.pkl')
        model      = joblib.load('model_geoffmax.pkl')
        return (tfidf, lex_scaler, selector, model), True, diag, None
    except Exception as e:
        return None, False, diag, str(e)

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
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
    help="Kolom yang dibutuhkan: 'ulasan' dan 'rating' (1-5)"
)

stopwords = load_stopwords()
stemmer   = load_stemmer()
kamus_positif, kamus_negatif = load_leksikon()
meta      = load_meta()
komponen_model, pkl_ada, pkl_diag, pkl_err = load_model()
if pkl_ada:
    tfidf_pkl, lex_scaler_pkl, selector_pkl, model_pkl = komponen_model

with st.sidebar.expander("🔧 Diagnostik file .pkl", expanded=not pkl_ada):
    for fname, status in pkl_diag.items():
        st.write(f"**{fname}**: {status}")
    if pkl_err:
        st.error(f"Gagal load: {pkl_err}")
    elif pkl_ada:
        st.success("Semua file model berhasil dimuat ✅")
    st.caption(f"Working directory: `{os.getcwd()}`")

if uploaded:
    df = pd.read_csv(uploaded)
    if 'ulasan' not in df.columns or 'rating' not in df.columns:
        st.sidebar.error("❌ Kolom 'ulasan' dan 'rating' tidak ditemukan!")
        st.stop()

    df['Y_Kode']  = df['rating'].apply(buat_label)
    df['Y_Label'] = df['Y_Kode'].map(LABEL_MAP)
    df['ulasan_bersih'] = df['ulasan'].apply(lambda t: preprocess(t, stopwords, stemmer))
    df[['n_kata_positif', 'n_kata_negatif', 'fitur_campuran']] = df['ulasan_bersih'].apply(
        lambda t: pd.Series(hitung_fitur_leksikon(t, kamus_positif, kamus_negatif)))

    # Samakan perlakuan dengan notebook (Sel 3a): buang ulasan yang setelah preprocessing
    # tinggal <=3 kata, supaya data yang dievaluasi di dashboard konsisten dengan data
    # yang dipakai untuk melatih model_geoffmax.pkl.
    BATAS_MIN_KATA = 3
    n_sebelum_filter = len(df)
    df['_jml_kata'] = df['ulasan_bersih'].apply(lambda t: len(str(t).split()))
    df = df[df['_jml_kata'] > BATAS_MIN_KATA].drop(columns='_jml_kata').reset_index(drop=True)
    n_dibuang = n_sebelum_filter - len(df)

    sumber = f"Data Upload ({len(df)} ulasan, {n_dibuang} ulasan \u2264{BATAS_MIN_KATA} kata dibuang)"

    if pkl_ada:
        # Evaluasi model yang SUDAH dilatih terhadap data yang diupload
        y_eval = df['Y_Kode'].values
        y_pred, y_proba = prediksi(df, tfidf_pkl, lex_scaler_pkl, selector_pkl, model_pkl)
    else:
        st.sidebar.error("File model (.pkl) tidak lengkap di repo — tidak bisa evaluasi.")
        st.stop()

    st.sidebar.success(f"✅ {sumber}")
    data_ada = True
else:
    df = None
    data_ada = False
    if pkl_ada:
        st.sidebar.success("✅ Model dimuat. Upload CSV untuk analisis lengkap.")
    else:
        st.sidebar.info("Upload CSV untuk memulai analisis.")

# ══════════════════════════════════════════════════════════
# HALAMAN 1: BERANDA
# ══════════════════════════════════════════════════════════
if halaman == "📊 Beranda":
    st.title("Dashboard Klasifikasi Kepuasan Pelanggan GEOFFMAX")
    st.caption("Produk Sepatu di Shopee · Naïve Bayes 5 Kelas + Chi-Square · CRISP-DM")

    akurasi_cv_num = meta['akurasi'] if meta else 0.7966
    akurasi_statis = f"{akurasi_cv_num*100:.2f}%"

    if data_ada:
        # PENTING: KPI "Akurasi Model" TIDAK dihitung ulang dari CSV yang diupload.
        # Kalau CSV yang diupload sama/tumpang tindih dengan data training, accuracy_score
        # terhadap data itu akan selalu bias tinggi (model sudah "menghafal" data ini) —
        # bukan estimasi performa yang jujur. Angka resmi model tetap dari 10-Fold CV di notebook.
        acc_upload = accuracy_score(y_eval, y_pred)
        y_bin_home = label_binarize(y_eval, classes=LABELS_ORDER)
        auc = roc_auc_score(y_bin_home, y_proba, multi_class='ovr', average='macro')

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card(BLUE_LIGHT, "📝", f"{len(df)}", "Total Ulasan")
        with c2: kpi_card(TEAL_LIGHT, "🎯", akurasi_statis, "Akurasi Model (10-Fold CV)")
        with c3: kpi_card(ORANGE_LIGHT, "📈", f"{auc:.3f}", "AUC (thd data upload)")
        with c4: kpi_card(PURPLE_LIGHT, "🔎", "Chi-Square", "Seleksi Fitur")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            section_start("Distribusi Kelas Kepuasan")
            dist = [df[df['Y_Label'] == l].shape[0] for l in LABELS_TEXT]
            fig_kelas, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(LABELS_TEXT, dist, color=[WARNA[l] for l in LABELS_TEXT], edgecolor='white', width=0.5)
            for bar, v in zip(bars, dist):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, str(v), ha='center', fontweight='bold')
            ax.set_ylabel("Jumlah Ulasan")
            ax.tick_params(axis='x', rotation=20)
            ax.grid(axis='y', alpha=0.4)
            plt.tight_layout(); st.pyplot(fig_kelas)
            section_end()

        with col2:
            section_start("Distribusi Rating Bintang")
            rcnt = df['rating'].value_counts().sort_index()
            fig_rating, ax = plt.subplots(figsize=(6, 4))
            warna_bar = ["#C0392B", CORAL, ORANGE, "#82C91E", TEAL]
            ax.bar(rcnt.index, rcnt.values, color=warna_bar[:len(rcnt)], edgecolor='white', width=0.6)
            for x, v in zip(rcnt.index, rcnt.values):
                ax.text(x, v + 0.3, str(v), ha='center', fontweight='bold')
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xlabel("Rating"); ax.set_ylabel("Jumlah")
            ax.grid(axis='y', alpha=0.4)
            plt.tight_layout(); st.pyplot(fig_rating)
            section_end()

        section_start("Cuplikan Data")
        st.dataframe(df[['ulasan', 'rating', 'Y_Label']].head(10), use_container_width=True)
        section_end()
    else:
        st.info("Upload data CSV di sidebar kiri untuk melihat analisis lengkap.")
        section_start("Cara menggunakan dashboard ini")
        st.markdown(f"""
        1. Siapkan file CSV dengan kolom `ulasan` dan `rating` (1-5)
        2. Upload di sidebar kiri
        3. Dashboard otomatis memproses dan menampilkan hasil

        **Hasil model yang sudah dilatih (5 kelas):**
        - Akurasi: **{akurasi_statis}**
        - Algoritma: Naïve Bayes (Multinomial/Complement, dipilih via Grid Search) + fitur leksikon InSet + seleksi Chi-Square
        - Validasi: 10-Fold Stratified Cross Validation
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
    st.dataframe(df[['rating', 'panjang']].describe().round(2), use_container_width=True)
    section_end()

    section_start("Sebelum vs Sesudah Preprocessing")
    preview = df[['ulasan', 'ulasan_bersih', 'Y_Label']].head(10).copy()
    preview.columns = ['Ulasan Asli', 'Setelah Preprocessing', 'Label Y']
    st.dataframe(preview, use_container_width=True)
    section_end()

    section_start("Top 20 Kata — Bobot TF-IDF Tertinggi")
    fitur_names = tfidf_pkl.get_feature_names_out()
    X_all = tfidf_pkl.transform(df['ulasan_bersih'])
    mean_tfidf = np.asarray(X_all.mean(axis=0)).flatten()
    top20_idx = mean_tfidf.argsort()[-20:][::-1]
    top20 = pd.DataFrame({'Kata': fitur_names[top20_idx], 'Bobot TF-IDF': mean_tfidf[top20_idx]})
    fig_top20, ax = plt.subplots(figsize=(8, 6))
    biru_grad = sns.light_palette(BLUE, n_colors=20, reverse=True)
    sns.barplot(data=top20, x='Bobot TF-IDF', y='Kata', palette=biru_grad, ax=ax)
    ax.set_title('Top 20 Kata dengan Bobot TF-IDF Tertinggi')
    ax.grid(axis='x', alpha=0.4)
    plt.tight_layout(); st.pyplot(fig_top20)
    section_end()

    section_start("Distribusi Panjang Ulasan")
    fig_panjang, ax = plt.subplots(figsize=(8, 3))
    ax.hist(df['panjang'], bins=30, color=BLUE, edgecolor='white')
    ax.set_xlabel('Jumlah Kata per Ulasan')
    ax.set_ylabel('Frekuensi')
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout(); st.pyplot(fig_panjang)
    section_end()

# ══════════════════════════════════════════════════════════
# HALAMAN 3: EVALUASI MODEL
# ══════════════════════════════════════════════════════════
elif halaman == "🧠 Evaluasi Model":
    st.title("Evaluasi Model Naïve Bayes (5 Kelas)")
    st.caption("TF-IDF + fitur leksikon InSet + seleksi Chi-Square + tuning otomatis · 10-Fold CV")

    if not data_ada:
        akurasi_statis = f"{meta['akurasi']*100:.2f}%" if meta else "79.66%"
        c1, c2 = st.columns(2)
        with c1: kpi_card(TEAL_LIGHT, "🎯", akurasi_statis, "Akurasi (hasil training notebook)")
        with c2: kpi_card(PURPLE_LIGHT, "🏷️", "5 Kelas", "Skema Label")
        st.info("Upload data CSV di sidebar untuk melihat evaluasi interaktif terhadap data tersebut.")
        st.stop()

    acc = accuracy_score(y_eval, y_pred)
    y_bin = label_binarize(y_eval, classes=LABELS_ORDER)
    auc_per_class = roc_auc_score(y_bin, y_proba, multi_class='ovr', average=None)
    auc_macro = roc_auc_score(y_bin, y_proba, multi_class='ovr', average='macro')

    akurasi_cv_num = meta['akurasi'] if meta else 0.7966
    c1, c2 = st.columns(2)
    with c1: kpi_card(TEAL_LIGHT, "🎯", f"{acc*100:.2f}%", "Akurasi pada CSV yang Diupload")
    with c2: kpi_card(BLUE_LIGHT, "📈", f"{auc_macro:.3f}", "AUC (thd data upload)")
    st.caption(
        "Catatan: evaluasi ini memakai model yang SUDAH dilatih (pipeline_geoffmax.pkl), diuji "
        "terhadap data yang baru diupload — bukan melatih ulang. Angka ini BUKAN pengganti akurasi "
        f"10-Fold CV resmi ({akurasi_cv_num*100:.2f}%) yang dilaporkan di Bab IV skripsi — angka di sini "
        "hanya valid sebagai estimasi performa 'jujur' kalau CSV yang diupload benar-benar belum "
        "pernah dilihat model saat training."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        section_start("Confusion Matrix")
        cm = confusion_matrix(y_eval, y_pred, labels=LABELS_ORDER)
        fig_cm, ax = plt.subplots(figsize=(5.5, 4.5))
        biru_cmap = sns.light_palette(BLUE, as_cmap=True)
        sns.heatmap(cm, annot=True, fmt='d', cmap=biru_cmap,
                    xticklabels=LABELS_TEXT, yticklabels=LABELS_TEXT,
                    ax=ax, linewidths=0.5, annot_kws={'size': 11})
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
        ax.tick_params(axis='x', rotation=30)
        plt.tight_layout(); st.pyplot(fig_cm)
        section_end()

    with col2:
        section_start("Metrik per Kelas")
        report = classification_report(y_eval, y_pred, target_names=LABELS_TEXT, output_dict=True)
        ring = pd.DataFrame({
            'Kelas': LABELS_TEXT,
            'Precision': [report[l]['precision'] for l in LABELS_TEXT],
            'Recall': [report[l]['recall'] for l in LABELS_TEXT],
            'F1-Score': [report[l]['f1-score'] for l in LABELS_TEXT],
            'AUC (OvR)': auc_per_class,
            'Support': [int(report[l]['support']) for l in LABELS_TEXT],
        })
        st.dataframe(ring.round(4), use_container_width=True, hide_index=True)
        section_end()

    section_start("ROC Curve (One-vs-Rest, 5 Kelas)")
    fig_roc, ax = plt.subplots(figsize=(7, 5))
    warna_roc = [WARNA[l] for l in LABELS_TEXT]
    for i, (lbl, color) in enumerate(zip(LABELS_TEXT, warna_roc)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{lbl} (AUC = {auc_per_class[i]:.3f})')
    ax.plot([0, 1], [0, 1], '--', color='#C9C6DE', label='Random Guess')
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4)
    plt.tight_layout(); st.pyplot(fig_roc)
    section_end()

    section_start("Kata Paling Berpengaruh per Kelas")
    fitur_names_semua = np.concatenate([
        tfidf_pkl.get_feature_names_out(),
        ['n_kata_positif', 'n_kata_negatif', 'fitur_campuran']
    ])
    fitur_names_terpilih = fitur_names_semua[selector_pkl.get_support()]

    fig_kata, axes = plt.subplots(1, 5, figsize=(24, 5))
    for cls_idx, (cls_name, color) in enumerate(zip(LABELS_TEXT, warna_roc)):
        log_prob = model_pkl.feature_log_prob_[cls_idx]
        top10_idx = log_prob.argsort()[-10:][::-1]
        top10_kata = [fitur_names_terpilih[i] for i in top10_idx]
        top10_skor = [log_prob[i] for i in top10_idx]
        axes[cls_idx].barh(top10_kata[::-1], top10_skor[::-1], color=color)
        axes[cls_idx].set_title(f'{cls_name}', fontsize=11)
        axes[cls_idx].set_xlabel('Log Probability')
        axes[cls_idx].grid(axis='x', alpha=0.4)
    plt.tight_layout(); st.pyplot(fig_kata)
    section_end()

    st.info("💡 Screenshot halaman ini untuk Tabel & Grafik Bab IV skripsi.")

# ══════════════════════════════════════════════════════════
# HALAMAN 4: PREDIKSI ULASAN BARU
# ══════════════════════════════════════════════════════════
elif halaman == "🔮 Prediksi Ulasan Baru":
    st.title("Prediksi Kepuasan dari Ulasan Baru")
    st.caption("Masukkan teks ulasan produk sepatu GEOFFMAX dari Shopee")

    if not pkl_ada:
        st.error("Model belum tersedia. Pastikan semua file .pkl model ada di repository.")
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
            teks_bersih = preprocess(teks_input, stopwords, stemmer)
            if not teks_bersih.strip():
                st.warning("Teks terlalu pendek setelah preprocessing. Coba ulasan yang lebih panjang.")
            else:
                n_pos, n_neg, campuran = hitung_fitur_leksikon(teks_bersih, kamus_positif, kamus_negatif)
                df_input = pd.DataFrame([{
                    'ulasan_bersih': teks_bersih,
                    'n_kata_positif': n_pos,
                    'n_kata_negatif': n_neg,
                    'fitur_campuran': campuran
                }])
                pred_arr, proba_arr = prediksi(df_input, tfidf_pkl, lex_scaler_pkl, selector_pkl, model_pkl)
                pred_kelas = pred_arr[0]
                pred_proba = proba_arr[0]
                label_hasil = LABEL_MAP[pred_kelas]

                section_start("Hasil Klasifikasi")
                kpi_card(WARNA_KPI_BG[label_hasil], EMOJI[pred_kelas], label_hasil, "Prediksi Kepuasan")

                st.caption(f"Teks setelah preprocessing: *{teks_bersih}*")

                st.markdown("**Probabilitas Tiap Kelas**")
                fig_proba, ax = plt.subplots(figsize=(6.5, 3))
                bars = ax.barh(LABELS_TEXT, pred_proba, color=[WARNA[l] for l in LABELS_TEXT])
                ax.set_xlim(0, 1)
                for bar, v in zip(bars, pred_proba):
                    ax.text(v + 0.01, bar.get_y() + bar.get_height()/2, f"{v*100:.1f}%", va='center', fontweight='bold')
                ax.set_xlabel("Probabilitas")
                ax.grid(axis='x', alpha=0.4)
                plt.tight_layout(); st.pyplot(fig_proba)
                section_end()
        else:
            st.warning("Mohon isi teks ulasan terlebih dahulu.")

    st.info(
        "💡 Fitur ini adalah **prototipe sistem klasifikasi otomatis** — "
        "ulasan baru dari Shopee bisa langsung diklasifikasikan "
        "tingkat kepuasannya secara real-time."
    )
