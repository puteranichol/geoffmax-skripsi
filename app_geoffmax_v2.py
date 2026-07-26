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

# ── KONSTANTA ─────────────────────────────────────────────
LABELS_TEXT  = ['Tidak Puas', 'Puas']
LABELS_ORDER = [0, 1]
LABEL_MAP    = {0: 'Tidak Puas', 1: 'Puas'}
WARNA        = {'Tidak Puas': '#E74C3C', 'Puas': '#27AE60'}

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
st.sidebar.title("👟 GEOFFMAX Dashboard")
st.sidebar.caption("Klasifikasi Kepuasan Pelanggan Shopee")
halaman = st.sidebar.radio("Navigasi", [
    "📊 Beranda",
    "📁 Data & Preprocessing",
    "🧠 Evaluasi Model",
    "🔮 Prediksi Ulasan Baru"
])
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
        c1.metric("Total Ulasan",    f"{len(df)}")
        c2.metric("Akurasi Model",   f"{acc*100:.2f}%")
        c3.metric("AUC",             f"{auc:.3f}")
        c4.metric("Metode Balancing","SMOTE")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribusi Kelas Kepuasan")
            dist = [df[df['Y_Label']==l].shape[0] for l in LABELS_TEXT]
            fig, ax = plt.subplots(figsize=(5,4))
            bars = ax.bar(LABELS_TEXT, dist,
                          color=[WARNA[l] for l in LABELS_TEXT],
                          edgecolor='white', width=0.5)
            for bar, v in zip(bars, dist):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.5,
                        str(v), ha='center', fontweight='bold')
            ax.set_ylabel("Jumlah Ulasan")
            plt.tight_layout(); st.pyplot(fig)

        with col2:
            st.subheader("Distribusi Rating Bintang")
            rcnt = df['rating'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(5,4))
            ax.bar(rcnt.index, rcnt.values,
                   color=['#E74C3C','#E74C3C','#27AE60','#27AE60','#27AE60'],
                   edgecolor='white', width=0.6)
            for x, v in zip(rcnt.index, rcnt.values):
                ax.text(x, v+0.3, str(v), ha='center', fontweight='bold')
            ax.set_xticks([1,2,3,4,5])
            ax.set_xlabel("Rating"); ax.set_ylabel("Jumlah")
            plt.tight_layout(); st.pyplot(fig)

        st.markdown("---")
        st.subheader("Cuplikan Data")
        st.dataframe(df[['ulasan','rating','Y_Label']].head(10),
                     use_container_width=True)
    else:
        st.info("Upload data CSV di sidebar kiri untuk melihat analisis lengkap.")
        st.markdown("""
        **Cara menggunakan dashboard ini:**
        1. Siapkan file CSV dengan kolom `ulasan` dan `rating`
        2. Upload di sidebar kiri
        3. Dashboard otomatis memproses dan menampilkan hasil

        **Hasil model yang sudah dilatih:**
        - Akurasi: **88.64%**
        - AUC: **0.949**
        - Algoritma: Multinomial Naïve Bayes + SMOTE
        - Validasi: 10-Fold Cross Validation
        """)

# ══════════════════════════════════════════════════════════
# HALAMAN 2: DATA & PREPROCESSING
# ══════════════════════════════════════════════════════════
elif halaman == "📁 Data & Preprocessing":
    st.title("Data & Preprocessing Teks")
    if not data_ada:
        st.warning("Upload data CSV terlebih dahulu melalui sidebar kiri.")
        st.stop()

    st.subheader("Statistik Deskriptif")
    df['panjang'] = df['ulasan'].str.split().str.len()
    st.dataframe(df[['rating','panjang']].describe().round(2),
                 use_container_width=True)

    st.subheader("Sebelum vs Sesudah Preprocessing")
    preview = df[['ulasan','ulasan_bersih','Y_Label']].head(10).copy()
    preview.columns = ['Ulasan Asli','Setelah Preprocessing','Label Y']
    st.dataframe(preview, use_container_width=True)

    st.subheader("Top 20 Kata — Bobot TF-IDF Tertinggi")
    fitur_names = tfidf.get_feature_names_out()
    X_all       = tfidf.transform(df['ulasan_bersih'])
    mean_tfidf  = np.asarray(X_all.mean(axis=0)).flatten()
    top20_idx   = mean_tfidf.argsort()[-20:][::-1]
    top20 = pd.DataFrame({
        'Kata': fitur_names[top20_idx],
        'Bobot TF-IDF': mean_tfidf[top20_idx]
    })
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(data=top20, x='Bobot TF-IDF', y='Kata',
                palette='Blues_r', ax=ax)
    ax.set_title('Top 20 Kata dengan Bobot TF-IDF Tertinggi')
    plt.tight_layout(); st.pyplot(fig)

    st.subheader("Distribusi Panjang Ulasan")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.hist(df['panjang'], bins=30, color='#2E75B6', edgecolor='white')
    ax.set_xlabel('Jumlah Kata per Ulasan')
    ax.set_ylabel('Frekuensi')
    plt.tight_layout(); st.pyplot(fig)

# ══════════════════════════════════════════════════════════
# HALAMAN 3: EVALUASI MODEL
# ══════════════════════════════════════════════════════════
elif halaman == "🧠 Evaluasi Model":
    st.title("Evaluasi Model Multinomial Naïve Bayes")
    st.caption("Parameter: alpha=0.01 · TF-IDF 1000 fitur unigram · SMOTE · 10-Fold CV")

    if not data_ada:
        # Tampilkan hasil statis dari model pkl
        st.subheader("Hasil Evaluasi Model (dari data pelatihan)")
        c1,c2,c3 = st.columns(3)
        c1.metric("Akurasi",  "88.64%")
        c2.metric("AUC",      "0.949")
        c3.metric("F1 Macro", "0.89")

        st.info("Upload data CSV di sidebar untuk melihat evaluasi interaktif.")
        st.markdown("""
        **Ringkasan metrik per kelas:**

        | Kelas | Precision | Recall | F1-Score | Support |
        |---|---|---|---|---|
        | Tidak Puas | 0.85 | 0.94 | 0.89 | 264 |
        | Puas | 0.93 | 0.83 | 0.88 | 264 |
        | **Macro avg** | **0.89** | **0.89** | **0.89** | **528** |
        """)
        st.stop()

    acc = accuracy_score(y_sm, y_pred)
    auc = roc_auc_score(y_sm, y_proba[:,1])
    st.metric("Akurasi Keseluruhan", f"{acc*100:.2f}%")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_sm, y_pred, labels=LABELS_ORDER)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=LABELS_TEXT, yticklabels=LABELS_TEXT,
                    ax=ax, linewidths=0.5, annot_kws={'size':14})
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
        plt.tight_layout(); st.pyplot(fig)

    with col2:
        st.subheader("Metrik per Kelas")
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
        st.metric("AUC", f"{auc:.3f}")

    st.markdown("---")
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_sm, y_proba[:,1])
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(fpr, tpr, color='#2E75B6', lw=2,
            label=f'Naive Bayes (AUC = {auc:.3f})')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#2E75B6')
    ax.plot([0,1],[0,1],'--', color='gray', label='Random Guess')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=11)
    ax.set_title("ROC Curve — Multinomial Naïve Bayes")
    plt.tight_layout(); st.pyplot(fig)

    st.markdown("---")
    st.subheader("Kata Paling Berpengaruh per Kelas")
    fitur_names = tfidf.get_feature_names_out()
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    for cls_idx, (cls_name, color) in enumerate(zip(LABELS_TEXT,
                                               ['#E74C3C','#27AE60'])):
        log_prob   = model.feature_log_prob_[cls_idx]
        top10_idx  = log_prob.argsort()[-10:][::-1]
        top10_kata = [fitur_names[i] for i in top10_idx]
        top10_skor = [log_prob[i] for i in top10_idx]
        axes[cls_idx].barh(top10_kata[::-1], top10_skor[::-1], color=color)
        axes[cls_idx].set_title(f'{cls_name}', fontsize=12)
        axes[cls_idx].set_xlabel('Log Probability')
    plt.suptitle('Top 10 Kata per Kelas Kepuasan', fontsize=13, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig)

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

    teks_input = st.text_area(
        "Ketik atau paste ulasan di sini:",
        height=120,
        placeholder="Contoh: sepatunya bagus banget, nyaman dipakai seharian, ukuran pas!"
    )

    if st.button("🔍 Klasifikasikan Ulasan", type="primary"):
        if teks_input.strip():
            teks_bersih = preprocess(teks_input, stopwords)
            if not teks_bersih.strip():
                st.warning("Teks terlalu pendek setelah preprocessing. Coba ulasan yang lebih panjang.")
            else:
                inp         = tfidf.transform([teks_bersih])
                pred_kelas  = model.predict(inp)[0]
                pred_proba  = model.predict_proba(inp)[0]
                label_hasil = LABEL_MAP[pred_kelas]

                if label_hasil == "Puas":
                    st.success(f"### Hasil Klasifikasi: {label_hasil} 😊")
                else:
                    st.error(f"### Hasil Klasifikasi: {label_hasil} 😞")

                st.caption(f"Teks setelah preprocessing: *{teks_bersih}*")

                st.subheader("Probabilitas Tiap Kelas")
                fig, ax = plt.subplots(figsize=(6,2.5))
                bars = ax.barh(LABELS_TEXT, pred_proba,
                               color=[WARNA[l] for l in LABELS_TEXT])
                ax.set_xlim(0,1)
                for bar, v in zip(bars, pred_proba):
                    ax.text(v+0.01, bar.get_y()+bar.get_height()/2,
                            f"{v*100:.1f}%", va='center', fontweight='bold')
                ax.set_xlabel("Probabilitas")
                plt.tight_layout(); st.pyplot(fig)
        else:
            st.warning("Mohon isi teks ulasan terlebih dahulu.")

    st.markdown("---")
    st.info(
        "💡 Fitur ini adalah **prototipe sistem klasifikasi otomatis** — "
        "ulasan baru dari Shopee bisa langsung diklasifikasikan "
        "tingkat kepuasannya secara real-time."
    )
