# ==============================================================================
# SISTEM REKOMENDASI TEMPAT WISATA YOGYAKARTA - VERSI FINAL
#
# Deskripsi:
# Aplikasi web Streamlit yang memberikan rekomendasi tempat wisata menggunakan
# pendekatan hybrid (Content-Based & Collaborative Filtering) dengan panel admin
# untuk manajemen data (CRUD).
#
# File yang dibutuhkan:
# - raw_data (2).csv : Berisi data mentah tempat wisata (hanya untuk inisialisasi).
#
# Library yang dibutuhkan (isi untuk file requirements.txt):
# streamlit
# pandas
# scikit-learn
# surprise
# passlib[bcrypt]
# ==============================================================================

import streamlit as st
import pandas as pd
import sqlite3
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD
from passlib.context import CryptContext

# --- Konfigurasi Awal ---
st.set_page_config(layout="wide")
st.title("Sistem Rekomendasi Tempat Wisata Yogyakarta")

# --- Konfigurasi Keamanan & Database ---
DATABASE = 'wisata_secure.db'
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Fungsi-fungsi Database dan Model ---

def get_db_conn():
    """Membuat koneksi ke database SQLite."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Inisialisasi database dan tabel jika belum ada."""
    conn = get_db_conn()
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS wisata (
        no INTEGER PRIMARY KEY,
        nama TEXT NOT NULL, vote_average REAL, vote_count INTEGER, type TEXT,
        htm_weekday INTEGER, htm_weekend INTEGER, latitude TEXT, longitude TEXT, description TEXT
    );''')
    if conn.execute("SELECT COUNT(*) FROM wisata").fetchone()[0] == 0:
        if os.path.exists('raw_data (2).csv'):
            try:
                df_raw = pd.read_csv('raw_data (2).csv')
                if 'no' in df_raw.columns and df_raw['no'].is_unique:
                    df_raw.to_sql('wisata', conn, if_exists='append', index=False)
                    st.toast("Data wisata dari CSV berhasil dimasukkan.")
                else:
                    st.error("Kolom 'no' di CSV tidak valid atau tidak unik.")
            except Exception as e:
                st.error(f"Gagal membaca file CSV: {e}")
        else:
            st.error("File 'raw_data (2).csv' tidak ditemukan! Data wisata tidak dapat dimuat.")

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    );''')
    if conn.execute("SELECT COUNT(*) FROM users WHERE username='admin'").fetchone()[0] == 0:
        hashed_password = pwd_context.hash('admin123')
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", ('admin', hashed_password))

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ratings (
        user_id INTEGER NOT NULL, wisata_id INTEGER NOT NULL, rating REAL NOT NULL, comment TEXT,
        PRIMARY KEY (user_id, wisata_id), FOREIGN KEY (wisata_id) REFERENCES wisata(no) ON DELETE CASCADE
    );''')
    if conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0] == 0:
        dummy_ratings = [
            (1, 1, 9.0, "Sangat indah!"), (1, 60, 8.0, "Cukup bagus"), (1, 108, 9.5, "Fantastis!"),
            (2, 60, 9.0, None), (2, 119, 8.5, "Pemandangan bagus"), (2, 76, 9.0, "Sangat direkomendasikan"),
        ]
        cursor.executemany("INSERT INTO ratings (user_id, wisata_id, rating, comment) VALUES (?, ?, ?, ?)", dummy_ratings)
    conn.commit()
    conn.close()

@st.cache_resource
def load_and_train_models():
    """Memuat data dan melatih model. Dicache untuk efisiensi."""
    st.info("Mempersiapkan data dan model, mohon tunggu...")
    conn = get_db_conn()
    df_wisata_data = pd.read_sql_query("SELECT * FROM wisata", conn)
    df_ratings_data = pd.read_sql_query("SELECT * FROM ratings", conn)
    conn.close()

    if df_wisata_data.empty:
        st.error("Tabel 'wisata' kosong. Tidak bisa melatih model.")
        return None, None, None

    df_wisata_processed = df_wisata_data.copy()
    df_wisata_processed['type'] = df_wisata_processed['type'].fillna('Tidak ada informasi')
    df_wisata_processed['description'] = df_wisata_processed['description'].fillna('Tidak ada informasi')
    df_wisata_processed['content'] = df_wisata_processed['type'] + ' ' + df_wisata_processed['description']
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_wisata_processed['content'])
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    st.success("Model Content-Based Filtering siap.")

    algo_cf_model = None
    if not df_ratings_data.empty:
        reader = Reader(rating_scale=(1, 10))
        data_cf = Dataset.load_from_df(df_ratings_data[['user_id', 'wisata_id', 'rating']], reader)
        trainset_cf = data_cf.build_full_trainset()
        algo_cf_model = SVD(n_factors=50, n_epochs=20, random_state=42)
        algo_cf_model.fit(trainset_cf)
        st.success("Model Collaborative Filtering siap.")

    return df_wisata_processed, cosine_sim_matrix, algo_cf_model

def dapatkan_rekomendasi_content_based(nama_wisata, df, cosine_sim_matrix):
    try:
        idx = df[df['nama'].str.lower() == nama_wisata.lower()].index[0]
        sim_scores = sorted(list(enumerate(cosine_sim_matrix[idx])), key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        wisata_indices = [i[0] for i in sim_scores]
        return df.iloc[wisata_indices]
    except IndexError:
        return pd.DataFrame()

def dapatkan_rekomendasi_collaborative_filtering(user_id, df_wisata, model_cf):
    if model_cf is None:
        return pd.DataFrame()
    
    conn = get_db_conn()
    rated_wisata_ids = {row['wisata_id'] for row in conn.execute('SELECT wisata_id FROM ratings WHERE user_id = ?', (user_id,)).fetchall()}
    conn.close()
    
    unrated_wisata_ids = set(df_wisata['no']) - rated_wisata_ids
    
    predictions = [model_cf.predict(user_id, wisata_id) for wisata_id in unrated_wisata_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    top_n_ids = [pred.iid for pred in predictions[:5]]
    return df_wisata[df_wisata['no'].isin(top_n_ids)]

def dapatkan_rekomendasi_hybrid(nama_wisata, user_id, df_wisata, cosine_sim, model_cf):
    rekomendasi_cb = dapatkan_rekomendasi_content_based(nama_wisata, df_wisata, cosine_sim)
    rekomendasi_cf = dapatkan_rekomendasi_collaborative_filtering(user_id, df_wisata, model_cf)
    
    final_rekomendasi = pd.concat([rekomendasi_cf, rekomendasi_cb]).drop_duplicates(subset=['no']).head(5)
    return final_rekomendasi

def register_user(username, password):
    conn = get_db_conn()
    try:
        if conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone():
            st.error("Username sudah terdaftar.")
            return False
        hashed_password = pwd_context.hash(password)
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        st.success("Registrasi berhasil! Silakan login.")
        return True
    finally:
        conn.close()

def login_user(username, password):
    conn = get_db_conn()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if user and pwd_context.verify(password, user['password']):
        return user
    return None

def logout_user():
    # Hapus semua state session
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def submit_rating(user_id, wisata_id, rating, comment):
    conn = get_db_conn()
    try:
        conn.execute('INSERT OR REPLACE INTO ratings (user_id, wisata_id, rating, comment) VALUES (?, ?, ?, ?)',
                     (user_id, wisata_id, rating, comment))
        conn.commit()
        st.toast("Rating berhasil disimpan!")
        st.cache_resource.clear()
    finally:
        conn.close()

def tambah_wisata(data):
    """Menambahkan data wisata baru ke database."""
    conn = get_db_conn()
    try:
        max_no = conn.execute("SELECT MAX(no) FROM wisata").fetchone()[0] or 0
        new_no = max_no + 1
        
        conn.execute("""
            INSERT INTO wisata (no, nama, vote_average, vote_count, type, htm_weekday, htm_weekend, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (new_no, data['nama'], data['vote_average'], 0, data['type'], data['htm_weekday'], data['htm_weekend'], data['description']))
        conn.commit()
        st.success(f"Data '{data['nama']}' berhasil ditambahkan!")
        return True
    except Exception as e:
        st.error(f"Gagal menambahkan data: {e}")
        return False
    finally:
        conn.close()

def edit_wisata(no, data):
    """Mengedit data wisata yang ada di database."""
    conn = get_db_conn()
    try:
        conn.execute("""
            UPDATE wisata SET
            nama = ?, vote_average = ?, type = ?, htm_weekday = ?, htm_weekend = ?, description = ?
            WHERE no = ?
        """, (data['nama'], data['vote_average'], data['type'], data['htm_weekday'], data['htm_weekend'], data['description'], no))
        conn.commit()
        st.success(f"Data wisata dengan NO {no} berhasil diperbarui!")
        return True
    except Exception as e:
        st.error(f"Gagal mengedit data: {e}")
        return False
    finally:
        conn.close()

def hapus_wisata(no):
    """Menghapus data wisata dari database."""
    conn = get_db_conn()
    try:
        conn.execute("DELETE FROM ratings WHERE wisata_id = ?", (no,))
        conn.execute("DELETE FROM wisata WHERE no = ?", (no,))
        conn.commit()
        st.warning(f"Data wisata dengan NO {no} berhasil dihapus permanen.")
        return True
    except Exception as e:
        st.error(f"Gagal menghapus data: {e}")
        return False
    finally:
        conn.close()


# --- Inisialisasi Aplikasi ---
init_db()
df_wisata_app, cosine_sim_app, algo_cf_app = load_and_train_models()

# --- Inisialisasi Session State ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_id = None
if "selected_wisata_id" not in st.session_state:
    st.session_state.selected_wisata_id = None
if "admin_choice" not in st.session_state:
    st.session_state.admin_choice = "Lihat Semua Data"


# --- Layout Sidebar ---
with st.sidebar:
    st.header("👤 Status Pengguna")
    if st.session_state.get("logged_in"):
        st.success(f"Selamat datang, **{st.session_state.username}**!")
        st.button("Logout", on_click=logout_user, use_container_width=True)
    else:
        with st.form("login_form"):
            st.write("**Login**")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login", use_container_width=True):
                user_data = login_user(username, password)
                if user_data:
                    st.session_state.logged_in = True
                    st.session_state.username = user_data["username"]
                    st.session_state.user_id = user_data["id"]
                    st.rerun()
                else:
                    st.error("Username atau password salah.")
        
        st.markdown("---")
        with st.form("register_form"):
            st.write("**Belum punya akun? Daftar di sini**")
            username_reg = st.text_input("Username Baru")
            password_reg = st.text_input("Password Baru", type="password")
            if st.form_submit_button("Daftar", use_container_width=True):
                if username_reg and password_reg:
                    register_user(username_reg, password_reg)
                else:
                    st.warning("Username dan password tidak boleh kosong.")
    
    # --- Menu Navigasi Admin ---
    if st.session_state.get("username") == "admin":
        st.markdown("---")
        st.header("⚙️ Panel Admin")
        admin_menu = ["Lihat Semua Data", "Tambah Data Wisata", "Edit/Hapus Data Wisata"]
        st.session_state.admin_choice = st.radio(
            "Pilih Aksi:",
            options=admin_menu,
            key="admin_nav"
        )

# --- Layout Halaman Utama ---
if df_wisata_app is not None:
    
    # TAMPILAN UNTUK ADMIN
    if st.session_state.get("username") == "admin":
        st.title(f"Panel Admin: {st.session_state.get('admin_choice', 'Lihat Semua Data')}")

        # --- HALAMAN 1: LIHAT SEMUA DATA ---
        if st.session_state.admin_choice == "Lihat Semua Data":
            st.write("Berikut adalah seluruh data tempat wisata yang ada di database.")
            st.dataframe(df_wisata_app, use_container_width=True)
            if st.button("Refresh Data"):
                st.cache_resource.clear()
                st.rerun()

        # --- HALAMAN 2: TAMBAH DATA WISATA ---
        elif st.session_state.admin_choice == "Tambah Data Wisata":
            st.write("Silakan isi form di bawah ini untuk menambahkan tempat wisata baru.")
            with st.form("tambah_form", clear_on_submit=True):
                st.write("**Data Wisata Baru**")
                data = {}
                data['nama'] = st.text_input("Nama Tempat Wisata")
                data['type'] = st.text_input("Kategori (Contoh: Alam, Budaya, Kuliner)")
                data['description'] = st.text_area("Deskripsi")
                col1, col2, col3 = st.columns(3)
                with col1:
                    data['htm_weekday'] = st.number_input("HTM Weekday", min_value=0, step=5000)
                with col2:
                    data['htm_weekend'] = st.number_input("HTM Weekend", min_value=0, step=5000)
                with col3:
                    data['vote_average'] = st.number_input("Rating Awal (1-10)", min_value=0.0, max_value=10.0, step=0.1)
                
                submitted = st.form_submit_button("Tambah Data", use_container_width=True)
                if submitted:
                    if data.get('nama') and data.get('type') and data.get('description'): # Cek field penting
                        if tambah_wisata(data):
                            st.cache_resource.clear()
                            st.rerun()
                    else:
                        st.warning("Nama, Kategori, dan Deskripsi tidak boleh kosong.")

        # --- HALAMAN 3: EDIT/HAPUS DATA WISATA ---
        elif st.session_state.admin_choice == "Edit/Hapus Data Wisata":
            st.write("Pilih tempat wisata untuk diedit atau dihapus.")
            wisata_list = df_wisata_app['nama'].tolist()
            if wisata_list:
                selected_nama = st.selectbox("Pilih Wisata:", options=wisata_list)

                if selected_nama:
                    wisata_data = df_wisata_app[df_wisata_app['nama'] == selected_nama].iloc[0].to_dict()
                    wisata_no = wisata_data['no']

                    st.markdown("---")
                    st.subheader(f"Edit Data: {wisata_data['nama']}")
                    
                    with st.form("edit_form"):
                        edit_data = {}
                        edit_data['nama'] = st.text_input("Nama Tempat Wisata", value=wisata_data['nama'])
                        edit_data['type'] = st.text_input("Kategori", value=wisata_data.get('type', ''))
                        edit_data['description'] = st.text_area("Deskripsi", value=wisata_data.get('description', ''), height=200)
                        col1_edit, col2_edit, col3_edit = st.columns(3)
                        with col1_edit:
                            edit_data['htm_weekday'] = st.number_input("HTM Weekday", value=int(wisata_data.get('htm_weekday', 0)))
                        with col2_edit:
                            edit_data['htm_weekend'] = st.number_input("HTM Weekend", value=int(wisata_data.get('htm_weekend', 0)))
                        with col3_edit:
                            edit_data['vote_average'] = st.number_input("Rating", value=float(wisata_data.get('vote_average', 0.0)))

                        if st.form_submit_button("Simpan Perubahan", use_container_width=True):
                            if edit_wisata(wisata_no, edit_data):
                                st.cache_resource.clear()
                                st.rerun()
                    
                    st.markdown("---")
                    st.subheader(f"Hapus Data: {wisata_data['nama']}")
                    st.warning("PERINGATAN: Aksi ini tidak dapat dibatalkan dan akan menghapus semua data rating terkait.")
                    if st.button("Hapus Data Wisata Ini Secara Permanen", type="primary", use_container_width=True, key=f"delete_{wisata_no}"):
                        if hapus_wisata(wisata_no):
                            st.cache_resource.clear()
                            st.rerun()
            else:
                st.warning("Tidak ada data wisata untuk dikelola.")

    # TAMPILAN UNTUK PENGGUNA BIASA
    else:
        if not st.session_state.selected_wisata_id:
            st.header("🔍 Cari Rekomendasi & Filter Wisata")
            
            wisata_options = df_wisata_app['nama'].tolist()
            nama_wisata_cari = st.selectbox(
                "Pilih tempat wisata yang Anda sukai untuk mendapatkan rekomendasi:",
                options=wisata_options,
                index=wisata_options.index("Candi Prambanan") if "Candi Prambanan" in wisata_options else 0
            )

            st.markdown("---")
            st.write("**Filter Hasil Rekomendasi (Opsional)**")
            
            all_types = sorted(df_wisata_app['type'].dropna().unique().tolist())
            selected_types = st.multiselect("Pilih Kategori Wisata:", options=all_types)
            
            if st.button("Cari Rekomendasi", use_container_width=True, type="primary"):
                with st.spinner("Mencari rekomendasi terbaik..."):
                    user_id_for_rec = st.session_state.user_id if st.session_state.logged_in else 1
                    rekomendasi = dapatkan_rekomendasi_hybrid(
                        nama_wisata_cari, user_id_for_rec, df_wisata_app, cosine_sim_app, algo_cf_app
                    )
                    
                    if selected_types:
                        rekomendasi = rekomendasi[rekomendasi['type'].isin(selected_types)]

                    st.session_state.rekomendasi_terakhir = rekomendasi
                    st.session_state.last_search_input = {"nama": nama_wisata_cari, "kategori": selected_types}

            if 'rekomendasi_terakhir' in st.session_state and st.session_state.rekomendasi_terakhir is not None:
                last_input = st.session_state.get('last_search_input', {"nama": "", "kategori": []})
                st.subheader(f"Rekomendasi Berdasarkan '{last_input['nama']}'")
                if last_input['kategori']:
                    st.write(f"*Difilter berdasarkan kategori: {', '.join(last_input['kategori'])}*")

                if not st.session_state.rekomendasi_terakhir.empty:
                    for i, row in st.session_state.rekomendasi_terakhir.iterrows():
                        with st.container(border=True):
                            st.write(f"**{row['nama']}**")
                            st.write(f"*{row['type']}* | ⭐ **{row['vote_average']:.1f}**")
                            if st.button("Lihat Detail & Beri Rating", key=f"btn_detail_{row['no']}"):
                                st.session_state.selected_wisata_id = row['no']
                                st.rerun()
                else:
                    st.warning("Tidak ada rekomendasi yang cocok dengan kriteria Anda.")
        
        else:
            selected_wisata = df_wisata_app[df_wisata_app['no'] == st.session_state.selected_wisata_id].iloc[0]
            
            if st.button("⬅️ Kembali ke Hasil Pencarian"):
                st.session_state.selected_wisata_id = None
                if 'rekomendasi_terakhir' in st.session_state:
                    del st.session_state['rekomendasi_terakhir']
                st.rerun()

            with st.container(border=True):
                st.header(f"📍 Detail: {selected_wisata['nama']}")
                st.write(f"**Tipe:** {selected_wisata['type']}")
                st.write(f"**Rating:** {selected_wisata['vote_average']:.1f} (dari {selected_wisata['vote_count']} suara)")
                st.write(f"**Harga Tiket:** Weekday: Rp{int(selected_wisata.get('htm_weekday', 0)):,}, Weekend: Rp{int(selected_wisata.get('htm_weekend', 0)):,}")
                st.write(f"**Deskripsi:**")
                st.info(f"{selected_wisata['description']}")
                
                st.subheader("Beri Rating dan Komentar Anda")
                if st.session_state.logged_in:
                    with st.form("rating_form"):
                        rating = st.slider("Rating (1-10)", min_value=1.0, max_value=10.0, value=7.5, step=0.5)
                        comment = st.text_area("Komentar (Opsional)")
                        if st.form_submit_button("Kirim Rating", use_container_width=True):
                            submit_rating(st.session_state.user_id, selected_wisata['no'], rating, comment)
                            st.rerun()
                else:
                    st.warning("Anda harus login untuk memberikan rating.")

else:
    st.error("Aplikasi tidak dapat dimuat karena data wisata tidak tersedia. Periksa file 'raw_data (2).csv' atau hubungi admin.")