# ==============================================================================
# SISTEM REKOMENDASI TEMPAT WISATA YOGYAKARTA - VERSI DEPLOYMENT STABIL
# Model: Content-Based Filtering
# ==============================================================================

import streamlit as st
import pandas as pd
import sqlite3
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from passlib.context import CryptContext

# --- Konfigurasi Awal ---
st.set_page_config(layout="wide")
st.title("Sistem Rekomendasi Tempat Wisata Yogyakarta")

# --- Konfigurasi Keamanan & Database ---
DATABASE = 'wisata_secure.db'
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Fungsi-fungsi Database dan Model ---

def get_db_conn():
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS wisata (
        no INTEGER PRIMARY KEY, nama TEXT NOT NULL, vote_average REAL, vote_count INTEGER, type TEXT,
        htm_weekday INTEGER, htm_weekend INTEGER, latitude TEXT, longitude TEXT, description TEXT
    );''')
    if conn.execute("SELECT COUNT(*) FROM wisata").fetchone()[0] == 0:
        if os.path.exists('raw_data (2).csv'):
            try:
                df_raw = pd.read_csv('raw_data (2).csv')
                df_raw.to_sql('wisata', conn, if_exists='append', index=False)
            except Exception as e:
                st.error(f"Gagal membaca file CSV: {e}")
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL UNIQUE, password TEXT NOT NULL
    );''')
    if conn.execute("SELECT COUNT(*) FROM users WHERE username='admin'").fetchone()[0] == 0:
        hashed_password = pwd_context.hash('admin123')
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", ('admin', hashed_password))
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ratings (
        user_id INTEGER NOT NULL, wisata_id INTEGER NOT NULL, rating REAL NOT NULL, comment TEXT,
        PRIMARY KEY (user_id, wisata_id), FOREIGN KEY (wisata_id) REFERENCES wisata(no) ON DELETE CASCADE
    );''')
    conn.commit()
    conn.close()

@st.cache_resource
def load_and_train_models():
    st.info("Mempersiapkan data dan model, mohon tunggu...")
    conn = get_db_conn()
    df_wisata_data = pd.read_sql_query("SELECT * FROM wisata", conn)
    conn.close()

    if df_wisata_data.empty:
        return None, None

    df_wisata_processed = df_wisata_data.copy()
    df_wisata_processed['type'] = df_wisata_processed['type'].fillna('Tidak ada informasi')
    df_wisata_processed['description'] = df_wisata_processed['description'].fillna('Tidak ada informasi')
    df_wisata_processed['content'] = df_wisata_processed['type'] + ' ' + df_wisata_processed['description']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_wisata_processed['content'])
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    st.success("Model Rekomendasi siap.")

    return df_wisata_processed, cosine_sim_matrix

def dapatkan_rekomendasi(nama_wisata, df, cosine_sim_matrix):
    try:
        idx = df[df['nama'].str.lower() == nama_wisata.lower()].index[0]
        sim_scores = sorted(list(enumerate(cosine_sim_matrix[idx])), key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11] # Memberi 10 rekomendasi
        wisata_indices = [i[0] for i in sim_scores]
        return df.iloc[wisata_indices]
    except IndexError:
        return pd.DataFrame()

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
    finally:
        conn.close()

def tambah_wisata(data):
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
    finally:
        conn.close()

def edit_wisata(no, data):
    conn = get_db_conn()
    try:
        conn.execute("""
            UPDATE wisata SET
            nama = ?, vote_average = ?, type = ?, htm_weekday = ?, htm_weekend = ?, description = ?
            WHERE no = ?
        """, (data['nama'], data['vote_average'], data['type'], data['htm_weekday'], data['htm_weekend'], data['description'], no))
        conn.commit()
        st.success(f"Data wisata dengan NO {no} berhasil diperbarui!")
    finally:
        conn.close()

def hapus_wisata(no):
    conn = get_db_conn()
    try:
        conn.execute("DELETE FROM ratings WHERE wisata_id = ?", (no,))
        conn.execute("DELETE FROM wisata WHERE no = ?", (no,))
        conn.commit()
        st.warning(f"Data wisata dengan NO {no} berhasil dihapus permanen.")
    finally:
        conn.close()

# --- Inisialisasi Aplikasi ---
init_db()
df_wisata_app, cosine_sim_app = load_and_train_models()

# --- Inisialisasi Session State ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "admin_choice" not in st.session_state:
    st.session_state.admin_choice = "Lihat Semua Data"
if "selected_wisata_id" not in st.session_state:
    st.session_state.selected_wisata_id = None

# --- Layout Sidebar ---
with st.sidebar:
    st.header("👤 Status Pengguna")
    if st.session_state.get("logged_in"):
        st.success(f"Selamat datang, **{st.session_state.get('username')}**!")
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
        with st.form("register_form"):
            st.write("**Belum punya akun? Daftar**")
            username_reg = st.text_input("Username Baru", key="reg_user")
            password_reg = st.text_input("Password Baru", type="password", key="reg_pass")
            if st.form_submit_button("Daftar", use_container_width=True):
                if username_reg and password_reg:
                    register_user(username_reg, password_reg)

    if st.session_state.get("username") == "admin":
        st.markdown("---")
        st.header("⚙️ Panel Admin")
        st.radio("Pilih Aksi:", ["Lihat Semua Data", "Tambah Data Wisata", "Edit/Hapus Data Wisata"], key="admin_choice")

# --- Layout Halaman Utama ---
if df_wisata_app is not None:
    if st.session_state.get("username") == "admin":
        st.title(f"Panel Admin: {st.session_state.admin_choice}")
        if st.session_state.admin_choice == "Lihat Semua Data":
            st.dataframe(df_wisata_app, use_container_width=True)
            if st.button("Refresh Data"):
                st.cache_resource.clear()
                st.rerun()
        elif st.session_state.admin_choice == "Tambah Data Wisata":
            with st.form("tambah_form"):
                data = {
                    'nama': st.text_input("Nama"), 'type': st.text_input("Kategori"),
                    'description': st.text_area("Deskripsi"),
                    'htm_weekday': st.number_input("HTM Weekday", min_value=0),
                    'htm_weekend': st.number_input("HTM Weekend", min_value=0),
                    'vote_average': st.number_input("Rating Awal", min_value=0.0, max_value=10.0)
                }
                if st.form_submit_button("Tambah Data"):
                    tambah_wisata(data)
                    st.cache_resource.clear()
                    st.rerun()
        elif st.session_state.admin_choice == "Edit/Hapus Data Wisata":
            wisata_list = df_wisata_app['nama'].tolist()
            selected_nama = st.selectbox("Pilih Wisata untuk Diedit/Dihapus", wisata_list)
            if selected_nama:
                wisata_data = df_wisata_app[df_wisata_app['nama'] == selected_nama].iloc[0]
                with st.form("edit_form"):
                    st.write(f"**Edit Data: {wisata_data['nama']}**")
                    edit_data = {
                        'nama': st.text_input("Nama", value=wisata_data['nama']),
                        'type': st.text_input("Kategori", value=wisata_data.get('type', '')),
                        'description': st.text_area("Deskripsi", value=wisata_data.get('description', '')),
                        'htm_weekday': st.number_input("HTM Weekday", value=int(wisata_data.get('htm_weekday',0))),
                        'htm_weekend': st.number_input("HTM Weekend", value=int(wisata_data.get('htm_weekend',0))),
                        'vote_average': st.number_input("Rating", value=float(wisata_data.get('vote_average',0.0)))
                    }
                    col1, col2 = st.columns(2)
                    if col1.form_submit_button("Simpan Perubahan"):
                        edit_wisata(wisata_data['no'], edit_data)
                        st.cache_resource.clear()
                        st.rerun()
                    if col2.form_submit_button("HAPUS DATA", type="primary"):
                        hapus_wisata(wisata_data['no'])
                        st.cache_resource.clear()
                        st.rerun()
    else:
        # Tampilan Pengguna Biasa
        if not st.session_state.selected_wisata_id:
            st.header("🔍 Cari Rekomendasi Wisata")
            wisata_options = df_wisata_app['nama'].tolist()
            nama_wisata_cari = st.selectbox("Pilih tempat wisata yang Anda sukai:", options=wisata_options)
            if st.button("Cari Rekomendasi", use_container_width=True):
                st.session_state.rekomendasi_terakhir = dapatkan_rekomendasi(nama_wisata_cari, df_wisata_app, cosine_sim_app)
                st.session_state.last_search = nama_wisata_cari

            if 'rekomendasi_terakhir' in st.session_state:
                st.subheader(f"Rekomendasi Berdasarkan '{st.session_state.last_search}'")
                rekomendasi = st.session_state.rekomendasi_terakhir
                if not rekomendasi.empty:
                    for i, row in rekomendasi.iterrows():
                         with st.container(border=True):
                            st.write(f"**{row['nama']}**")
                            st.write(f"*{row['type']}* | ⭐ **{row['vote_average']:.1f}**")
                            if st.button("Lihat Detail & Beri Rating", key=f"btn_detail_{row['no']}"):
                                st.session_state.selected_wisata_id = row['no']
                                st.rerun()
                else:
                    st.write("Tidak ada rekomendasi yang ditemukan.")
        else:
             # Tampilan Detail
            selected_wisata = df_wisata_app[df_wisata_app['no'] == st.session_state.selected_wisata_id].iloc[0]
            if st.button("⬅️ Kembali"):
                st.session_state.selected_wisata_id = None
                st.rerun()
            st.header(f"📍 Detail: {selected_wisata['nama']}")
            st.write(f"**Tipe:** {selected_wisata['type']}")
            st.write(f"**Rating:** {selected_wisata['vote_average']:.1f} (dari {selected_wisata['vote_count']} suara)")
            st.write(f"**Deskripsi:**")
            st.info(f"{selected_wisata['description']}")
            st.subheader("Beri Rating dan Komentar Anda")
            if st.session_state.get('logged_in'):
                with st.form("rating_form"):
                    rating = st.slider("Rating (1-10)", 1.0, 10.0, 7.5, 0.5)
                    comment = st.text_area("Komentar (Opsional)")
                    if st.form_submit_button("Kirim Rating", use_container_width=True):
                        submit_rating(st.session_state.user_id, selected_wisata['no'], rating, comment)
                        st.rerun()
            else:
                st.warning("Anda harus login untuk memberikan rating.")

else:
    st.error("Aplikasi gagal dimuat.")
