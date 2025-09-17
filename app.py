
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import date, timedelta

# --- Konfigurasi Awal & Konstanta ---
WINDOW_SIZE = 30 # Sesuaikan dengan window_size yang Anda gunakan saat training

# --- Fungsi untuk Prediksi Multi-Langkah ---
def predict_future(model, initial_sequence, n_steps):
    """
    Fungsi untuk melakukan prediksi berulang (recursive forecasting).
    """
    future_preds = []
    current_sequence = initial_sequence.copy()
    for _ in range(n_steps):
        # Siapkan input data sesuai format model LSTM
        input_data = current_sequence.reshape((1, WINDOW_SIZE, 1))
        
        # Lakukan prediksi untuk 1 hari ke depan
        # verbose=0 agar log TensorFlow tidak muncul di konsol deployment
        next_pred = model.predict(input_data, verbose=0)[0, 0]
        
        # Simpan hasil prediksi
        future_preds.append(next_pred)
        
        # Perbarui sequence untuk prediksi hari berikutnya
        current_sequence = np.append(current_sequence[1:], next_pred)
        
    return future_preds

# --- Fungsi untuk Memuat Aset (Model, Scaler, Data) ---
@st.cache_resource
def load_assets():
    """
    Fungsi untuk memuat semua aset yang dibutuhkan.
    Menggunakan caching agar hanya dimuat sekali.
    """
    # Muat model Keras. compile=False untuk menghindari error saat loading.
    model = tf.keras.models.load_model('best_model.h5', compile=False)
    
    # Muat objek scaler
    scaler = joblib.load('scaler.joblib')
    
    # Muat data historis dari file Excel
    df = pd.read_excel('harga_telur_clean.xlsx', parse_dates=['Tanggal'])
    df = df.set_index('Tanggal')
    
    return model, scaler, df

# --- Tampilan Utama Aplikasi Streamlit ---
st.title('🥚 Sistem Prediksi Harga Telur di Wonosobo')

# Tampilkan spinner saat aset sedang dimuat untuk pertama kali
with st.spinner("Sedang memuat model dan data historis, harap tunggu..."):
    model, scaler, df = load_assets()

st.success("Model dan data berhasil dimuat! Aplikasi siap digunakan.")
st.write("Pilih tanggal mulai dan durasi prediksi untuk melihat estimasi harga di masa depan.")

# --- Input dari Pengguna ---
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Pilih Tanggal Mulai Prediksi",
        value=date.today() + timedelta(days=1),
        min_value=date.today() + timedelta(days=1)
    )
with col2:
    n_days = st.slider("Pilih Durasi Prediksi (hari)", min_value=1, max_value=30, value=7)

# --- Tombol dan Logika Prediksi ---
if st.button('🚀 Buat Prediksi'):
    try:
        # PENTING: Ganti 'Harga' jika nama kolom di file Excel Anda berbeda!
        column_name = 'Harga'
        
        # Ambil data terakhir dari DataFrame sebagai input awal
        last_data = df[column_name][df.index < pd.to_datetime(start_date)].tail(WINDOW_SIZE).values
        
        if len(last_data) < WINDOW_SIZE:
            st.error(f"Error: Data historis tidak cukup. Dibutuhkan {WINDOW_SIZE} hari data sebelum tanggal mulai.")
        else:
            with st.spinner(f'Membuat prediksi untuk {n_days} hari ke depan...'):
                # Scaling input data
                last_data_scaled = scaler.transform(last_data.reshape(-1, 1)).flatten()
                
                # Lakukan prediksi
                predictions_scaled = predict_future(model, last_data_scaled, n_days)
                
                # Kembalikan prediksi ke skala semula
                predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
                
                # Siapkan tanggal untuk hasil prediksi
                prediction_dates = pd.to_datetime([start_date + timedelta(days=i) for i in range(n_days)])
                
                # Buat DataFrame hasil
                results_df = pd.DataFrame({
                    'Tanggal': prediction_dates,
                    'Prediksi Harga': predictions.astype(int)
                })
                
                st.success('Prediksi berhasil dibuat!')
                
                # --- Tampilkan Hasil Prediksi ---
                st.subheader(f'Hasil Prediksi untuk {n_days} Hari ke Depan')
                
                if not results_df.empty:
                    # Tampilkan grafik garis
                    chart_df = results_df.set_index('Tanggal')
                    st.line_chart(chart_df['Prediksi Harga'])
                    
                    # Tampilkan tabel data
                    st.dataframe(results_df)
                else:
                    st.warning("Tidak ada data untuk ditampilkan. Hasil prediksi kosong.")
                    
    except KeyError:
        st.error(f"Error: Kolom '{column_name}' tidak ditemukan di file Excel. Harap periksa kembali nama kolom data harga Anda.")
    except Exception as e:
        st.error(f"Terjadi kesalahan tak terduga: {e}")
