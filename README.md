# Handwritten Digit Recognition

Aplikasi sederhana untuk mengenali angka tulisan tangan 0–9 menggunakan PyTorch dan dataset `load_digits` dari Scikit-learn.

## Identitas
- Nama: Aldo Bayu Prastyo
- NPM: 202343501845
- Mata Kuliah: Machine Learning

## Isi proyek
- `app.py`: antarmuka aplikasi Streamlit.
- `digit_model.pth`: bobot model PyTorch yang digunakan aplikasi.
- `train_model.py`: kode untuk melatih ulang model.
- `requirements.txt`: daftar pustaka.

## Menjalankan aplikasi
1. Instal Python.
2. Buka terminal di folder proyek.
3. Jalankan `pip install -r requirements.txt`.
4. Jalankan `streamlit run app.py`.
5. Buka alamat lokal yang ditampilkan Streamlit.

## Model
Model dilatih menggunakan dataset angka tulisan tangan `load_digits` (gambar 8×8 piksel). Pembagian data latih dan uji menggunakan `test_size=0.2`, `random_state=42`, dan `stratify=y`. Akurasi pada data uji saat pembuatan paket: **96.94%**.

## Catatan akademik
Notebook awal yang dibagikan menggunakan StandardScaler, PCA, t-SNE, LDA, dan evaluasi Logistic Regression. File `.pth` di proyek aplikasi ini adalah model klasifikasi PyTorch tambahan yang dilatih pada dataset `load_digits`; bukan file hasil ekspor langsung dari eksperimen PCA/Logistic Regression di notebook awal.
