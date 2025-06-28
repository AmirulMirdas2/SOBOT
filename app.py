import streamlit as st
from transformers import pipeline
import PyPDF2
import io

# Menggunakan cache untuk memuat model agar tidak di-load berulang kali setiap ada interaksi user
# Ini sangat penting untuk performa aplikasi Streamlit.
@st.cache_resource
def load_model():
    """
    Memuat pipeline 'text2text-generation' dari Hugging Face.
    Model 'google/flan-t5-base' dipilih karena kemampuannya mengikuti instruksi.
    """
    try:
        generator = pipeline('text2text-generation', model='google/flan-t5-base')
        return generator
    except Exception as e:
        # Menampilkan error yang lebih informatif jika model gagal dimuat
        st.error(f"Gagal memuat model. Pastikan Anda terhubung ke internet. Error: {e}")
        return None

def extract_text_from_pdf(pdf_file):
    """
    Mengekstrak teks dari file PDF yang diunggah.
    """
    try:
        # Membaca file PDF dari memory
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        # Loop melalui setiap halaman dan ekstrak teksnya
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Gagal membaca file PDF. Error: {e}")
        return None

# --- UI Aplikasi Streamlit ---

# Konfigurasi judul dan ikon halaman
st.set_page_config(page_title="AI Pembuat Soal", page_icon="✍️")

st.title("✍️ AI Pembuat Soal dari PDF")
st.write("Unggah file PDF Anda, dan biarkan AI membuatkan soal latihan secara otomatis!")

# Memuat model AI dan menampilkannya sebagai pesan status
generator = load_model()
if generator:
    st.success("Model AI berhasil dimuat!")
else:
    st.stop() # Hentikan eksekusi jika model gagal dimuat

# Kolom untuk input dari pengguna
st.header("1. Unggah Dokumen Anda")
uploaded_file = st.file_uploader("Pilih file PDF", type="pdf")

st.header("2. Atur Soal Anda")
col1, col2 = st.columns(2)

with col1:
    # Opsi untuk memilih jenis soal
    question_type = st.selectbox(
        "Pilih Jenis Soal:",
        ("Pilihan Ganda", "Jawaban Singkat (Esai)", "Benar atau Salah")
    )

with col2:
    # Opsi untuk memilih jumlah soal
    num_questions = st.number_input(
        "Jumlah Soal:",
        min_value=1,
        max_value=15,
        value=5
    )

# Tombol untuk memulai proses pembuatan soal
submit_button = st.button("Buat Soal Sekarang!")

# --- Logika Backend ---

if submit_button and uploaded_file is not None:
    # Tampilkan spinner saat proses berjalan
    with st.spinner("AI sedang membaca PDF dan membuat soal... Mohon tunggu sebentar..."):
        # 1. Ekstrak teks dari PDF
        pdf_text = extract_text_from_pdf(uploaded_file)
        
        if pdf_text and len(pdf_text) > 100: # Pastikan teks cukup panjang untuk dibuat soal
            
            # 2. Membuat prompt (perintah) yang detail untuk model AI
            # Prompt ini akan disesuaikan berdasarkan pilihan user
            prompt_template = f"""
            Berdasarkan teks di bawah ini, buatlah {num_questions} soal dengan format "{question_type}".
            Jika formatnya "Pilihan Ganda", sertakan 4 pilihan (A, B, C, D) dan berikan kunci jawabannya.
            Jika formatnya "Benar atau Salah", berikan juga jawabannya.
            Pastikan pertanyaan dan jawaban hanya berasal dari informasi yang ada dalam teks.

            Teks:
            ---
            {pdf_text[:4000]} 
            ---
            """ 
            # Mengambil hanya 4000 karakter pertama untuk efisiensi

            try:
                # 3. Panggil model AI dengan prompt yang sudah dibuat
                results = generator(prompt_template, max_length=1024, num_beams=5, early_stopping=True)
                
                # 4. Tampilkan hasilnya
                st.subheader("🎉 Hasil Soal:")
                st.success("Soal berhasil dibuat!")
                st.markdown(results[0]['generated_text'])

            except Exception as e:
                st.error(f"Terjadi kesalahan saat membuat soal. Error: {e}")

        elif pdf_text:
            st.warning("Teks dalam PDF terlalu singkat untuk dibuatkan soal. Mohon gunakan PDF dengan konten yang lebih panjang.")
        else:
            st.error("Gagal mengekstrak teks dari PDF. File mungkin berupa gambar atau terproteksi.")

elif submit_button:
    st.warning("Mohon unggah file PDF terlebih dahulu!")