import logging
from pathlib import Path
import re
from urllib.parse import urlparse

import pandas as pd
from flask import Flask, render_template, request # buat ngatur routing nya
from sklearn.neighbors import NearestNeighbors #buat penyetaraan skala fitur
from sklearn.preprocessing import StandardScaler

# 1. KONFIGURASI APLIKASI

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024  # Batas ukuran request (10KB)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Dataset_Paket_Tour.csv"

# berisi kriteria penentu rekomendasi
FITUR = [
    "durasi_hari",
    "harga",
    "muslim_friendly",
    "halal_food",
    "wisata_alam",
    "wisata_budaya",
    "wisata_belanja",
    "wisata_sejarah",
    "winter",
    "spring",
    "autumn",
    "summer",
    "tingkat_aktivitas",
]

FITUR_BERTIPE_KATEGORI = [
    "muslim_friendly",
    "halal_food",
    "wisata_alam",
    "wisata_budaya",
    "wisata_belanja",
    "wisata_sejarah",
    "winter",
    "spring",
    "autumn",
    "summer",
]

BOBOT = {
    "durasi_hari": 1.2,
    "harga": 2.0,  # Bobot ditingkatkan agar faktor harga dominan
    "muslim_friendly": 1.4,
    "halal_food": 1.4,
    "wisata_alam": 1.5,
    "wisata_budaya": 1.5,
    "wisata_belanja": 1.5,
    "wisata_sejarah": 1.5,
    "winter": 1.0,
    "spring": 1.0,
    "autumn": 1.0,
    "summer": 1.0,
    "tingkat_aktivitas": 1.2,
}

# 2. MEMBACA DAN MEMVALIDASI DATASET

# buat mastiin data yang masuk itu bersih dan aman
def load_dataset():
    #buat mastiin file csv nya itu dijalur yang bener
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_PATH}")

    data = pd.read_csv(
        DATASET_PATH, sep=None, encoding="utf-8-sig", engine="python"
    )

    data.columns = (
        data.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    )

#mastiin semua kolom kaya fitur utama itu gada yang kurang
    kolom_wajib = FITUR + ["nama_paket", "negara", "link_paket"]

    kolom_hilang = [
        kolom for kolom in kolom_wajib if kolom not in data.columns
    ]

    if kolom_hilang:
        raise ValueError(f"Kolom dataset belum lengkap: {kolom_hilang}")

    # Konversi fitur menjadi numerik
    for kolom in FITUR:
        data[kolom] = pd.to_numeric(data[kolom], errors="coerce")

    # Hapus baris data yang ada angka kosongnya
    data = data.dropna(subset=FITUR).copy()

    data = data[
        data[FITUR]
        .apply(lambda col: col.map(lambda value: abs(value) != float("inf")))
        .all(axis=1)
    ].copy()

    for kolom in ["nama_paket", "negara", "link_paket"]:
        data[kolom] = data[kolom].fillna("").astype(str).str.strip()

    data = data[
        (data["nama_paket"] != "") & (data["negara"] != "")
    ].copy()

    if data.empty:
        raise ValueError("Dataset tidak memiliki paket valid.")

    if (data["harga"] < 0).any():
        raise ValueError("Harga paket tidak boleh negatif.")

    if (data["durasi_hari"] <= 0).any():
        raise ValueError("Durasi paket harus lebih dari 0.")

    if (data["tingkat_aktivitas"] < 1).any() or (
        data["tingkat_aktivitas"] > 3
    ).any():
        raise ValueError("Tingkat aktivitas harus berada pada rentang 1–3.")

    for kolom in FITUR_BERTIPE_KATEGORI:
        if not data[kolom].isin([0, 1]).all():
            raise ValueError(f"Kolom {kolom} hanya boleh berisi 0 atau 1.")

    return data.reset_index(drop=True)

#biar data siap dipake pas aplikasi jalan
df = load_dataset()

# Inisialisasi Scaler Global

#tujuannya supaya rentang angka yang jauh kaya harga dan durasi bisa di setarakan nilainya dan ga ngerusak perhitungan jarak
scaler = StandardScaler()
scaler.fit(df[FITUR])

# menentukan seberapa penting tiap kriteria saat dicocokkan dengan keinginan pengguna
bobot_array = [BOBOT[kolom] for kolom in FITUR]
#finish


# 3. FUNGSI PARSE BUDGET (DOKUMEN TITIK, KOMA, JUTA)
def parse_budget(text):
    text_lower = text.lower()

    # 1. Format kata 'juta' atau 'jt' (contoh: 12 juta, 4,5 juta, 4.5 jt)
    match_juta = re.search(
        r"(?:rp\.?\s*)?(\d+(?:[.,]\d+)?)\s*(juta|jt)\b", text_lower
    )
    if match_juta:
        angka = float(match_juta.group(1).replace(",", "."))
        return angka * 1_000_000

    # 2. Format nominal angka lengkap (dengan pemisah titik/koma atau tanpa pemisah)
    # Contoh: Rp 12.000.000 / Rp 12,000,000 / Rp12000000 / 12000000
    match_rp = re.search(
        r"(?:rp\.?\s*)?(\d{1,3}(?:[.,]\d{3})+|\d+)", text_lower
    )
    if match_rp:
        raw_str = match_rp.group(1)

        # Jika ada titik atau koma pemisah ribuan
        if "." in raw_str or "," in raw_str:
            angka_clean = re.sub(r"[.,]", "", raw_str)
            return float(angka_clean)

        # Jika angka murni (minimal 5 digit agar tidak bentrok dengan durasi hari)
        if len(raw_str) >= 5:
            return float(raw_str)

    return None


# =========================================================
# 4. FUNGSI PARSER INPUT PENGGUNA
# =========================================================


def parse_user_input(text):
    text_lower = text.lower()

    info = {
        "budget": 10_000_000.0,
        "durasi": 3,
        "muslim_friendly": 1,
        "halal_food": 1,
        "wisata_alam": 0,
        "wisata_budaya": 0,
        "wisata_belanja": 0,
        "wisata_sejarah": 0,
        "winter": 0,
        "spring": 0,
        "autumn": 0,
        "summer": 0,
        "tingkat_aktivitas": 2,
        "musim_disebut": False,
        "jenis_wisata_disebut": False,
        "budget_disebut": False,
        "durasi_disebut": False,
        "halal_disebut": False,
        "negara_disebut": None,
    }

    # --- DETEKSI NAMA NEGARA (DENGAN PEMETAAN VARIASI) ---
    pemetaan_negara = {
        "singapura": "singapura",
        "singapore": "singapura",
        "jepang": "jepang",
        "japan": "jepang",
        "korea": "korea",
        "china": "china",
        "cina": "china",
        "thailand": "thailand",
        "turki": "turki",
        "turkey": "turki",
        "indonesia": "indonesia",
        "malaysia": "malaysia",
        "vietnam": "vietnam",
        "eropa": "eropa",
        "europe": "eropa",
        "saudi": "saudi",
        "arab": "saudi",
        "rusia": "rusia",
        "moskow": "rusia",
        "kazakhstan": "kazakhstan",
        "inggris": "inggris",
        "uk": "inggris",
        "palestina": "palestina",
        "makkah": "makkah",
        "umroh": "makkah",
    }

    for kunci, nilai_resmi in pemetaan_negara.items():
        if kunci in text_lower:
            info["negara_disebut"] = nilai_resmi
            break

    # --- DETEKSI BUDGET ---
    budget = parse_budget(text_lower)
    if budget is not None:
        info["budget"] = budget
        info["budget_disebut"] = True

    # --- DETEKSI DURASI ---
    match_durasi = re.search(r"\b(\d+)\s*(hari|day|days)\b", text_lower)
    if match_durasi:
        info["durasi"] = max(1, min(int(match_durasi.group(1)), 365))
        info["durasi_disebut"] = True

    # --- DETEKSI NEGASI ---
    def is_negated(start):
        sebelum = text_lower[max(0, start - 30) : start]
        return bool(
            re.search(
                r"(tidak|nggak|enggak|gak|ga|bukan|tanpa)"
                r"(?:\s+\w+){0,2}\s*$",
                sebelum,
            )
        )

    # --- DETEKSI JENIS WISATA ---
    kategori = {
        "wisata_alam": r"\b(alam|nature|pantai|gunung)\b",
        "wisata_budaya": r"\b(budaya|culture)\b",
        "wisata_belanja": r"\b(belanja|shopping)\b",
        "wisata_sejarah": r"\b(sejarah|history|museum)\b",
    }

    for kolom, pola in kategori.items():
        match = re.search(pola, text_lower)
        if match and not is_negated(match.start()):
            info[kolom] = 1
            info["jenis_wisata_disebut"] = True

    if not info["jenis_wisata_disebut"]:
        info["wisata_alam"] = 1

    # --- DETEKSI MUSIM ---

    # --- DETEKSI MUSIM + NEGASI ---
    musim = {
        "winter": r"\b(winter|dingin|salju)\b",
        "spring": r"\b(spring|semi)\b",
        "autumn": r"\b(autumn|gugur)\b",
        "summer": r"\b(summer|panas)\b",
    }

    # Menyimpan musim yang secara eksplisit tidak diinginkan
    info["musim_ditolak"] = []

    pola_penolakan_musim = re.compile(
        r"\b("
        r"tidak mau|tak mau|gak mau|ga mau|nggak mau|"
        r"enggak mau|gamau|gak|ga|nggak|enggak|"
        r"tidak|bukan|hindari|hindarin|tanpa|"
        r"tidak ingin|gak ingin|ga ingin|nggak ingin|"
        r"enggak ingin|ga suka|gak suka|nggak suka|"
        r"enggak suka|kurang suka"
        r")\b"
    )

    for kolom, pola in musim.items():
        for match in re.finditer(pola, text_lower):
            # Periksa konteks sebelum nama musim
            sebelum = text_lower[max(0, match.start() - 60):match.start()]

            # Ambil klausa terakhir agar negasi tidak merembet
            # ke musim lain yang disebut setelah kata penghubung.
            klausa = re.split(
                r"[,;.!?]|\b(tapi|tetapi|namun|sedangkan|melainkan)\b",
                sebelum
            )[-1]

            if pola_penolakan_musim.search(klausa):
                if kolom not in info["musim_ditolak"]:
                    info["musim_ditolak"].append(kolom)
            else:
                info[kolom] = 1
                info["musim_disebut"] = True

    # Musim yang ditolak tidak boleh menjadi preferensi positif
    for kolom in info["musim_ditolak"]:
        info[kolom] = 0

    # Jika pengguna hanya menolak musim tertentu, jangan otomatis
    # menganggap musim yang ditolak sebagai pilihan.
    if not info["musim_disebut"]:
        for kolom in musim:
            info[kolom] = 0
    # --- DETEKSI HALAL ---
    non_halal = re.search(
        r"\b(non[- ]?halal|tidak halal|nggak halal|enggak halal|gak halal)\b",
        text_lower,
    )

    if non_halal:
        info["halal_food"] = 0
        info["muslim_friendly"] = 0
        info["halal_disebut"] = True
    elif re.search(r"\b(halal|muslim[- ]?friendly)\b", text_lower):
        info["halal_disebut"] = True

    # --- DETEKSI AKTIVITAS ---
    santai = re.search(r"\b(santai|relax|rileks)\b", text_lower)
    aktif = re.search(
        r"\b(aktif|petualang|adventure|trekking|hiking)\b", text_lower
    )

    if santai and not is_negated(santai.start()):
        info["tingkat_aktivitas"] = 1
    elif aktif and not is_negated(aktif.start()):
        info["tingkat_aktivitas"] = 3

    # Buat DataFrame pengguna
    user_data = {
        "durasi_hari": [info["durasi"]],
        "harga": [info["budget"]],
        "muslim_friendly": [info["muslim_friendly"]],
        "halal_food": [info["halal_food"]],
        "wisata_alam": [info["wisata_alam"]],
        "wisata_budaya": [info["wisata_budaya"]],
        "wisata_belanja": [info["wisata_belanja"]],
        "wisata_sejarah": [info["wisata_sejarah"]],
        "winter": [info["winter"]],
        "spring": [info["spring"]],
        "autumn": [info["autumn"]],
        "summer": [info["summer"]],
        "tingkat_aktivitas": [info["tingkat_aktivitas"]],
    }

    user_df = pd.DataFrame(user_data, columns=FITUR)
    return user_df, info


# =========================================================
# 5. FUNGSI REKOMENDASI (FILTER KETAT NEGARA & BUDGET)
# =========================================================


def recommend_packages(user_df, info, jumlah=3):
    df_filtered = df.copy()
    
    # Keluarkan paket yang musimnya secara eksplisit ditolak
    for kolom_musim in info.get("musim_ditolak", []):
        df_filtered = df_filtered[
            df_filtered[kolom_musim] != 1
        ].copy()

    # Berhenti jika tidak ada paket yang memenuhi filter
    if df_filtered.empty:
        return []

    # 1. PERBAIKAN: KUNCI FILTER NEGARA BILA DISEBUTKAN PENGGUNA
    if info.get("negara_disebut"):
        matched_negara = df_filtered[
            df_filtered["negara"]
            .str.lower()
            .str.contains(info["negara_disebut"], na=False)
        ]
        if not matched_negara.empty:
            df_filtered = matched_negara.reset_index(drop=True)

    # 2. FILTER BUDGET PADA DATASET YANG SUDAH DIFILTER NEGARA
    if info.get("budget_disebut"):
        budget_df = df_filtered[df_filtered["harga"] <= info["budget"]]
        if not budget_df.empty:
            df_filtered = budget_df.reset_index(drop=True)

    # 3. PROSES SCALER & KNN
    X_filtered_scaled = scaler.transform(df_filtered[FITUR])
    X_filtered_weighted = X_filtered_scaled * bobot_array

    jumlah_kandidat = min(30, len(df_filtered))

    knn = NearestNeighbors(
        n_neighbors=jumlah_kandidat, metric="euclidean", algorithm="auto"
    )
    knn.fit(X_filtered_weighted)

    user_scaled = scaler.transform(user_df)
    user_weighted = user_scaled * bobot_array

    distances, indices = knn.kneighbors(
        user_weighted, n_neighbors=jumlah_kandidat
    )

    kandidat = []

    for jarak, indeks in zip(distances[0], indices[0]):
        paket = df_filtered.iloc[indeks]
        harga = float(paket["harga"])

        durasi_paket = float(paket["durasi_hari"])

        penalti_durasi = 0
        if info.get("durasi_disebut"):
            selisih_durasi = abs(durasi_paket - info["durasi"])
            if selisih_durasi > 0:
                penalti_durasi = selisih_durasi * 2.5
        

        penalti_budget = 0
    # Penalti harga hanya aktif JIKA user menyebutkan budget secara spesifik
        if info.get("budget_disebut"):
            if harga > info["budget"]:
                selisih = (harga - info["budget"]) / max(info["budget"], 1)
                penalti_budget = selisih * 10.0

        skor = float(jarak) + penalti_budget + penalti_durasi

        if info["musim_disebut"]:
            musim_diminta = ["winter", "spring", "autumn", "summer"]
            if not any(
                info[musim] == 1 and int(paket[musim]) == 1
                for musim in musim_diminta
            ):
                skor += 2.0

        if info["halal_disebut"] and info["halal_food"] == 1:
            if (
                int(paket["halal_food"]) != 1
                or int(paket["muslim_friendly"]) != 1
            ):
                skor += 3.0

        kandidat.append(
            {
                "index": int(indeks),
                "jarak": float(jarak),
                "skor": skor,
                "harga": harga,
                "sesuai_budget": harga <= info["budget"],
            }
        )

    kandidat.sort(key=lambda item: item["skor"])

    hasil = []
    for item in kandidat[:jumlah]:
        paket = df_filtered.iloc[item["index"]]

        link = str(paket["link_paket"]).strip()
        parsed_url = urlparse(link)
        if parsed_url.scheme not in ("http", "https"):
            link = ""

        kemiripan = (1 / (1 + item["jarak"])) * 100

        hasil.append(
            {
                "nama": paket["nama_paket"],
                "negara": paket["negara"],
                "durasi": int(paket["durasi_hari"]),
                "harga": f"Rp{item['harga']:,.0f}",
                "link": link,
                "skor": round(item["skor"], 3),
                "kemiripan": round(kemiripan, 1),
                "sesuai_budget": item["sesuai_budget"],
            }
        )

    return hasil


# 6. ROUTE FLASK

# 6. ROUTE FLASK

@app.route("/", methods=["GET", "POST"])
def index():
    rekomendasi_list = []
    paket_utama = None
    user_prompt = ""
    parsed_info = None
    error_message = None

    if request.method == "POST":
        user_prompt = request.form.get("user_prompt", "").strip()

        if not user_prompt:
            error_message = "Masukkan preferensi liburan terlebih dahulu."

        elif len(user_prompt) > 1000:
            error_message = "Input terlalu panjang. Maksimal 1000 karakter."

        else:
            try:
                user_df, parsed_info = parse_user_input(user_prompt)

                rekomendasi_list = recommend_packages(
                    user_df, parsed_info, jumlah=3
                )

                if rekomendasi_list:
                    paket_utama = rekomendasi_list[0]
                else:
                    error_message = (
                        "Belum ada paket yang cocok dengan kriteria Anda."
                    )

            except Exception:
                logger.exception("Gagal memproses permintaan rekomendasi.")
                error_message = (
                    "Terjadi kesalahan saat mencari paket. Silakan coba lagi."
                )

    # Ambil daftar unik negara dan link paket pertama yang mewakili dari dataset untuk running text yang bisa diklik
    running_destinations = []
    for _, row in df.drop_duplicates(subset=["negara"]).iterrows():
        link_val = str(row["link_paket"]).strip()
        if not link_val.startswith("http"):
            link_val = "#"
        running_destinations.append({
            "name": str(row["negara"]).strip(),
            "link": link_val
        })

    return render_template(
        "index.html",
        rekomendasi=rekomendasi_list,
        paket_utama=paket_utama,
        user_prompt=user_prompt,
        parsed_info=parsed_info,
        error_message=error_message,
        running_destinations=running_destinations,
    )

# 7. MENJALANKAN SERVER

if __name__ == "__main__":
    app.run(debug=True)