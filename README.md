# Laporan Proyek Machine Learning - Berwyn Izzut Taghyir

## Overview Proyek
Seiring dengan kemajuan teknologi dan meningkatnya produksi konten digital, jumlah film yang dirilis setiap tahunnya juga meningkat pesat. Hal ini memberikan banyak pilihan kepada penikmat film, tetapi di sisi lain juga menimbulkan tantangan dalam menemukan film yang benar-benar sesuai dengan selera dan preferensi pribadi masing-masing individu. Sistem rekomendasi hadir sebagai solusi untuk membantu pengguna menyaring informasi dan menemukan konten yang relevan berdasarkan preferensi mereka.

Menurut studi yang dilakukan oleh Harper dan Konstan (2016) terkait dataset MovieLens, pemahaman mendalam mengenai sejarah dan konteks data rating pengguna sangat krusial dalam membangun sistem rekomendasi yang efektif. Mereka menekankan bahwa dataset seperti MovieLens, yang kaya akan interaksi pengguna, memungkinkan penerapan berbagai teknik, termasuk *collaborative filtering*, untuk menghasilkan rekomendasi yang dipersonalisasi. Proyek ini mengadopsi pendekatan tersebut dengan menerapkan *collaborative filtering* berbasis *deep learning* dan *content-based filtering* yang menggabungkan kemiripan konten (judul dan genre) dengan preferensi pengguna serta kemiripan genre murni.

**Pentingnya Proyek**<br>
Membangun sistem rekomendasi film tidak hanya bertujuan untuk meningkatkan pengalaman menonton pengguna, tetapi juga berdampak signifikan pada peningkatan *engagement* pengguna terhadap platform penyedia layanan film. Bagi industri perfilman, sistem ini dapat membantu mempromosikan film-film yang mungkin kurang populer namun relevan bagi segmen pengguna tertentu, sehingga meningkatkan diversitas konsumsi konten.

Proyek ini penting untuk diselesaikan karena:
-   Membantu pengguna mengatasi masalah *information overload* dalam memilih film.
-   Meningkatkan efisiensi pengguna dalam menemukan film yang sesuai dengan selera.
-   Memberikan pengalaman menonton yang lebih personal dan memuaskan.

Proyek ini dirancang dengan membangun dua jenis sistem rekomendasi berbasis machine learning, yaitu *collaborative filtering* menggunakan model *deep learning* dan *content-based filtering* yang diperkaya dengan heuristik berbasis preferensi pengguna dan kemiripan genre, dengan memanfaatkan dataset MovieLens yang diakses melalui Kaggle.

**Referensi:**<br>
Harper, F. M., & Konstan, J. A. (2016). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)*, 5(4), 1-19. https://doi.org/10.1145/2827872

## Business Understanding
### **Problem Statements:**
1.  Pengguna seringkali kesulitan menemukan film baru yang sesuai dengan selera pribadi mereka di tengah banyaknya pilihan film yang tersedia, terutama jika mereka tidak memiliki banyak waktu untuk eksplorasi.
2.  Rekomendasi film yang bersifat generik atau tidak dipersonalisasi (misalnya, hanya berdasarkan popularitas umum) kurang efektif dalam memenuhi ekspektasi pengguna, sehingga mengurangi kepuasan dan pengalaman menonton.
3.  Pengguna yang telah memberikan rating pada film-film sebelumnya mengharapkan informasi tersebut dimanfaatkan untuk mendapatkan saran film lain yang relevan dengan preferensi mereka, namun seringkali platform belum memaksimalkannya secara optimal.

### **Goals:**
1.  Mengembangkan sistem rekomendasi film yang dapat membantu pengguna menemukan film baru yang sesuai dengan preferensi historis mereka secara efisien, sehingga meningkatkan kepuasan menonton.
2.  Mengembangkan sistem rekomendasi film yang dapat memberikan daftar rekomendasi film top-N yang secara spesifik disesuaikan dengan profil dan preferensi unik masing-masing pengguna menggunakan pendekatan *collaborative filtering* berbasis *deep learning*.
3.  Mengembangkan sistem rekomendasi *content-based* yang tidak hanya mengandalkan kemiripan konten (genre dan judul) tetapi juga memperhitungkan popularitas film di antara pengguna dengan selera serupa dan kemiripan genre murni untuk memberikan rekomendasi yang lebih relevan dan terdiversifikasi.

### **Solution Statements:**
Untuk mencapai tujuan tersebut, proyek ini akan menggunakan dua pendekatan solusi utama:

1.  **Collaborative Filtering (Deep Learning)**
    Pendekatan ini menggunakan model jaringan saraf untuk mempelajari representasi laten (embedding) dari pengguna dan film berdasarkan histori rating. Model *RecommenderNet* yang dibangun akan memprediksi rating yang mungkin diberikan pengguna terhadap film yang belum ditonton, dan merekomendasikan film dengan prediksi rating tertinggi.
2.  **Content-Based Filtering (dengan Heuristik Preferensi Pengguna dan Kemiripan Genre)**
    Pendekatan ini merekomendasikan film berdasarkan kemiripan konten (judul dan genre) menggunakan TF-IDF dan *cosine similarity*. Sistem ini diperkaya dengan heuristik yang menghitung skor gabungan (`score`) berdasarkan popularitas film di antara pengguna dengan selera serupa dan popularitas umum. Skor ini kemudian dikombinasikan dengan skor kemiripan genre murni (`sim_score`) antara film input dan film kandidat untuk menghasilkan `final_score` yang dinormalisasi.

## Data Understanding
### Dataset
Proyek ini menggunakan dataset MovieLens yang tersedia di Kaggle ([Movie Recommendation System Dataset](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system/data)). Dataset ini populer untuk penelitian sistem rekomendasi karena berisi informasi rating film oleh pengguna serta metadata film.

Dataset ini bertujuan untuk memungkinkan pembangunan sistem yang memahami preferensi pengguna berdasarkan histori rating mereka terhadap film. Target dari dataset ini adalah untuk menghasilkan rekomendasi film yang relevan. Dataset ini terdiri dari 2 file utama: `movies.csv` dan `ratings.csv`.

**Deskripsi Variabel**
**Dataset `movies.csv`**<br>
Total data: 62.423 data (setelah pembersihan `(no genres listed)` menjadi 57.361 data)<br>
Berikut adalah daftar fitur (variabel) yang terdapat dalam dataset:
-   `movieId`: Nomor unik identifikasi film (Integer)
-   `title`: Judul Film, seringkali menyertakan tahun rilis (String)
-   `genres`: Genre film, dipisahkan oleh `|` (String)
-   `clean_title`: Judul film yang sudah dibersihkan (String, dibuat saat preprocessing)
-   `genres_list`: Genre film yang sudah dibersihkan (String, spasi sebagai pemisah, dibuat saat preprocessing)

**Dataset `ratings.csv`**<br>
Total data: 25.000.095 data<br>
Berikut adalah daftar fitur (variabel) yang terdapat dalam dataset:
-   `userId`: ID unik pengguna (Integer)
-   `movieId`: Nomor unik identifikasi film (Integer)
-   `rating`: Rating film dari pengguna (Float, skala 0.5-5)
-   `timestamp`: Waktu pemberian rating (Integer, dihapus saat preprocessing)

### Tahapan Awal Eksplorasi Data<br>
Beberapa tahapan eksplorasi dan pemahaman data yang dilakukan:
#### 1. Load Dataset<br>
Dataset `movies.csv` dan `ratings.csv` dimuat ke dataframe pandas.
#### 2. Melihat Struktur Dataset<br>
Menggunakan fungsi `info()` pada kedua dataframe.<br>
**Insight:**
-   Dataset `movies` memiliki 3 kolom (`movieId`, `title`, `genres`) dengan tipe data `int64` dan `object`. Memori yang digunakan sekitar 1.4+ MB.
-   Dataset `ratings` memiliki 4 kolom (`userId`, `movieId`, `rating`, `timestamp`) dengan tipe data `int64` dan `float64`. Memori yang digunakan sekitar 762.9 MB.
#### 3. Cek Missing Values dan Duplikat<br>
Dilakukan dengan fungsi `isna().sum()` dan `duplicated().sum()`.<br>
**Insight:**
-   Tidak ada nilai null pada dataset `movies` dan `ratings` pada kolom-kolom utama.
-   Tidak ada data duplikat pada kedua dataset.
-   Terdapat 5062 film dengan `(no genres listed)` pada kolom `genres`.
#### 4. Univariate Exploratory Data Analysis
1.  Jumlah Film, Judul, Genre Unik<br>
    **Insight:**
    -   Jumlah Film Unik: 62.423
    -   Jumlah Judul Film Unik: 62.325
    -   Jumlah Kombinasi Genre Unik: 1.639
2.  Jumlah UserID Unik, Film yang Diulas, Total Rating<br>
    **Insight:**
    -   Jumlah UserID Unik: 162.541
    -   Jumlah Film yang Diulas: 59.047
    -   Jumlah Data Rating: 25.000.095
3.  Top Genres<br>
    Menampilkan distribusi jumlah film per genre.<br>
    ![Top Genres](https://github.com/Wynnzzz/movie-recommendation-system/raw/blob/8c7d9dc0c281856a4883e0bea9101fd200fa640e/img/output1.png) <br>
    **Insight:**
    -   Genre Drama, Comedy, dan Thriller merupakan genre yang paling banyak muncul dalam dataset. Action dan Romance juga cukup dominan.
4.  Top 10 Film dengan Rata-Rata Rating Tertinggi<br>
    Setelah menggabungkan `movies` dan `ratings`, dihitung rata-rata rating per film.<br>
    **Insight:**
    -   Banyak film dengan rata-rata rating 5.0, namun ini bisa jadi karena jumlah pemberi ratingnya sedikit. Contohnya "'Master Harold' ... And the Boys (2010)".

## Data Preparation
Tahapan ini mencakup langkah-langkah untuk mempersiapkan data sebelum digunakan dalam proses modeling agar data dalam kondisi bersih, konsisten, dan sesuai untuk masing-masing algoritma.
1.  **Pembersihan Kolom `genres` pada `df_movies`**:
    -   Mengganti karakter `|` dengan spasi.
    -   Menghapus baris film dengan `genres` berisi `(no genres listed)`.
    -   Membuat kolom baru `genres_list` yang identik dengan kolom `genres` yang sudah dibersihkan.
    *Alasan:* Memastikan format genre konsisten dan menghilangkan data yang tidak informatif untuk content-based filtering.

2.  **Pembersihan Kolom `title` pada `df_movies`**:
    -   Membuat fungsi `clean_title` untuk mengubah judul menjadi huruf kecil, menghapus tahun dalam tanda kurung, menghapus karakter spesial, dan menghilangkan spasi berlebih.
    -   Menerapkan fungsi ini untuk membuat kolom baru `clean_title`.
    *Alasan:* Normalisasi judul film untuk meningkatkan akurasi pencarian dan perhitungan kemiripan berbasis judul.

3.  **Menghapus Kolom `timestamp` dari `df_ratings`**:
    *Alasan:* Kolom ini tidak digunakan dalam model.

4.  **Menggabungkan Dataset `df_ratings` dan `df_movies`**:
    -   Menggabungkan kedua dataframe berdasarkan `movieId` menggunakan `pd.merge()`.
    *Alasan:* Untuk mendapatkan informasi film (judul, genre) bersama dengan data rating pengguna dalam satu dataframe (`combined_data`) yang akan digunakan pada Content-Based Filtering.

5.  **Persiapan Data untuk Collaborative Filtering (Deep Learning)**:
    -   **Encoding User dan Movie ID**: Mengubah `userId` dan `movieId` pada `df_ratings` menjadi integer berurutan (encoded ID) untuk digunakan sebagai input pada lapisan Embedding model deep learning. Mapping asli disimpan untuk menerjemahkan kembali.
        *Alasan:* Model deep learning memerlukan input numerik kategorikal yang dimulai dari 0.
    -   **Normalisasi Rating**: Mengubah skala rating pada `df_ratings` menjadi rentang [0, 1] dengan mengurangkan `min_rating` dan membagi dengan (`max_rating` - `min_rating`).
        *Alasan:* Membantu stabilisasi proses training model deep learning.
    -   **Pengacakan Data (Shuffling)**: Mengacak urutan baris pada `df_ratings` menggunakan `df_ratings.sample(frac=1, random_state=42)`.
        *Alasan:* Untuk memastikan bahwa pembagian data latih dan validasi tidak bias oleh urutan data asli (misalnya, jika data diurutkan berdasarkan waktu atau pengguna). Ini penting untuk mendapatkan evaluasi model yang lebih representatif.
    -   **Pembagian Data**: Membagi data rating yang telah diacak (fitur: encoded user & movie ID, target: normalized rating) menjadi data latih (80%) dan data validasi (20%).
        *Alasan:* Untuk melatih model dan mengevaluasi performanya pada data yang tidak terlihat saat pelatihan.

6.  **Persiapan Data untuk Content-Based Filtering**:
    -   **TF-IDF Vectorization**: Menerapkan `TfidfVectorizer` pada kolom `clean_title` dan `genres_list` dari `df_movies` dengan `ngram_range=(1,2)`.
    *Alasan:* Mengubah data teks (judul dan genre) menjadi representasi vektor numerik yang dapat digunakan untuk menghitung *cosine similarity* antar film.

## Modeling and Results

### **1. Collaborative Filtering (Deep Learning)**
Pendekatan ini menggunakan model jaringan saraf `RecommenderNet` untuk mempelajari pola dari interaksi pengguna-film dan menghasilkan rekomendasi.

<br>Model terdiri dari lapisan Embedding untuk pengguna dan film, lapisan bias, dan fungsi aktivasi sigmoid untuk prediksi rating ternormalisasi.

**Kelebihan:**
-   Mampu menangkap pola kompleks.
-   Potensial menghasilkan rekomendasi personal yang akurat.

**Kekurangan:**
-   Masalah *cold-start*.
-   Membutuhkan data interaksi yang cukup.

**Untuk Training model**:
-   Optimizer `Adam` (learning rate 1e-4), loss `binary_crossentropy`, metrik `RootMeanSquaredError`.
-   Dilatih selama 20 epoch dengan callbacks (ReduceLROnPlateau, EarlyStopping, ModelCheckpoint).

**Hasil Rekomendasi** (Contoh untuk User ID 8868):<br>
| No | Title                                  | Genres                   |
|----|----------------------------------------|--------------------------|
| 1  | Harakiri (Seppuku) (1962)             | Drama                    |
| 2  | The Blue Planet (2001)                | Documentary              |
| 3  | Planet Earth (2006)                   | Documentary              |
| 4  | Life (2009)                           | Documentary              |
| 5  | Over the Garden Wall (2013)           | Adventure, Animation, Drama |
| ... | ...                                    | ...                      |

### **2. Content-Based Filtering (dengan Heuristik Preferensi Pengguna dan Kemiripan Genre)**
Pendekatan ini merekomendasikan film berdasarkan kemiripan konten (judul dan genre) yang diperkaya dengan informasi preferensi pengguna lain dan kemiripan genre murni.

<br>Konten yang dianalisis adalah `clean_title` dan `genres_list` menggunakan TF-IDF dan *cosine similarity*. Fungsi `scores_calculator` menghitung:
1.  **Skor Preferensi Pengguna Serupa (`similar_user_recs`)**: Normalisasi dari jumlah pengguna (yang menyukai film input) yang juga menyukai film kandidat.
2.  **Skor Preferensi Umum (`all_users_recs`)**: Normalisasi dari jumlah semua pengguna (yang menyukai film-film yang disukai oleh pengguna serupa) yang juga menyukai film kandidat.
3.  **Skor Gabungan Awal (`score`)**: Kombinasi berbobot (0.6 untuk `similar_user_recs` dan 0.4 untuk `all_users_recs`), dengan penyesuaian bobot jika genre film kandidat mirip dengan film input (bobot 1.5 untuk `similar_user_recs` dan 0.9 untuk `all_users_recs`).
4.  **Skor Kemiripan Genre Murni (`sim_score`)**: *Cosine similarity* antara vektor TF-IDF genre film input dan film kandidat.
5.  **Skor Akhir (`final_score`)**: Kombinasi berbobot (0.7 untuk `score` dan 0.3 untuk `sim_score`), kemudian dinormalisasi ke rentang [0,1] menggunakan `MinMaxScaler`.

**Kelebihan:**
-   Mempertimbangkan kemiripan konten dan popularitas kontekstual.
-   Transparan, dengan skor kemiripan genre yang eksplisit.
-   Lebih tahan terhadap *over-specialization*.

**Kekurangan:**
-   Kualitas bergantung pada metadata dan efektivitas heuristik.
-   Perhitungan skor bisa kompleks dan bobot perlu di-tuning.

Langkahnya: pengguna memasukkan judul -> `search_by_title` mencari kandidat -> pengguna memilih -> `scores_calculator` menghitung skor -> `recommendation_results` menampilkan top-N.

Contoh: Jika pengguna mencari "Toy Story" dan memilih "Toy Story 2 (1999)":

**Hasil Rekomendasi:**<br>
| No | Recommended Movie                        | Genres                                       | score    | sim_score | final_score |
|----|--------------------------------------------|----------------------------------------------|----------|-----------|-------------|
| 1  | Toy Story 2 (1999)                        | Adventure, Animation, Children, Comedy, Fantasy | 0.983611 | 1.000000  | 1.000000    |
| 2  | Toy Story (1995)                          | Adventure, Animation, Children, Comedy, Fantasy | 0.874127 | 1.000000  | 0.922469    |
| 3  | Monsters, Inc. (2001)                     | Adventure, Animation, Children, Comedy, Fantasy | 0.409754 | 1.000000  | 0.593626    |
| 4  | Shrek (2001)                              | Adventure, Animation, Children, Comedy, Fantasy, Romance | 0.420460 | 0.886559  | 0.566780    |
| 5  | Finding Nemo (2003)                       | Adventure, Animation, Children, Comedy       | 0.376863 | 0.858481  | 0.527384    |
| 6  | Shawshank Redemption, The (1994)          | Crime, Drama                                 | 0.741638 | 0.000000  | 0.525157    |
| 7  | Matrix, The (1999)                        | Action, Sci-Fi, Thriller                     | 0.677699 | 0.000000  | 0.479879    |
| 8  | Star Wars: Episode IV - A New Hope (1977) | Action, Adventure, Sci-Fi                    | 0.638385 | 0.087919  | 0.478722    |
| 9  | Incredibles, The (2004)                   | Action, Adventure, Animation, Children, Comedy | 0.337925 | 0.773725  | 0.474088    |
| 10 | Pulp Fiction (1994)                       | Comedy, Crime, Drama, Thriller               | 0.646498 | 0.039931  | 0.469903    |

## Evaluation

### **1. Collaborative Filtering (Deep Learning)**
**Metrik Evaluasi yang Digunakan**
<br>Root Mean Squared Error (RMSE) digunakan sebagai metrik utama untuk mengevaluasi seberapa akurat model `RecommenderNet` dalam memprediksi rating pengguna.
**Cara Kerja Metrik RMSE:**
<br>RMSE menghitung akar kuadrat dari rata-rata kuadrat selisih antara nilai rating aktual ($y_i$) yang telah dinormalisasi (skala 0-1) dan nilai rating prediksi ($\hat{y}_i$) dari model. Langkah-langkahnya adalah sebagai berikut:
1.  Untuk setiap pasangan pengguna-film dalam set data validasi, hitung selisih antara rating aktual ternormalisasi dan rating prediksi dari model: $(y_i - \hat{y}_i)$.
2.  Kuadratkan selisih tersebut: $(y_i - \hat{y}_i)^2$. Langkah ini memberikan bobot lebih besar pada kesalahan prediksi yang lebih besar.
3.  Hitung rata-rata dari semua nilai kuadrat selisih tersebut untuk mendapatkan Mean Squared Error (MSE).
4.  Ambil akar kuadrat dari MSE untuk mendapatkan RMSE.
Nilai RMSE memiliki skala yang sama dengan data target (rating ternormalisasi), sehingga memudahkan interpretasi. Semakin kecil nilai RMSE, semakin dekat prediksi model dengan nilai aktual, yang mengindikasikan akurasi prediksi yang lebih baik.

**Kenapa Menggunakan RMSE?**
<br>RMSE dipilih karena memberikan gambaran yang jelas mengenai besarnya error rata-rata prediksi model dalam satuan yang sama dengan data target. Metrik ini sensitif terhadap error besar, yang penting dalam konteks prediksi rating. Meskipun fungsi loss yang digunakan selama training adalah `binary_crossentropy` (karena output sigmoid), RMSE tetap menjadi metrik yang intuitif untuk menilai performa prediksi rating.

<br>**Rumus:**<br>

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

<br>
**Hasil RMSE Model:**<br>
Visualisasi learning curve RMSE selama proses training (20 epoch) disajikan di bawah ini:<br>

![DL Model RMSE](https://github.com/Wynnzzz/movie-recommendation-system/raw/blob/main/img/output2.png)<br>

**Insight:**
-   Pada akhir epoch ke-20, model `RecommenderNet` memperoleh nilai `root_mean_squared_error` sebesar **0.1857** untuk data latih dan **0.1874** untuk data validasi.
-   Perbedaan yang kecil antara RMSE pada data latih dan data validasi menunjukkan bahwa model tidak mengalami overfitting yang signifikan dan memiliki kemampuan generalisasi yang baik terhadap data yang belum pernah dilihat sebelumnya. Penurunan RMSE yang konsisten pada kedua set data selama epoch menunjukkan bahwa proses pembelajaran model berjalan efektif.

### 2. Content-Based Filtering (dengan Heuristik Preferensi Pengguna dan Kemiripan Genre)

**Metrik Evaluasi yang Digunakan**  
Untuk sistem Content-Based Filtering ini, evaluasi performa dalam menghasilkan daftar rekomendasi top-N dilakukan menggunakan metrik **Precision@K** dan **Recall@K**. Metrik ini menilai kualitas daftar rekomendasi yang dihasilkan oleh `final_score`, yang merupakan skor gabungan dari preferensi pengguna dan kemiripan konten.

**Cara Kerja Metrik Precision@K dan Recall@K:**
1.  **Pemilihan Pengguna dan Pembagian Data (Simulasi):**
    *   Dalam implementasi evaluasi ini, dipilih satu `user_id` (misalnya, 123) sebagai sampel untuk pengujian.
    *   **Set Film yang Disukai Pengguna (Ground Truth / Hold-out):** Diambil semua film yang telah diberi rating tinggi (rating >= 4.0) oleh `user_id` tersebut dari `df_ratings`. Daftar `movieId` dari film-film ini (`hold_out_ids`) dijadikan sebagai ground truth item yang relevan.
    *   **Input untuk Rekomendasi:** Untuk mensimulasikan skenario pengguna, sebuah film input (misalnya, "Toy Story 2 (1999)" yang merupakan pilihan ke-3 dari hasil pencarian "Toy Story") digunakan untuk memicu sistem rekomendasi Content-Based Filtering.

2.  **Generasi Rekomendasi:**
    *   Berdasarkan film input yang dipilih ("Toy Story 2 (1999)"), sistem menghasilkan daftar top-K rekomendasi (K=10) film menggunakan fungsi `recommendation_results`. Daftar `movieId` dari rekomendasi ini (`recommended_ids`) kemudian dievaluasi.

3.  **Perhitungan Metrik:**
    *   **Item Relevan yang Direkomendasikan:** Dihitung jumlah film dari `recommended_ids` (top-K) yang juga terdapat dalam `hold_out_ids` (film yang benar-benar disukai pengguna).
    *   **Precision@K:** Mengukur seberapa banyak film yang direkomendasikan dalam K teratas yang memang relevan (disukai) oleh pengguna. Jika sistem merekomendasikan K film, dan `r` di antaranya ada di `hold_out_ids`, maka Precision@K = `r/K`. Metrik ini menunjukkan ketepatan rekomendasi.
    *   **Recall@K:** Mengukur seberapa banyak film relevan (yang disukai pengguna) yang berhasil ditemukan dalam K rekomendasi teratas. Jika ada `R` film total di `hold_out_ids`, dan `r` di antaranya ada di top-K rekomendasi, maka Recall@K = `r/R`. Metrik ini menunjukkan kelengkapan sistem dalam menemukan item relevan.

<br>**Rumus:**<br>

$$\text{Precision@K} = \frac{|\{\text{Rekomendasi relevan di top K}\} \cap \{\text{Item di Hold-out}\}|}{K}$$
<br>

$$\text{Recall@K} = \frac{|\{\text{Rekomendasi relevan di top K}\} \cap \{\text{Item di Hold-out}\}|}{|\{\text{Total item relevan dalam Hold-out set}\}|}$$
<br>

Untuk evaluasi yang lebih komprehensif, prosedur ini idealnya diulang untuk banyak pengguna uji dan hasilnya dirata-ratakan.

**Kenapa Menggunakan Precision@K dan Recall@K?**  
Metrik ini secara langsung menilai kualitas dari daftar rekomendasi top-N yang dihasilkan. Precision@K penting untuk memastikan pengguna tidak disajikan banyak rekomendasi yang tidak relevan, sementara Recall@K penting untuk memastikan pengguna tidak melewatkan banyak item yang mungkin mereka sukai.

**Hasil Metrik (untuk Pengguna Sampel dan Input Tertentu):**  
Berdasarkan simulasi evaluasi untuk `user_id = 123` dengan input "Toy Story 2 (1999)" dan K=10:
-   Recommended Movie IDs (top-10): `[1, 260, 296, 318, 2571, 3114, 4306, 4886, 6377, 8961]`
-   Relevant Movie IDs (liked by user, `hold_out_ids`): `[223, 327, ..., 3114, 3265, 3267, 3418]` (total 40 film)
-   Film yang relevan dan direkomendasikan di top-10: `[2571, 3114]` (2 film)
-   **Precision@10**: 2 / 10 = **0.2000**
-   **Recall@10**: 2 / 40 = **0.0500**

**Analisis Kualitatif Tambahan (Menggunakan `Genre Similarity to Input`):**
Kolom `Genre Similarity to Input` (`sim_score`) dalam output rekomendasi Content-Based Filtering dihitung menggunakan *cosine similarity* antara vektor TF-IDF genre film input dan film kandidat.
-   **Cara Kerja `sim_score`**: Skor ini berkisar antara 0 dan 1. Nilai mendekati 1 menunjukkan kemiripan genre yang sangat tinggi, sementara nilai mendekati 0 menunjukkan perbedaan genre yang signifikan.
-   **Interpretasi dalam Output**:
    -   Untuk input "Toy Story 2 (1999)":
        -   Film seperti "Toy Story (1995)" dan "Monsters, Inc. (2001)" memiliki `sim_score` = 1.0000, menunjukkan kemiripan genre yang sempurna dengan input. `final_score` mereka juga tinggi (0.922 dan 0.686), yang mengindikasikan bahwa selain kemiripan genre, preferensi pengguna lain juga mendukung rekomendasi ini.
        -   Film "Shawshank Redemption, The (1994)" memiliki `sim_score` = 0.0000 (genre sangat berbeda), namun `final_score` nya masih cukup tinggi (0.525). Ini menunjukkan bahwa rekomendasi ini lebih didorong oleh skor preferensi pengguna yang kuat (`score` awal sebesar 0.741) daripada kemiripan genre.
        -   Film seperti "Star Wars: Episode IV - A New Hope (1977)" menunjukkan keseimbangan, dengan `sim_score` rendah (0.0879) namun `score` preferensi pengguna yang moderat (0.638), menghasilkan `final_score` 0.478.

**Insight (Kualitatif dan Kuantitatif):**
-   Secara kuantitatif untuk pengguna sampel, Precision@10 sebesar 0.2000 berarti 2 dari 10 film yang direkomendasikan memang disukai oleh pengguna tersebut. Recall@10 sebesar 0.0500 berarti dari 40 film yang disukai pengguna, sistem berhasil merekomendasikan 2 di antaranya dalam 10 besar. Nilai ini memberikan baseline performa, yang dapat ditingkatkan dengan tuning bobot atau heuristik lebih lanjut.
-   Secara kualitatif, analisis `sim_score` menunjukkan bahwa sistem mampu mengidentifikasi film dengan genre yang sangat mirip. Kombinasinya dengan skor preferensi dalam `final_score` memungkinkan sistem untuk juga merekomendasikan film dengan genre berbeda namun populer di antara pengguna dengan selera serupa, yang berpotensi meningkatkan *serendipity*.
-   Kehadiran `sim_score` memberikan transparansi terhadap aspek kemiripan konten dalam rekomendasi akhir.

## Conclusion

-   Model Collaborative Filtering berbasis Deep Learning mencapai RMSE **0.1857** (latih) dan **0.1874** (validasi), menunjukkan performa prediksi rating yang baik.
-   Model Content-Based Filtering yang diperbarui, yang menggabungkan kemiripan konten dengan preferensi pengguna dan kemiripan genre murni, menghasilkan daftar rekomendasi. Evaluasi pada pengguna sampel menunjukkan Precision@10 sebesar 0.2000 dan Recall@10 sebesar 0.0500. Analisis kualitatif terhadap `Genre Similarity to Input` dan `final_score` menunjukkan kemampuan sistem untuk menyeimbangkan relevansi konten dan popularitas.
-   Kedua pendekatan menawarkan solusi yang valid untuk masalah rekomendasi film.
---
