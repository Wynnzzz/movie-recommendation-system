# Laporan Proyek Machine Learning - Berwyn Izzut Taghyir

## Overview Proyek
Seiring dengan kemajuan teknologi dan meningkatnya produksi konten digital, jumlah film yang dirilis setiap tahunnya juga meningkat pesat. Hal ini memberikan banyak pilihan kepada penikmat film, tetapi di sisi lain juga menimbulkan tantangan dalam menemukan film yang benar-benar sesuai dengan selera dan preferensi pribadi masing-masing individu. Sistem rekomendasi hadir sebagai solusi untuk membantu pengguna menyaring informasi dan menemukan konten yang relevan berdasarkan preferensi mereka.

Menurut studi yang dilakukan oleh Harper dan Konstan (2016) terkait dataset MovieLens, pemahaman mendalam mengenai sejarah dan konteks data rating pengguna sangat krusial dalam membangun sistem rekomendasi yang efektif. Mereka menekankan bahwa dataset seperti MovieLens, yang kaya akan interaksi pengguna, memungkinkan penerapan berbagai teknik, termasuk *collaborative filtering*, untuk menghasilkan rekomendasi yang dipersonalisasi. Proyek ini mengadopsi pendekatan tersebut dengan menerapkan *collaborative filtering* berbasis *deep learning* dan *content-based filtering* yang menggabungkan kemiripan konten dengan preferensi pengguna.

**Pentingnya Proyek**<br>
Membangun sistem rekomendasi film tidak hanya bertujuan untuk meningkatkan pengalaman menonton pengguna, tetapi juga berdampak signifikan pada peningkatan *engagement* pengguna terhadap platform penyedia layanan film. Bagi industri perfilman, sistem ini dapat membantu mempromosikan film-film yang mungkin kurang populer namun relevan bagi segmen pengguna tertentu, sehingga meningkatkan diversitas konsumsi konten.

Proyek ini penting untuk diselesaikan karena:
-   Membantu pengguna mengatasi masalah *information overload* dalam memilih film.
-   Meningkatkan efisiensi pengguna dalam menemukan film yang sesuai dengan selera.
-   Memberikan pengalaman menonton yang lebih personal dan memuaskan.

Proyek ini dirancang dengan membangun dua jenis sistem rekomendasi berbasis machine learning, yaitu *collaborative filtering* menggunakan model *deep learning* dan *content-based filtering* yang diperkaya dengan heuristik berbasis preferensi pengguna, dengan memanfaatkan dataset MovieLens yang diakses melalui Kaggle.

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
3.  Mengembangkan sistem rekomendasi *content-based* yang tidak hanya mengandalkan kemiripan konten (genre dan judul) tetapi juga memperhitungkan popularitas film di antara pengguna dengan selera serupa untuk memberikan rekomendasi yang lebih relevan.

### **Solution Statements:**
Untuk mencapai tujuan tersebut, proyek ini akan menggunakan dua pendekatan solusi utama:

1.  **Collaborative Filtering (Deep Learning)**
    Pendekatan ini menggunakan model jaringan saraf untuk mempelajari representasi laten (embedding) dari pengguna dan film berdasarkan histori rating. Model *RecommenderNet* yang dibangun akan memprediksi rating yang mungkin diberikan pengguna terhadap film yang belum ditonton, dan merekomendasikan film dengan prediksi rating tertinggi.
2.  **Content-Based Filtering (dengan Heuristik Preferensi Pengguna)**
    Pendekatan ini merekomendasikan film berdasarkan kemiripan konten (judul dan genre) menggunakan TF-IDF dan *cosine similarity*. Lebih lanjut, sistem ini diperkaya dengan heuristik yang menghitung skor berdasarkan seberapa sering film-film dengan konten serupa juga disukai oleh pengguna lain (baik pengguna dengan selera mirip maupun populasi pengguna secara umum), sehingga rekomendasi tidak hanya berdasarkan kemiripan teks tetapi juga popularitas kontekstual.

## Data Understanding
### Dataset
Proyek ini menggunakan dataset MovieLens yang tersedia di Kaggle ([Movie Recommendation System Dataset](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system/data)). Dataset ini populer untuk penelitian sistem rekomendasi karena berisi informasi rating film oleh pengguna serta metadata film.

Dataset ini bertujuan untuk memungkinkan pembangunan sistem yang memahami preferensi pengguna berdasarkan histori rating mereka terhadap film. Target dari dataset ini adalah untuk menghasilkan rekomendasi film yang relevan. Dataset ini terdiri dari 2 file utama: `movies.csv` dan `ratings.csv`.

**Deskripsi Variabel**
**Dataset `movies.csv`** (Cell 3, 6 notebook)<br>
Total data: 62.423 data (setelah pembersihan `(no genres listed)` menjadi 57.361 data)<br>
Berikut adalah daftar fitur (variabel) yang terdapat dalam dataset:
-   `movieId`: Nomor unik identifikasi film (Integer)
-   `title`: Judul Film, seringkali menyertakan tahun rilis (String)
-   `genres`: Genre film, dipisahkan oleh `|` (String)
-   `clean_title`: Judul film yang sudah dibersihkan (String, dibuat saat preprocessing)
-   `genres_list`: Genre film yang sudah dibersihkan (String, spasi sebagai pemisah, dibuat saat preprocessing)

**Dataset `ratings.csv`** (Cell 4, 7 notebook)<br>
Total data: 25.000.095 data<br>
Berikut adalah daftar fitur (variabel) yang terdapat dalam dataset:
-   `userId`: ID unik pengguna (Integer)
-   `movieId`: Nomor unik identifikasi film (Integer)
-   `rating`: Rating film dari pengguna (Float, skala 0.5-5)
-   `timestamp`: Waktu pemberian rating (Integer, dihapus saat preprocessing)

### Tahapan Awal Eksplorasi Data<br>
Beberapa tahapan eksplorasi dan pemahaman data yang dilakukan:
#### 1. Load Dataset<br>
Dataset `movies.csv` dan `ratings.csv` dimuat ke dataframe pandas (Cell 2 notebook).
#### 2. Melihat Struktur Dataset<br>
Menggunakan `df_movies.info()` (Cell 6) dan `df_ratings.info()` (Cell 7).<br>
**Insight:**
-   Dataset `movies` memiliki 3 kolom (`movieId`, `title`, `genres`) dengan tipe data `int64` dan `object`. Memori yang digunakan sekitar 1.4+ MB.
-   Dataset `ratings` memiliki 4 kolom (`userId`, `movieId`, `rating`, `timestamp`) dengan tipe data `int64` dan `float64`. Memori yang digunakan sekitar 762.9 MB.
#### 3. Cek Missing Values dan Duplikat<br>
Dilakukan dengan `isna().sum()` dan `duplicated().sum()` (Cell 8, 9, 11, 12 notebook).<br>
**Insight:**
-   Tidak ada nilai null pada dataset `movies` dan `ratings` pada kolom-kolom utama.
-   Tidak ada data duplikat pada kedua dataset.
-   Terdapat 5062 film dengan `(no genres listed)` pada kolom `genres` (Cell 7 EDA notebook).
#### 4. Univariate Exploratory Data Analysis
1.  Jumlah Film, Judul, Genre Unik (Cell 10 notebook)<br>
    **Insight:**
    -   Jumlah Film Unik: 62.423
    -   Jumlah Judul Film Unik: 62.325 (menunjukkan ada beberapa film dengan movieId berbeda namun judul sama, atau sebaliknya)
    -   Jumlah Kombinasi Genre Unik: 1.639
2.  Jumlah UserID Unik, Film yang Diulas, Total Rating (Cell 13 notebook)<br>
    **Insight:**
    -   Jumlah UserID Unik: 162.541
    -   Jumlah Film yang Diulas: 59.047
    -   Jumlah Data Rating: 25.000.095
3.  Top Genres (Cell 20 notebook)<br>
    Menampilkan distribusi jumlah film per genre.<br>
    ![Top Genres](img/movie_top_genres.png) <br>
    **Insight:**
    -   Genre Drama, Comedy, dan Thriller merupakan genre yang paling banyak muncul dalam dataset. Action dan Romance juga cukup dominan.
4.  Top 10 Film dengan Rata-Rata Rating Tertinggi (Cell 25 notebook)<br>
    Setelah menggabungkan `movies` dan `ratings`, dihitung rata-rata rating per film.<br>
    **Insight:**
    -   Banyak film dengan rata-rata rating 5.0, namun ini bisa jadi karena jumlah pemberi ratingnya sedikit, sehingga belum tentu mencerminkan popularitas atau kualitas secara luas. Contohnya "Humans vs Zombies (2011)".

## Data Preparation
Tahapan ini mencakup langkah-langkah untuk mempersiapkan data sebelum digunakan dalam proses modeling agar data dalam kondisi bersih, konsisten, dan sesuai untuk masing-masing algoritma.
1.  **Pembersihan Kolom `genres` pada `df_movies`** (Cell 5 EDA, Cell 8 EDA, Cell 30 "Content Based Filtering" notebook):
    -   Mengganti karakter `|` dengan spasi untuk memudahkan pemrosesan teks.
    -   Menghapus baris film dengan `genres` berisi `(no genres listed)` karena tidak memberikan informasi konten.
    -   Membuat kolom baru `genres_list` yang identik dengan kolom `genres` yang sudah dibersihkan, untuk digunakan pada TF-IDF.
    *Alasan:* Memastikan format genre konsisten dan menghilangkan data yang tidak informatif untuk content-based filtering.

2.  **Pembersihan Kolom `title` pada `df_movies`** (Cell 18, 19 EDA notebook):
    -   Membuat fungsi `clean_title` untuk mengubah judul menjadi huruf kecil, menghapus tahun dalam tanda kurung (misal, `(1995)`), menghapus karakter spesial, dan menghilangkan spasi berlebih.
    -   Menerapkan fungsi ini untuk membuat kolom baru `clean_title`.
    *Alasan:* Normalisasi judul film untuk meningkatkan akurasi pencarian dan perhitungan kemiripan berbasis judul pada content-based filtering.

3.  **Menghapus Kolom `timestamp` dari `df_ratings`** (Cell 11 "df_ratings" EDA notebook):
    -   Kolom `timestamp` tidak digunakan dalam model rekomendasi yang dibangun.
    *Alasan:* Mengurangi dimensi data yang tidak relevan.

4.  **Menggabungkan Dataset** (Cell 12 "df_ratings" EDA notebook):
    -   Menggabungkan `df_ratings` dan `df_movies` (yang sudah diproses) berdasarkan `movieId`.
    *Alasan:* Untuk mendapatkan informasi film (judul, genre) bersama dengan data rating pengguna dalam satu dataframe.

5.  **Persiapan Data untuk Collaborative Filtering (Deep Learning)** (Cell 26-30 "Data Preprocessing" notebook):
    -   **Encoding User dan Movie ID**: Mengubah `userId` dan `movieId` menjadi integer berurutan (encoded ID) untuk digunakan sebagai input pada lapisan Embedding model deep learning. Mapping asli disimpan untuk menerjemahkan kembali.
        *Alasan:* Model deep learning memerlukan input numerik kategorikal yang dimulai dari 0.
    -   **Normalisasi Rating**: Mengubah skala rating menjadi rentang [0, 1] dengan mengurangkan `min_rating` dan membagi dengan (`max_rating` - `min_rating`).
        *Alasan:* Membantu stabilisasi proses training model deep learning, terutama dengan fungsi aktivasi sigmoid di lapisan output.
    -   **Pembagian Data**: Membagi data rating (fitur: encoded user & movie ID, target: normalized rating) menjadi data latih (80%) dan data validasi (20%) setelah diacak.
        *Alasan:* Untuk melatih model dan mengevaluasi performanya pada data yang tidak terlihat saat pelatihan.

6.  **Persiapan Data untuk Content-Based Filtering** (Cell 31 "Content Based Filtering" notebook):
    -   **TF-IDF Vectorization**: Menerapkan `TfidfVectorizer` pada kolom `clean_title` dan `genres_list` dari `df_movies`. `ngram_range=(1,2)` digunakan untuk menangkap kata tunggal dan bigram.
        *Alasan:* Mengubah data teks (judul dan genre) menjadi representasi vektor numerik yang dapat digunakan untuk menghitung *cosine similarity* antar film.

## Modeling and Results

### **1. Collaborative Filtering (Deep Learning)**
Pendekatan ini menggunakan model jaringan saraf untuk mempelajari pola dari interaksi pengguna-film (rating) dan menghasilkan rekomendasi yang dipersonalisasi. Model yang digunakan adalah `RecommenderNet`.

<br>Model ini terdiri dari lapisan Embedding untuk pengguna dan film, serta lapisan bias untuk keduanya. Hasil dot product dari embedding pengguna dan film, ditambah dengan bias, kemudian dilewatkan melalui fungsi aktivasi sigmoid untuk menghasilkan prediksi rating (yang sudah dinormalisasi).

**Kelebihan:**
-   Mampu menangkap pola kompleks dalam data interaksi.
-   Dapat menghasilkan rekomendasi yang lebih personal dan akurat dibandingkan metode CF tradisional pada dataset besar.
-   Tidak secara eksplisit memerlukan fitur konten.

**Kekurangan:**
-   Masalah *cold-start* untuk pengguna dan item baru.
-   Membutuhkan jumlah data interaksi yang cukup besar untuk performa optimal.
-   Proses training bisa memakan waktu dan sumber daya komputasi.

**Untuk Training model** (Cell 31, 33 "Recommendation" notebook):
-   `user_embedding` dan `movie_embedding`: Mengubah ID pengguna dan ID film menjadi vektor embedding berukuran 50.
-   `user_bias` dan `movie_bias`: Menambahkan bias untuk setiap pengguna dan film.
-   `dot_user_movie`: Menghitung kesamaan (dot product) antara vektor pengguna dan film.
-   Output dilewatkan ke fungsi aktivasi `sigmoid` karena target rating dinormalisasi ke [0, 1].
-   Model di-compile menggunakan loss `binary_crossentropy` (cocok untuk output sigmoid), optimizer `Adam` (learning rate 1e-4), dan metrik `RootMeanSquaredError`.
-   Dilatih selama 20 epoch dengan callbacks (ReduceLROnPlateau, EarlyStopping, ModelCheckpoint).

**Hasil Training** (Plot dari Cell 34 notebook):<br>
![DL Model RMSE](img/dl_model_rmse_plot.png) <br>
**Insight:** Model mencapai nilai `root_mean_squared_error` (RMSE ternormalisasi) sekitar 0.1857 untuk data latih dan 0.1874 untuk data validasi pada epoch terakhir (epoch ke-20). Kurva loss dan RMSE menunjukkan konvergensi yang baik tanpa overfitting yang signifikan.

**Hasil Rekomendasi** (Contoh dari Cell 36 notebook):<br>
Misal untuk User ID 8868, setelah menampilkan film rating tertinggi yang pernah ditontonnya, 10 rekomendasi teratas adalah:
![CF Movie Recs](img/cf_movie_recs.png) <br>
(Contoh Output yang diharapkan, diisi dari notebook)
1.  Harakiri (Seppuku) (1962) - Genre: Drama
2.  The Blue Planet (2001) - Genre: Documentary
3.  Planet Earth (2006) - Genre: Documentary
4.  Life (2009) - Genre: Documentary
5.  Over the Garden Wall (2013) - Genre: Adventure Animation Drama
... (dst)

### **2. Content-Based Filtering (dengan Heuristik Preferensi Pengguna)**
Pendekatan ini merekomendasikan film berdasarkan kemiripan konten (judul dan genre) yang diperkaya dengan informasi preferensi pengguna lain.

<br>Konten yang dianalisis adalah `clean_title` dan `genres_list`. TF-IDF dan *cosine similarity* digunakan untuk menemukan film dengan judul atau genre yang mirip. Heuristik skor kemudian diterapkan yang menggabungkan:
1.  Seberapa sering film kandidat disukai oleh pengguna yang juga menyukai film input awal (rating >= 4).
2.  Seberapa sering film kandidat disukai oleh semua pengguna.
3.  Kemiripan genre (dengan bobot lebih tinggi jika genre mirip).
Skor akhir dihitung sebagai rasio preferensi pengguna serupa terhadap preferensi semua pengguna, dengan penyesuaian bobot untuk kemiripan genre.

**Kelebihan:**
-   Tidak sepenuhnya bergantung pada rating pengguna input, bisa bekerja untuk film dengan sedikit rating jika kontennya jelas.
-   Transparan, rekomendasi dapat dijelaskan berdasarkan kemiripan konten dan popularitas kontekstual.
-   Upaya mengatasi *over-specialization* dengan memasukkan preferensi pengguna lain.

**Kekurangan:**
-   Kualitas sangat bergantung pada kualitas metadata (judul, genre) dan efektivitas heuristik skor.
-   Mungkin masih bias terhadap film dengan genre populer jika tidak diimbangi dengan baik.
-   Perhitungan skor bisa menjadi kompleks.

Langkahnya dimulai dengan pengguna memasukkan judul film. Sistem mencari kandidat film dengan judul mirip (Cell 28). Setelah pengguna memilih salah satu kandidat (misal, film dengan `movieId` tertentu), fungsi `scores_calculator` (Cell 26) menghitung skor untuk film lain. `recommendation_results` (Cell 27) kemudian menampilkan top-N rekomendasi.

Contoh: Jika pengguna mencari "Toy Story" dan memilih "Toy Story 4 (2019)" (Cell 28, 32 notebook):

**Hasil Rekomendasi:**<br>
![CBF Movie Recs](img/cbf_movie_recs.png) <br>
(Contoh Output yang diharapkan, diisi dari notebook Cell 32)
1.  Toy Story 4 (2019) - score: 712.00, genres: Adventure Animation Children Comedy
2.  Cafard (2015) - score: 712.00, genres: Animation Drama War
3.  Eastern Condors (1987) - score: 712.00, genres: Action War
... (dst)

## Evaluation
### **1. Collaborative Filtering (Deep Learning)**
**Metrik Evaluasi yang Digunakan**
<br>Root Mean Squared Error (RMSE) digunakan sebagai metrik utama untuk mengevaluasi seberapa akurat model memprediksi rating pengguna. RMSE mengukur rata-rata besarnya kesalahan prediksi. Nilai RMSE yang lebih kecil menunjukkan performa prediksi yang lebih baik. Metrik ini dihitung pada rating yang sudah dinormalisasi ke skala [0,1].

**Kenapa Menggunakan RMSE?**
<br>RMSE dipilih karena memberikan gambaran yang jelas tentang seberapa jauh hasil prediksi model dari nilai sebenarnya. Meskipun loss function yang digunakan adalah `binary_crossentropy` (karena output sigmoid), RMSE tetap relevan untuk menginterpretasikan error dalam konteks prediksi rating.

<br>**Rumus (konseptual untuk rating asli, implementasi di TF menggunakan output ternormalisasi):**
$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

<br>**Keterangan:**
- $y_i$ = Nilai rating aktual (ternormalisasi).
- $\hat{y}_i$ = Nilai rating prediksi (output model).
- $n$ = Jumlah data.

**Visualisasi**
<br>Learning curve (RMSE terhadap epoch) digunakan untuk memantau proses training dan melihat apakah model mengalami overfitting atau underfitting.
<br>

Hasil RMSE Model (dari output training Cell 33 dan plot Cell 34 notebook): <br>
![DL Model RMSE Plot](img/dl_model_rmse_plot.png) <br>
**Insight:**
-   Pada akhir epoch ke-20, model memperoleh nilai `root_mean_squared_error`: **0.1857** untuk data latih dan **0.1874** untuk data validasi.
-   Perbedaan antara RMSE latih dan validasi tidak terlalu besar, menunjukkan model tidak mengalami overfitting yang parah dan memiliki generalisasi yang cukup baik pada data validasi. Penurunan RMSE yang stabil pada kedua set data mengindikasikan proses pembelajaran yang efektif.

### **2. Content-Based Filtering (dengan Heuristik Preferensi Pengguna)**

**Metrik Evaluasi yang Digunakan**
<br>Untuk pendekatan Content-Based Filtering yang diterapkan dalam notebook ini, evaluasi kuantitatif standar seperti Precision@K atau Recall@K tidak diimplementasikan secara langsung. Sistem ini lebih mengandalkan heuristik untuk menghasilkan skor internal yang digunakan untuk perangkingan. Oleh karena itu, evaluasi lebih bersifat **kualitatif** berdasarkan relevansi output rekomendasi terhadap input film yang diberikan.

**Kenapa Evaluasi Kualitatif?**
<br>Pendekatan ini menggabungkan beberapa sinyal (kemiripan judul, kemiripan genre, preferensi pengguna serupa, preferensi semua pengguna) ke dalam satu skor heuristik. Untuk mengukur Precision@K secara formal, diperlukan dataset uji dengan ground truth film relevan yang disukai pengguna, yang tidak disiapkan dalam alur notebook ini untuk metode CBF.

**Contoh Hasil (Kualitatif)**
Berdasarkan input "Toy Story" (dan pilihan "Toy Story 4 (2019)"), sistem merekomendasikan (Cell 32):
1.  Toy Story 4 (2019)
2.  Cafard (2015) (Animation Drama War)
3.  Eastern Condors (1987) (Action War)
...

**Insight (Kualitatif):**
-   Film input ("Toy Story 4") muncul sebagai rekomendasi teratas, yang diharapkan.
-   Rekomendasi lainnya ("Cafard", "Eastern Condors") memiliki skor yang sama tinggi. Ini mungkin menunjukkan bahwa heuristik skor yang digunakan cenderung memberikan skor serupa untuk banyak film yang memenuhi kriteria tertentu atau perlu penyesuaian lebih lanjut untuk diferensiasi yang lebih baik. Relevansi genre dari rekomendasi ini terhadap "Toy Story" (Animation, Adventure, Children, Comedy) perlu dianalisis lebih lanjut secara manual untuk menilai kualitasnya. Beberapa rekomendasi mungkin memiliki genre yang cukup berbeda, menandakan pengaruh kuat dari preferensi pengguna lain dalam heuristik skor.

## Conclusion

-   Model Collaborative Filtering berbasis Deep Learning berhasil dilatih dan mampu memberikan rekomendasi film yang dipersonalisasi kepada pengguna. Model ini mencapai RMSE sebesar **0.1857** pada data latih dan **0.1874** pada data validasi (untuk rating ternormalisasi), menunjukkan performa prediksi yang baik dan generalisasi yang cukup.
-   Model Content-Based Filtering yang dikembangkan menggunakan TF-IDF untuk kemiripan judul dan genre, serta diperkaya dengan heuristik skor berbasis preferensi pengguna. Evaluasi kualitatif menunjukkan bahwa model ini dapat menghasilkan daftar rekomendasi, namun skor internal yang dihasilkan mungkin memerlukan kalibrasi lebih lanjut untuk diferensiasi yang lebih baik antar film yang direkomendasikan.
-   Kedua pendekatan menawarkan cara yang berbeda untuk mengatasi masalah rekomendasi film, dengan Collaborative Filtering fokus pada pola interaksi pengguna dan Content-Based Filtering pada karakteristik film yang diperkaya preferensi komunitas.
