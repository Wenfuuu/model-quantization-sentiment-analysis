# PTQ VS QAT
Source Paper : [Optimizing Large Language Models through
Quantization: A Comparative Analysis of PTQ and
QAT Techniques](https://arxiv.org/pdf/2411.06084)

## PTQ 
* Pretrain model, lalu baru di quantize
* Cepat utk eksperiment karna ga perlu melatih ulang modelnya

## QAT
* Pada saat di finetune itu sekalian dibuat untuk beradaptasi dengan presisi rendah (jadinya sekalian difinetune)
* Kalau dari papernya biasanya bakalan menghasilkan akurasi yang jauh lebih baik karna modelnya sudah beradaptasi 

## Need to know before read
* Scaling Factor -> Nilai aslinya = Nilai INT x Scaling Factor jadinya return nilainya mendekati asli. contohnya ada benda 1.25cm, 5.75cm. penggaris kita cuman ada 1cm sampai 10cm yang dimana cuman ad 10 potongan jadinya cuman 1,2,3,4,5,6,7,8,9,10 ga bisa 1,25cm. scaling factor ngekecilin penggarisnya jadinya bisa lebih detail

## PTQ -> INT8
* pakai calibration dataset yang kecil untuk tentuin nilai min/max dari bobot model lalu itung scaling factor simple supaya muat di 8-bit
* hasilnya stabil, pengurangan ukuran model lumayan, masih ada penurunan akurasi tapi cuman dikit aja
* ini dalam proses membuatnya cpt banget dan ga perlu retrain atau finetune
* throughputnya 2.4x lebih cepat
* hemat dayanya sampai 40%

## QAT -> INT8
* Modelnya berpura pura diquantize saat latian (fake quantization). Modelnya belajar untuk toleransi sedikit noise 
* Akurasinya sedikit lebih tinggi daripada metode PTQ
* Ini dari papernya udah tunjukin kalau ini merupakan metode yang udah cukup bagus dibandingkan PTQ

## PTQ -> INT4 
* ini based on papernya dibilang kea maksain banget, jadinya ada degradasi performance yang parah banget sehingga akurasinya anjlok banget. jadinya ga recommended

## QAT -> INT4 + Scaling Factor 
* dibilangnya harus pake scaling factor
* cara kerjanya ada 2 Mixed-Precision & Adaptasi
* Mixed-Precision -> tidak semua layernya dipukul rata jadi INT4. mereka cek layer mana yang sebenarnya sensitif (penting banget) yang harus di keep presisi tinggi dan mana yang aman untuk dikompress ke INT4
* Adaptasi -> Selama QAT, model belajar menyesuaikan bobotnya secara aktif untuk meminimalkan loss akibat resolusi yang rendah
* ini jadinya paling goat, ukurannya turun smpe 68%, througputnya 3x lipat, hemat daya 60%, akurasinya tetap terjaga dalam batas 6% dari model aslinya (masih berkurang tapi bisa ditoleransi)