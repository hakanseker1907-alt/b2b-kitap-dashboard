B2B Kitap Dağıtım Paneli

ISL 467E – Management Engineering Applications with Python · Homework I

Bir B2B kitap distribütörünün 2020 yılı satış verisiyle hazırlanmış Streamlit dashboard'u. Satış, pazarlama ve tedarik zinciri ekiplerinin karar vermesine yardımcı olmak amacıyla hazırlandı.

Canlı dashboard: https://b2b-kitap-paneli.streamlit.app/

Grup üyeleri
Hakan ŞEKER– 070220326
Yiğitkan Doğan – 070220269
Ahmet Hazineli - 
Nursel İlgün
Jalil Nazarli

Dosya	Açıklama
streamlit_app.py	Dashboard'un kaynak kodu
veri_hazirlama.ipynb	Veri temizleme adımları (Jupyter)
b2b_clean.parquet	Temizlenmiş veri, dashboard bu dosyayı okur
veri_kalitesi.csv	Temizleme raporu (silinen ve doldurulan satır sayıları)
requirements.txt	Gerekli Python kütüphaneleri

Ham veri (B2B_Transaction_Data_.xlsx, 664.154 satır) Ninova'da paylaşılan dosyadır.

Veri hazırlama
Birebir aynı olan 6.213 satır silindi (aynı fatura kalemi iki kez sayılmasın diye).
Adet bilgisi boş olan 6 satır silindi.
Şehri (6.002) ve kategorisi (63) boş olan satırlar silinmedi, "Bilinmiyor" olarak işaretlendi. Toplam ciroda yer alıyorlar, şehir kırılımlarında gösterilmiyorlar.
Ciro Quantity × NetPrice olarak hesaplandı (iskontolu satış fiyatı).
Tarih sütunu tarih formatına çevrildi; ay ve haftanın günü sütunları eklendi.
Dashboard içeriği

Sol menüdeki tarih, şehir ve kategori filtreleri tüm sekmelere uygulanır.

Genel Bakış: Temel KPI'lar (son ayın bir önceki aya göre değişimiyle), aylık ciro, kategori ve şehir payları.
Satış: En büyük müşteriler, haftanın günlerine göre ciro, kategori × ay ısı haritası, şehir tablosu ve 30 günlük ciro tahmini.
Pazarlama: RFM analiziyle 4 müşteri segmenti (Şampiyonlar, Sadık Müşteriler, Risk Altında, Kaybedilmiş) ve aranacak müşteri listesi.
Tedarik Zinciri: Ayarlanabilir sınırlarla ABC analizi, en çok satan ürünler, kategori bazında aylık talep ve son 90 gündür satılmayan ürünler.
Kullanılan yöntemler
RFM: Müşteriler son alım tarihi, fatura sayısı ve toplam ciroya göre 1–4 arası puanlandı; R puanı ile F-M ortalamasına göre segmentlere ayrıldı.
ABC analizi: Ürünler ciroya göre sıralanıp kümülatif paya göre A (%80), B (%15) ve C (%5) sınıflarına ayrıldı.
Çoklu doğrusal regresyon: Günlük ciro; trend, haftanın günü (kukla değişkenler), pandemi dönemi (16 Mart – 31 Mayıs 2020) ve tatil günleriyle modellendi. Model R² ≈ 0,57.
Yerelde çalıştırma
pip install -r requirements.txt
streamlit run streamlit_app.py
