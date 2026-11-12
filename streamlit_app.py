import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ---------- Sayfa ayarları ----------
st.set_page_config(page_title="B2B Kitap Dağıtım Paneli", page_icon="📚", layout="wide")


# ---------- Veri ----------
@st.cache_data
def veri_yukle():
    return pd.read_parquet("b2b_clean.parquet")


df = veri_yukle()


# ---------- Renkler (lacivert + turkuaz) ----------
LACIVERT = "#1B2A49"
TURKUAZ = "#14A3A3"
GRI = "#9AA5B1"
TURUNCU = "#E8A33D"
SEGMENT_RENK = {
    "Şampiyonlar": LACIVERT,
    "Sadık Müşteriler": TURKUAZ,
    "Risk Altında": TURUNCU,
    "Kaybedilmiş": GRI,
}
SINIF_RENK = {"A": LACIVERT, "B": TURKUAZ, "C": GRI}
AY_ADLARI = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
             "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
GUN_ADLARI = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]

# Tahmin modeli için özel dönemler
PANDEMI_BAS, PANDEMI_BIT = pd.Timestamp("2020-03-16"), pd.Timestamp("2020-05-31")
# Resmi tatiller + Temmuz sonu kapanışı (veride bu hafta içi günlerde satış yok)
TATILLER = pd.to_datetime([
    "2020-01-01", "2020-04-23", "2020-05-01", "2020-05-19", "2020-05-24", "2020-05-25", "2020-05-26",
    "2020-07-15", "2020-10-29", "2021-01-01",
]).append(pd.date_range("2020-07-20", "2020-08-03"))


# ---------- Yardımcı fonksiyonlar ----------
def grafik_duzenle(fig, yukseklik=380):
    """Tüm grafiklere aynı sade görünümü verir"""
    fig.update_layout(
        template="plotly_white",
        height=yukseklik,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(size=13),
        title_font=dict(size=16, color=LACIVERT),
        showlegend=False,
    )
    return fig


def tl(sayi):
    """29417403.65 -> '29.417.404 ₺' (Türkçe binlik ayırıcı)"""
    return f"{sayi:,.0f} ₺".replace(",", ".")


def yuzde(sayi, basamak=1):
    """12.43 -> '%12,4'"""
    return f"%{sayi:.{basamak}f}".replace(".", ",")


def adet(sayi):
    return f"{sayi:,.0f}".replace(",", ".")


def kpi_hesapla(veri):
    ciro = veri["Revenue"].sum()
    fatura = veri["InvoiceNo"].nunique()
    return {
        "ciro": ciro,
        "fatura": fatura,
        "musteri": veri["CustomerID"].nunique(),
        "ort_fatura": ciro / fatura if fatura > 0 else 0,
        "adet": veri["Quantity"].sum(),
    }


def rfm_hesapla(veri, referans_gun):
    """Her müşteri için Recency, Frequency, Monetary değerleri, 1-4 puanları ve segmenti"""
    rfm = veri.groupby("CustomerID").agg(
        SonAlim=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
    )
    rfm["Recency"] = (referans_gun - rfm["SonAlim"]).dt.days

    # Müşterileri sıralayıp 4 eşit gruba bölüyoruz (4 = en iyi)
    def puan(seri, buyuk_iyi=True):
        return np.ceil(seri.rank(pct=True, ascending=buyuk_iyi, method="first") * 4).astype(int)

    rfm["R"] = puan(rfm["Recency"], buyuk_iyi=False)  # az gün = yakın zamanda almış = iyi
    rfm["F"] = puan(rfm["Frequency"])
    rfm["M"] = puan(rfm["Monetary"])
    deger = np.round((rfm["F"] + rfm["M"]) / 2)

    yakin = rfm["R"] >= 3
    degerli = deger >= 3
    rfm["Segment"] = np.select(
        [yakin & degerli, yakin & ~degerli, ~yakin & degerli],
        ["Şampiyonlar", "Sadık Müşteriler", "Risk Altında"],
        default="Kaybedilmiş",
    )
    return rfm


def abc_hesapla(veri, a_siniri=80, b_siniri=95):
    """Ürünleri ciroya göre sıralayıp kümülatif paya göre A/B/C sınıfına ayırır"""
    urun = (
        veri.groupby("StockCode")
        .agg(Urun=("Description", "first"), Kategori=("Category", "first"),
             Ciro=("Revenue", "sum"), Adet=("Quantity", "sum"), SonSatis=("InvoiceDate", "max"))
        .sort_values("Ciro", ascending=False)
    )
    urun["KumulatifPay"] = urun["Ciro"].cumsum() / urun["Ciro"].sum() * 100
    # Bir ürün, kendisinden ÖNCEKİ kümülatif pay sınırın altındaysa o sınıfa girer
    onceki_pay = urun["KumulatifPay"].shift(fill_value=0)
    urun["Sinif"] = np.select([onceki_pay < a_siniri, onceki_pay < b_siniri], ["A", "B"], default="C")
    return urun


def ozellikler(tarihler, baslangic_gunu):
    """Regresyonun açıklayıcı değişkenleri: trend, haftanın günü, pandemi ve tatil"""
    X = pd.DataFrame(index=tarihler)
    X["Trend"] = (tarihler - baslangic_gunu).days
    for i, gun in enumerate(GUN_ADLARI[1:], start=1):  # Pazartesi referans gün
        X[gun] = (tarihler.dayofweek == i).astype(int)
    X["Pandemi"] = ((tarihler >= PANDEMI_BAS) & (tarihler <= PANDEMI_BIT)).astype(int)
    X["Tatil"] = tarihler.isin(TATILLER).astype(int)
    return X


def tahmin_yap(veri, son_tarih, ufuk=30):
    """Günlük ciroyu çoklu doğrusal regresyonla modelleyip sonraki 'ufuk' günü tahmin eder"""
    gunluk = veri.groupby("InvoiceDate")["Revenue"].sum()
    takvim = pd.date_range(gunluk.index.min(), son_tarih)
    y = gunluk.reindex(takvim, fill_value=0)  # satış olmayan günler 0
    X = ozellikler(takvim, takvim[0])

    model = LinearRegression().fit(X, y)
    uyum = model.predict(X)

    gelecek = pd.date_range(son_tarih + pd.Timedelta(days=1), periods=ufuk)
    tahmin = np.clip(model.predict(ozellikler(gelecek, takvim[0])), 0, None)
    hata_std = np.std(y - uyum)

    gecmis = pd.DataFrame({"Tarih": takvim, "Gerçek": y.values, "Model": uyum})
    ileri = pd.DataFrame({
        "Tarih": gelecek,
        "Tahmin": tahmin,
        "Alt": np.clip(tahmin - 1.96 * hata_std, 0, None),
        "Üst": tahmin + 1.96 * hata_std,
    })
    katsayilar = pd.Series(model.coef_, index=X.columns)
    olcum = {"r2": r2_score(y, uyum), "mae": mean_absolute_error(y, uyum)}
    return gecmis, ileri, katsayilar, olcum


def degisim(son, onceki):
    """Son ayın bir önceki aya göre yüzde değişimi"""
    if onceki == 0:
        return None
    return f"{(son - onceki) / onceki * 100:+.1f}% son ay"


# ---------- Başlık ----------
st.title("📚 B2B Kitap Dağıtım Paneli")
st.caption("2020 satış verisi · Satış, pazarlama ve tedarik zinciri ekipleri için karar destek paneli")

# ---------- Sol menü: filtreler ----------
st.sidebar.header("Filtreler")

ilk_gun = df["InvoiceDate"].min().date()
son_gun = df["InvoiceDate"].max().date()
tarih = st.sidebar.date_input(
    "Tarih aralığı", value=(ilk_gun, son_gun), min_value=ilk_gun, max_value=son_gun
)

sehirler = st.sidebar.multiselect(
    "Şehir", sorted(df["City"].unique()), placeholder="Tüm şehirler"
)
kategoriler = st.sidebar.multiselect(
    "Kategori", sorted(df["Category"].unique()), placeholder="Tüm kategoriler"
)
st.sidebar.caption("Boş bırakılan filtre tüm değerleri kapsar.")

# Takvimde sadece başlangıç seçildiyse bitiş gelene kadar bekle
if len(tarih) != 2:
    st.warning("Lütfen takvimden bitiş tarihini de seçin.")
    st.stop()
baslangic, bitis = pd.Timestamp(tarih[0]), pd.Timestamp(tarih[1])

# 1) Şehir ve kategori filtresi (KPI karşılaştırması için tarih uygulanmadan)
df_sk = df.copy()
if sehirler:
    df_sk = df_sk[df_sk["City"].isin(sehirler)]
if kategoriler:
    df_sk = df_sk[df_sk["Category"].isin(kategoriler)]

# 2) Tarih filtresi -> tüm grafiklerin kullanacağı veri
dff = df_sk[(df_sk["InvoiceDate"] >= baslangic) & (df_sk["InvoiceDate"] <= bitis)]

if dff.empty:
    st.warning("Seçilen filtrelere uygun veri yok. Filtreleri genişletin.")
    st.stop()

# ---------- Sekmeler ----------
tab_genel, tab_satis, tab_pazarlama, tab_tedarik = st.tabs(
    ["Genel Bakış", "Satış", "Pazarlama", "Tedarik Zinciri"]
)

with tab_genel:
    # Kartlar seçilen dönemin toplamını, oklar ise seçilen dönemin
    # son ayını bir önceki ayla karşılaştırır
    son_ay = dff["InvoiceDate"].dt.to_period("M").max()
    aylar = df_sk["InvoiceDate"].dt.to_period("M")
    k_son = kpi_hesapla(df_sk[aylar == son_ay])
    k_onceki = kpi_hesapla(df_sk[aylar == son_ay - 1])
    k = kpi_hesapla(dff)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Toplam Ciro", tl(k["ciro"]), degisim(k_son["ciro"], k_onceki["ciro"]))
    c2.metric("Fatura Sayısı", adet(k["fatura"]), degisim(k_son["fatura"], k_onceki["fatura"]))
    c3.metric("Aktif Müşteri", adet(k["musteri"]), degisim(k_son["musteri"], k_onceki["musteri"]))
    c4.metric("Ort. Fatura Tutarı", tl(k["ort_fatura"]), degisim(k_son["ort_fatura"], k_onceki["ort_fatura"]))
    c5.metric("Satılan Adet", adet(k["adet"]), degisim(k_son["adet"], k_onceki["adet"]))

    # --- Aylık ciro trendi ---
    aylik = dff.groupby(dff["InvoiceDate"].dt.month)["Revenue"].sum().reset_index()
    aylik.columns = ["Ay", "Ciro"]
    aylik["Ay Adı"] = aylik["Ay"].apply(lambda a: AY_ADLARI[a - 1])

    fig = px.bar(aylik, x="Ay Adı", y="Ciro", title="Aylık Ciro")
    fig.update_traces(marker_color=TURKUAZ, hovertemplate="%{x}: %{y:,.0f} ₺<extra></extra>")
    fig.update_yaxes(title=None, tickformat=",.0f")
    fig.update_xaxes(title=None)
    st.plotly_chart(grafik_duzenle(fig))

    # --- Kategori ve şehir payları (yan yana) ---
    sol, sag = st.columns(2)

    kat = dff.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    kat_pay = (kat / kat.sum() * 100).head(10).reset_index()
    kat_pay.columns = ["Kategori", "Pay"]
    fig = px.bar(kat_pay, x="Pay", y="Kategori", orientation="h", title="Ciroda İlk 10 Kategori (%)")
    fig.update_traces(marker_color=LACIVERT, hovertemplate="%{y}: %{x:.1f}%<extra></extra>")
    fig.update_yaxes(title=None, autorange="reversed")
    fig.update_xaxes(title=None)
    sol.plotly_chart(grafik_duzenle(fig))

    # Şehri bilinmeyen satırlar şehir grafiğine alınmıyor (temizlik kararı 3)
    bilinen = dff[dff["City"] != "Bilinmiyor"]
    sehir = bilinen.groupby("City")["Revenue"].sum().sort_values(ascending=False)
    sehir_pay = (sehir / sehir.sum() * 100).head(10).reset_index()
    sehir_pay.columns = ["Şehir", "Pay"]
    fig = px.bar(sehir_pay, x="Pay", y="Şehir", orientation="h", title="Ciroda İlk 10 Şehir (%)")
    fig.update_traces(marker_color=TURKUAZ, hovertemplate="%{y}: %{x:.1f}%<extra></extra>")
    fig.update_yaxes(title=None, autorange="reversed")
    fig.update_xaxes(title=None)
    sag.plotly_chart(grafik_duzenle(fig))
    bilinmeyen_pay = 100 - bilinen["Revenue"].sum() / dff["Revenue"].sum() * 100
    sag.caption(f"Şehir bilgisi olmayan satışlar ({yuzde(bilinmeyen_pay)}) bu grafiğe dahil değil.")

    # --- Öne çıkanlar (filtreye göre otomatik değişir) ---
    with st.container(border=True):
        st.markdown("**💡 Öne çıkanlar**")
        notlar = []
        if len(aylik) > 1:
            en_iyi = aylik.loc[aylik["Ciro"].idxmax()]
            en_kotu = aylik.loc[aylik["Ciro"].idxmin()]
            notlar.append(
                f"En yüksek ciro **{en_iyi['Ay Adı']}** ayında ({tl(en_iyi['Ciro'])}), "
                f"en düşük ciro **{en_kotu['Ay Adı']}** ayında ({tl(en_kotu['Ciro'])})."
            )
        notlar.append(
            f"En çok ciro getiren kategori **{kat_pay.iloc[0]['Kategori']}** "
            f"(toplam cirodaki payı {yuzde(kat_pay.iloc[0]['Pay'])}). "
            f"İlk 10 kategorinin toplam payı {yuzde(kat_pay['Pay'].sum(), 0)}."
        )
        if not sehir_pay.empty:
            notlar.append(
                f"Satışların merkezi **{sehir_pay.iloc[0]['Şehir']}** (payı {yuzde(sehir_pay.iloc[0]['Pay'], 0)}). "
                f"İlk 3 şehrin toplam payı {yuzde(sehir_pay['Pay'].head(3).sum(), 0)}."
            )
        for n in notlar:
            st.markdown(f"- {n}")

    # --- Veri kalitesi raporu (Jupyter'de üretilen) ---
    with st.expander("Veri temizleme raporu"):
        rapor = pd.read_csv("veri_kalitesi.csv", index_col=0).iloc[:, 0]
        st.dataframe(rapor.rename("Değer"))

with tab_satis:
    # --- Hesaplamalar ---
    musteri = dff.groupby("CustomerID")["Revenue"].sum().sort_values(ascending=False)
    ilk10 = (musteri / musteri.sum() * 100).head(10).reset_index()
    ilk10.columns = ["Müşteri", "Pay"]
    ilk10["Ciro"] = musteri.head(10).values

    # Önce her günün cirosu, sonra haftanın gününe göre ortalaması
    gunluk = dff.groupby(dff["InvoiceDate"].dt.date)["Revenue"].sum().reset_index()
    gunluk.columns = ["Tarih", "Ciro"]
    gunluk["Gün"] = pd.to_datetime(gunluk["Tarih"]).dt.dayofweek
    gun_ort = gunluk.groupby("Gün")["Ciro"].mean().reset_index()
    gun_ort["Gün Adı"] = gun_ort["Gün"].apply(lambda g: GUN_ADLARI[g])
    en_guclu_gun = gun_ort.loc[gun_ort["Ciro"].idxmax()]

    sehir_tablo = (
        bilinen.groupby("City")
        .agg(Ciro=("Revenue", "sum"), Musteri=("CustomerID", "nunique"), Fatura=("InvoiceNo", "nunique"))
        .sort_values("Ciro", ascending=False)
    )
    sehir_tablo["Pay"] = sehir_tablo["Ciro"] / sehir_tablo["Ciro"].sum() * 100
    sehir_tablo["MusteriBasina"] = sehir_tablo["Ciro"] / sehir_tablo["Musteri"]

    # --- KPI kartları ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("İlk 10 Müşterinin Payı", yuzde(ilk10["Pay"].sum()))
    c2.metric("En Güçlü Gün", en_guclu_gun["Gün Adı"])
    c3.metric("Müşteri Başına Ciro", tl(musteri.mean()))
    c4.metric("Satış Yapılan Şehir", adet(len(sehir_tablo)))

    # --- En büyük müşteriler ve haftanın günleri (yan yana) ---
    sol, sag = st.columns(2)

    fig = px.bar(ilk10, x="Ciro", y="Müşteri", orientation="h", title="En Büyük 10 Müşteri",
                 custom_data=["Pay"])
    fig.update_traces(marker_color=LACIVERT,
                      hovertemplate="%{y}: %{x:,.0f} ₺ (pay %{customdata[0]:.1f}%)<extra></extra>")
    fig.update_yaxes(title=None, autorange="reversed")
    fig.update_xaxes(title=None, tickformat=",.0f")
    sol.plotly_chart(grafik_duzenle(fig))

    fig = px.bar(gun_ort, x="Gün Adı", y="Ciro", title="Haftanın Günlerine Göre Ortalama Günlük Ciro")
    fig.update_traces(marker_color=TURKUAZ, hovertemplate="%{x}: %{y:,.0f} ₺<extra></extra>")
    fig.update_yaxes(title=None, tickformat=",.0f")
    fig.update_xaxes(title=None)
    sag.plotly_chart(grafik_duzenle(fig))

    # --- Kategori x ay ısı haritası (ilk 10 kategori) ---
    ilk10_kat = kat.head(10).index
    isi = (
        dff[dff["Category"].isin(ilk10_kat)]
        .pivot_table(index="Category", columns="Month", values="Revenue", aggfunc="sum", fill_value=0)
        .reindex(ilk10_kat)
    )
    isi.columns = [AY_ADLARI[a - 1] for a in isi.columns]
    fig = px.imshow(isi, aspect="auto", title="İlk 10 Kategorinin Aylık Cirosu",
                    color_continuous_scale=["#FFFFFF", TURKUAZ, LACIVERT])
    fig.update_traces(hovertemplate="%{y} · %{x}: %{z:,.0f} ₺<extra></extra>")
    fig.update_xaxes(title=None, side="top")
    fig.update_yaxes(title=None)
    fig = grafik_duzenle(fig, yukseklik=420)
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig)

    # --- Öne çıkanlar ---
    with st.container(border=True):
        st.markdown("**💡 Öne çıkanlar**")
        notlar = [
            f"Cironun {yuzde(ilk10['Pay'].sum(), 0)} kadarı yalnızca 10 müşteriden geliyor. "
            f"En büyük müşteri **{ilk10.iloc[0]['Müşteri']}** tek başına {yuzde(ilk10.iloc[0]['Pay'])} paya sahip.",
        ]
        hafta_sonu = gun_ort[gun_ort["Gün"] >= 5]["Ciro"].mean()
        if pd.notna(hafta_sonu) and hafta_sonu > 0:
            notlar.append(
                f"**{en_guclu_gun['Gün Adı']}** günleri ortalama {tl(en_guclu_gun['Ciro'])} satış yapılıyor; "
                f"bu, hafta sonu ortalamasının yaklaşık {en_guclu_gun['Ciro'] / hafta_sonu:.0f} katı."
            )
        if isi.shape[1] > 1:
            zirve = isi.idxmax(axis=1).value_counts()
            notlar.append(
                f"İlk 10 kategoriden {zirve.iloc[0]} tanesi en yüksek cirosunu **{zirve.index[0]}** ayında yapıyor."
            )
        buyuk_sehirler = sehir_tablo[sehir_tablo["Musteri"] >= 5]
        if not buyuk_sehirler.empty:
            en_verimli = buyuk_sehirler["MusteriBasina"].idxmax()
            notlar.append(
                f"En az 5 müşterisi olan şehirler arasında müşteri başına ciroda **{en_verimli}** önde "
                f"({tl(buyuk_sehirler.loc[en_verimli, 'MusteriBasina'])})."
            )
        for n in notlar:
            st.markdown(f"- {n}")

    # --- Şehir detay tablosu ---
    st.markdown("**Şehir detay tablosu**")
    st.caption("Sütun başlığına tıklayarak sıralayabilirsiniz. Şehir bilgisi olmayan satışlar dahil değil.")
    st.dataframe(
        sehir_tablo.reset_index()[["City", "Ciro", "Pay", "Musteri", "Fatura", "MusteriBasina"]],
        hide_index=True,
        height=350,
        column_config={
            "City": "Şehir",
            "Ciro": st.column_config.NumberColumn("Ciro (₺)", format="localized"),
            "Pay": st.column_config.NumberColumn("Pay (%)", format="%.1f"),
            "Musteri": "Müşteri",
            "Fatura": "Fatura",
            "MusteriBasina": st.column_config.NumberColumn("Müşteri Başına Ciro (₺)", format="localized"),
        },
    )

    # ================= Ciro tahmini =================
    st.divider()
    st.subheader("Önümüzdeki 30 Günün Ciro Tahmini")
    st.caption("Çoklu doğrusal regresyon: günlük ciro ~ trend + haftanın günü + pandemi dönemi + tatil günleri")

    if dff["InvoiceDate"].nunique() < 60:
        st.info("Tahmin için en az 60 günlük veri gerekiyor. Tarih aralığını genişletin.")
    else:
        gecmis, ileri, katsayilar, olcum = tahmin_yap(dff, bitis)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("30 Günlük Tahmini Ciro", tl(ileri["Tahmin"].sum()))
        c2.metric("Son 30 Günün Gerçek Cirosu", tl(gecmis["Gerçek"].tail(30).sum()))
        c3.metric("Modelin Açıklama Gücü (R²)", f"{olcum['r2']:.2f}".replace(".", ","))
        c4.metric("Ortalama Günlük Hata", tl(olcum["mae"]))

        # Son 120 gün + tahmin, okunaklı olsun diye
        gorunen = gecmis.tail(120)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gorunen["Tarih"], y=gorunen["Gerçek"], name="Gerçekleşen",
                                 line=dict(color=GRI, width=1.5),
                                 hovertemplate="%{x|%d.%m.%Y}: %{y:,.0f} ₺<extra>Gerçekleşen</extra>"))
        fig.add_trace(go.Scatter(x=gorunen["Tarih"], y=gorunen["Model"], name="Model",
                                 line=dict(color=LACIVERT, width=1.5),
                                 hovertemplate="%{x|%d.%m.%Y}: %{y:,.0f} ₺<extra>Model</extra>"))
        fig.add_trace(go.Scatter(x=pd.concat([ileri["Tarih"], ileri["Tarih"][::-1]]),
                                 y=pd.concat([ileri["Üst"], ileri["Alt"][::-1]]),
                                 fill="toself", fillcolor="rgba(20,163,163,0.15)", line=dict(width=0),
                                 name="%95 aralık", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=ileri["Tarih"], y=ileri["Tahmin"], name="Tahmin",
                                 line=dict(color=TURKUAZ, width=2.5, dash="dash"),
                                 hovertemplate="%{x|%d.%m.%Y}: %{y:,.0f} ₺<extra>Tahmin</extra>"))
        fig.update_yaxes(tickformat=",.0f")
        fig = grafik_duzenle(fig, yukseklik=420)
        fig.update_layout(title="Günlük Ciro: Son 120 Gün ve Tahmin", showlegend=True,
                          legend=dict(orientation="h", y=-0.15, title=None))
        st.plotly_chart(fig)

        with st.container(border=True):
            st.markdown("**💡 Model ne söylüyor?**")
            st.markdown(
                f"- Önümüzdeki 30 günde toplam **{tl(ileri['Tahmin'].sum())}** ciro bekleniyor "
                f"(günlük ortalama {tl(ileri['Tahmin'].mean())})."
            )
            st.markdown(
                f"- Diğer koşullar aynıyken ciro her ay yaklaşık **{tl(katsayilar['Trend'] * 30)}** "
                f"{'artma' if katsayilar['Trend'] > 0 else 'azalma'} eğiliminde."
            )
            if gecmis["Tarih"].min() <= PANDEMI_BAS and gecmis["Tarih"].max() >= PANDEMI_BIT:
                st.markdown(
                    f"- Pandemi kapanması (16 Mart – 31 Mayıs) günlük ciroyu ortalama "
                    f"**{tl(abs(katsayilar['Pandemi']))}** {'düşürmüş' if katsayilar['Pandemi'] < 0 else 'artırmış'}."
                )
            st.markdown(
                f"- Pazartesi'ye göre Cumartesi günleri ortalama {tl(abs(katsayilar['Cumartesi']))}, "
                f"Pazar günleri {tl(abs(katsayilar['Pazar']))} daha az satış yapılıyor. "
                f"Sevkiyat ve depo personeli planı hafta başına ağırlık vermeli."
            )

        with st.expander("Model detayları"):
            st.markdown(
                """
Satış olmayan günler (hafta sonları, tatiller) 0 ciro olarak modele dahil edildi. Değişkenler:
- **Trend:** Veri başlangıcından bu yana geçen gün sayısı
- **Haftanın günü:** Salı–Pazar için 0/1 kukla değişkenler (Pazartesi referans)
- **Pandemi:** 16 Mart – 31 Mayıs 2020 arası 1, diğer günler 0
- **Tatil:** Resmi tatiller ve 20 Temmuz – 3 Ağustos kapanışı için 1

Gölgeli alan, modelin hatalarının standart sapmasına göre hesaplanan yaklaşık %95 aralığıdır.
"""
            )
            tablo = katsayilar.rename("Katsayı (₺/gün)").reset_index().rename(columns={"index": "Değişken"})
            st.dataframe(tablo, hide_index=True,
                         column_config={"Katsayı (₺/gün)": st.column_config.NumberColumn(format="localized")})

with tab_pazarlama:
    referans = bitis + pd.Timedelta(days=1)
    rfm = rfm_hesapla(dff, referans)
    segment_ozet = (
        rfm.groupby("Segment")
        .agg(Musteri=("Monetary", "size"), Ciro=("Monetary", "sum"),
             MedyanGun=("Recency", "median"), MedyanFatura=("Frequency", "median"))
        .reindex(SEGMENT_RENK.keys())
        .fillna(0)
    )
    segment_ozet["CiroPayi"] = segment_ozet["Ciro"] / segment_ozet["Ciro"].sum() * 100
    segment_ozet["MusteriPayi"] = segment_ozet["Musteri"] / segment_ozet["Musteri"].sum() * 100

    # --- KPI kartları: her segment bir kart ---
    kartlar = st.columns(4)
    for kart, (seg, satir) in zip(kartlar, segment_ozet.iterrows()):
        kart.metric(seg, f"{adet(satir['Musteri'])} müşteri", f"Ciro payı {yuzde(satir['CiroPayi'])}",
                    delta_color="off")

    # --- Segment payları ve RFM dağılımı (yan yana) ---
    sol, sag = st.columns(2)

    paylar = segment_ozet.reset_index().melt(
        id_vars="Segment", value_vars=["MusteriPayi", "CiroPayi"], var_name="Ölçü", value_name="Pay"
    )
    paylar["Ölçü"] = paylar["Ölçü"].map({"MusteriPayi": "Müşteri payı", "CiroPayi": "Ciro payı"})
    fig = px.bar(paylar, x="Segment", y="Pay", color="Ölçü", barmode="group",
                 title="Segmentlerin Müşteri ve Ciro Payı (%)",
                 color_discrete_map={"Müşteri payı": GRI, "Ciro payı": TURKUAZ})
    fig.update_traces(hovertemplate="%{x}: %{y:.1f}%<extra></extra>")
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None)
    fig = grafik_duzenle(fig)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.15, title=None))
    sol.plotly_chart(fig)

    fig = px.scatter(rfm.reset_index(), x="Recency", y="Monetary", color="Segment",
                     color_discrete_map=SEGMENT_RENK, log_y=True, opacity=0.7,
                     title="Son Alımdan Bu Yana Geçen Gün vs. Toplam Ciro",
                     custom_data=["CustomerID", "Frequency"])
    fig.update_traces(marker=dict(size=7),
                      hovertemplate="%{customdata[0]}<br>Son alım: %{x} gün önce<br>"
                                    "Ciro: %{y:,.0f} ₺<br>Fatura: %{customdata[1]}<extra></extra>")
    fig.update_xaxes(title="Son alımdan bu yana geçen gün")
    fig.update_yaxes(title="Toplam ciro (₺, log ölçek)")
    fig = grafik_duzenle(fig)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.25, title=None))
    sag.plotly_chart(fig)

    # --- Öne çıkanlar ---
    with st.container(border=True):
        st.markdown("**💡 Öne çıkanlar**")
        sampiyon = segment_ozet.loc["Şampiyonlar"]
        risk = segment_ozet.loc["Risk Altında"]
        kayip = segment_ozet.loc["Kaybedilmiş"]
        st.markdown(
            f"- Müşterilerin {yuzde(sampiyon['MusteriPayi'], 0)} kadarı **Şampiyonlar** segmentinde ve "
            f"cironun {yuzde(sampiyon['CiroPayi'], 0)} kadarını getiriyor. Bu grubu kaybetmemek önceliğimiz olmalı."
        )
        if risk["Musteri"] > 0:
            st.markdown(
                f"- **Risk Altında** {adet(risk['Musteri'])} müşteri var. Bu müşteriler tipik olarak "
                f"{risk['MedyanFatura']:.0f} fatura kesmiş ama son alımlarının üzerinden yaklaşık "
                f"{risk['MedyanGun']:.0f} gün geçmiş. Bu müşterilerin dönemdeki cirosu {tl(risk['Ciro'])}; "
                f"aşağıdaki listeyle satış temsilcileri tek tek aranabilir."
            )
        if kayip["Musteri"] > 0:
            st.markdown(
                f"- **Kaybedilmiş** segmentte {adet(kayip['Musteri'])} müşteri var ama ciroya katkıları yalnızca "
                f"{yuzde(kayip['CiroPayi'])}. Bu gruba pahalı kampanyalar yerine toplu e-posta gibi düşük maliyetli "
                f"yollar daha uygun."
            )

    # --- Risk altındaki müşteriler listesi ---
    st.markdown("**📞 Aranacak müşteriler: Risk Altında segmenti**")
    sehir_bilgisi = dff.groupby("CustomerID")["City"].agg(lambda s: s.mode().iloc[0])
    risk_liste = rfm[rfm["Segment"] == "Risk Altında"].join(sehir_bilgisi).sort_values("Monetary", ascending=False)
    risk_liste = risk_liste.reset_index()[["CustomerID", "City", "SonAlim", "Recency", "Frequency", "Monetary"]]
    st.dataframe(
        risk_liste,
        hide_index=True,
        height=320,
        column_config={
            "CustomerID": "Müşteri",
            "City": "Şehir",
            "SonAlim": st.column_config.DateColumn("Son Alım", format="DD.MM.YYYY"),
            "Recency": "Geçen Gün",
            "Frequency": "Fatura Sayısı",
            "Monetary": st.column_config.NumberColumn("Toplam Ciro (₺)", format="localized"),
        },
    )
    st.download_button(
        "Listeyi indir (CSV)",
        risk_liste.to_csv(index=False).encode("utf-8-sig"),
        file_name="risk_altindaki_musteriler.csv",
        mime="text/csv",
    )

    # --- Yöntem açıklaması ---
    with st.expander("Segmentler nasıl belirlendi?"):
        st.markdown(
            """
Her müşteri için üç değer hesaplandı:
- **Recency:** Seçilen dönemin sonuna göre son alımdan bu yana geçen gün
- **Frequency:** Kesilen fatura sayısı
- **Monetary:** Toplam ciro

Müşteriler her değere göre sıralanıp 4 eşit gruba bölündü ve 1–4 arası puan aldı (4 en iyi).
F ve M puanlarının ortalaması **değer puanı** olarak kullanıldı.

| | Değer puanı ≥ 3 | Değer puanı < 3 |
|---|---|---|
| **R puanı ≥ 3** (yakın zamanda almış) | Şampiyonlar | Sadık Müşteriler |
| **R puanı < 3** (uzun süredir almıyor) | Risk Altında | Kaybedilmiş |
"""
        )
        st.dataframe(
            segment_ozet[["Musteri", "Ciro", "MedyanGun", "MedyanFatura"]].reset_index(),
            hide_index=True,
            column_config={
                "Musteri": "Müşteri",
                "Ciro": st.column_config.NumberColumn("Ciro (₺)", format="localized"),
                "MedyanGun": st.column_config.NumberColumn("Medyan Geçen Gün", format="%.0f"),
                "MedyanFatura": st.column_config.NumberColumn("Medyan Fatura", format="%.0f"),
            },
        )

with tab_tedarik:
    # --- ABC sınırları kullanıcı tarafından değiştirilebilir ---
    s1, s2, _ = st.columns([1, 1, 2])
    a_siniri = s1.slider("A sınıfı: cironun ilk %…", 60, 90, 80, step=5)
    b_siniri = s2.slider("B sınıfı: cironun ilk %…", a_siniri + 5, 99, max(95, a_siniri + 5))

    urun = abc_hesapla(dff, a_siniri, b_siniri)
    sinif_ozet = (
        urun.groupby("Sinif")
        .agg(Urun=("Ciro", "size"), Ciro=("Ciro", "sum"), Adet=("Adet", "sum"))
        .reindex(["A", "B", "C"])
        .fillna(0)
    )
    sinif_ozet["UrunPayi"] = sinif_ozet["Urun"] / sinif_ozet["Urun"].sum() * 100
    sinif_ozet["CiroPayi"] = sinif_ozet["Ciro"] / sinif_ozet["Ciro"].sum() * 100

    # Yavaş dönenler: dönem içinde en az 50 adet satılmış ama son 90 gündür hiç satılmamış ürünler
    urun["GecenGun"] = (bitis - urun["SonSatis"]).dt.days
    yavas = urun[(urun["GecenGun"] > 90) & (urun["Adet"] >= 50)].sort_values("Adet", ascending=False)

    iskonto = 1 - dff["Revenue"].sum() / (dff["Quantity"] * dff["UnitPrice"]).sum()

    # --- KPI kartları ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Satılan Farklı Ürün", adet(len(urun)))
    c2.metric("A Sınıfı Ürün", adet(sinif_ozet.loc["A", "Urun"]),
              f"Ürünlerin {yuzde(sinif_ozet.loc['A', 'UrunPayi'])}", delta_color="off")
    c3.metric("Ortalama İskonto", yuzde(iskonto * 100))
    c4.metric("Yavaş Dönen Ürün", adet(len(yavas)))

    # --- ABC eğrisi ve sınıf özeti (yan yana) ---
    sol, sag = st.columns(2)

    # 43 bin ürünün hepsini çizmek sayfayı yavaşlatır; eğri için 1.000 noktalık örnek yeterli
    egri = urun.reset_index(drop=True)
    egri["UrunPayi"] = (egri.index + 1) / len(egri) * 100
    egri = egri.iloc[:: max(1, len(egri) // 1000)]
    fig = px.area(egri, x="UrunPayi", y="KumulatifPay", color="Sinif", color_discrete_map=SINIF_RENK,
                  title="ABC Eğrisi: Ürünlerin Kümülatif Ciro Payı")
    fig.update_traces(hovertemplate="Ürünlerin ilk %{x:.1f}%'i → cironun %{y:.1f}%'i<extra></extra>")
    fig.update_xaxes(title="Ürünlerin payı (%, cirosu en yüksekten düşüğe)")
    fig.update_yaxes(title="Cironun kümülatif payı (%)", range=[0, 101])
    fig = grafik_duzenle(fig)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.25, title=None))
    sol.plotly_chart(fig)

    paylar = sinif_ozet.reset_index().melt(
        id_vars="Sinif", value_vars=["UrunPayi", "CiroPayi"], var_name="Ölçü", value_name="Pay"
    )
    paylar["Ölçü"] = paylar["Ölçü"].map({"UrunPayi": "Ürün sayısı payı", "CiroPayi": "Ciro payı"})
    paylar["Sinif"] = paylar["Sinif"] + " sınıfı"
    fig = px.bar(paylar, x="Sinif", y="Pay", color="Ölçü", barmode="group",
                 title="Sınıfların Ürün Sayısı ve Ciro Payı (%)",
                 color_discrete_map={"Ürün sayısı payı": GRI, "Ciro payı": LACIVERT})
    fig.update_traces(hovertemplate="%{x}: %{y:.1f}%<extra></extra>")
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    fig = grafik_duzenle(fig)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.15, title=None))
    sag.plotly_chart(fig)

    # --- En çok satan ürünler ve aylık talep (yan yana) ---
    sol, sag = st.columns(2)

    en_cok = urun.sort_values("Adet", ascending=False).head(15).reset_index()
    en_cok["Etiket"] = en_cok["Urun"].str.slice(0, 35)
    fig = px.bar(en_cok, x="Adet", y="Etiket", color="Sinif", orientation="h",
                 color_discrete_map=SINIF_RENK, title="Adet Bazında En Çok Satan 15 Ürün",
                 custom_data=["Urun", "Kategori"])
    fig.update_traces(hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>%{x:,.0f} adet<extra></extra>")
    fig.update_yaxes(title=None, autorange="reversed")
    fig.update_xaxes(title=None, tickformat=",.0f")
    fig = grafik_duzenle(fig, yukseklik=480)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1, title=None))
    sol.plotly_chart(fig)

    ilk5_kat = dff.groupby("Category")["Quantity"].sum().nlargest(5).index
    talep = (
        dff[dff["Category"].isin(ilk5_kat)]
        .groupby(["Month", "Category"])["Quantity"].sum()
        .reset_index()
    )
    talep["Ay Adı"] = talep["Month"].apply(lambda a: AY_ADLARI[a - 1])
    fig = px.line(talep, x="Ay Adı", y="Quantity", color="Category", markers=True,
                  title="En Çok Satan 5 Kategorinin Aylık Talebi (adet)",
                  color_discrete_sequence=[LACIVERT, TURKUAZ, TURUNCU, GRI, "#6C7A96"])
    fig.update_traces(hovertemplate="%{x}: %{y:,.0f} adet<extra></extra>")
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None, tickformat=",.0f")
    fig = grafik_duzenle(fig, yukseklik=480)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.15, title=None))
    sag.plotly_chart(fig)

    # --- Öne çıkanlar ---
    with st.container(border=True):
        st.markdown("**💡 Öne çıkanlar**")
        a = sinif_ozet.loc["A"]
        c = sinif_ozet.loc["C"]
        st.markdown(
            f"- Ürünlerin yalnızca {yuzde(a['UrunPayi'])} kadarı (**{adet(a['Urun'])} ürün**) cironun "
            f"{yuzde(a['CiroPayi'], 0)} kadarını getiriyor. Stokta hiç tükenmemesi gereken liste bu A sınıfı ürünler."
        )
        if c["Urun"] > 0:
            st.markdown(
                f"- **C sınıfı** {adet(c['Urun'])} ürün, ürün çeşidinin {yuzde(c['UrunPayi'], 0)} kadarı olmasına rağmen "
                f"cironun sadece {yuzde(c['CiroPayi'])} kadarını oluşturuyor. Bu ürünlerde stok tutmak yerine "
                f"siparişe göre tedarik düşünülebilir."
            )
        if not talep.empty and talep["Month"].nunique() > 1:
            toplam_talep = dff.groupby("Month")["Quantity"].sum()
            st.markdown(
                f"- Toplam talep en yüksek **{AY_ADLARI[toplam_talep.idxmax() - 1]}** ayında "
                f"({adet(toplam_talep.max())} adet). Depo ve sevkiyat kapasitesi bu dönem için önceden planlanmalı."
            )
        if not yavas.empty:
            st.markdown(
                f"- **{adet(len(yavas))} ürün** dönem içinde en az 50 adet satılmış ama son 90 gündür hiç satılmamış. "
                f"Bu ürünlerin depodaki stoğu gözden geçirilmeli."
            )

    # --- Yavaş dönen ürünler listesi ---
    st.markdown("**📦 Stoğu gözden geçirilecek ürünler (son 90 gündür satışı yok, dönemde en az 50 adet satılmış)**")
    yavas_tablo = yavas.reset_index()[["StockCode", "Urun", "Kategori", "Sinif", "Adet", "SonSatis", "GecenGun"]]
    st.dataframe(
        yavas_tablo,
        hide_index=True,
        height=320,
        column_config={
            "StockCode": st.column_config.TextColumn("Stok Kodu"),
            "Urun": "Ürün",
            "Sinif": "ABC",
            "Adet": st.column_config.NumberColumn("Satılan Adet", format="%d"),
            "SonSatis": st.column_config.DateColumn("Son Satış", format="DD.MM.YYYY"),
            "GecenGun": "Geçen Gün",
        },
    )
    st.download_button(
        "Listeyi indir (CSV)",
        yavas_tablo.to_csv(index=False).encode("utf-8-sig"),
        file_name="yavas_donen_urunler.csv",
        mime="text/csv",
    )

    # --- Yöntem açıklaması ---
    with st.expander("ABC analizi nasıl yapıldı?"):
        st.markdown(
            f"""
Ürünler seçilen dönemdeki cirolarına göre büyükten küçüğe sıralandı ve her ürünün kümülatif ciro payı hesaplandı.
- **A sınıfı:** Cironun ilk %{a_siniri} dilimini oluşturan ürünler. Stok seviyesi yakından takip edilmeli.
- **B sınıfı:** Sonraki dilim (cironun %{b_siniri - a_siniri} kadarı). Düzenli ama daha seyrek kontrol yeterli.
- **C sınıfı:** Kalan dilim (cironun %{100 - b_siniri} kadarı). Az stok veya siparişe göre tedarik uygun.

Sınırları sayfanın üstündeki kaydırıcılarla değiştirebilirsiniz; tüm grafikler ve öneriler buna göre yeniden hesaplanır.
"""
        )
