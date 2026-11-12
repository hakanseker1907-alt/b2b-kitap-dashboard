import streamlit as st
import pandas as pd
import plotly.express as px

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
AY_ADLARI = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
             "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
GUN_ADLARI = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]


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

with tab_pazarlama:
    st.info("Bu sekme hazırlanıyor.")

with tab_tedarik:
    st.info("Bu sekme hazırlanıyor.")
