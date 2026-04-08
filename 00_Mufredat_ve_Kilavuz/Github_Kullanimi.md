# Veri Bilimi Akademisi: GitHub Kullanım Kılavuzu 🚀

Bu kılavuz, mentorluk programı boyunca ödevlerinizi teslim etmeniz ve projelerde iş birliği yapmanız için gereken temel GitHub bilgilerini içerir.

## 1. İlk Kurulum (Bir Kez Yapılacak)

Git'i bilgisayarınıza kurduktan sonra terminal (veya Git Bash) açarak kendinizi tanıtın:

```bash
git config --global user.name "Adınız Soyadınız"
git config --global user.email "eposta@örnek.com"
```

## 2. Program Reposunu Fork'lamak

Proje ana sayfasındaki **"Fork"** butonuna basarak depoyu kendi GitHub hesabınıza kopyalayın. Artık `github.com/kullanici-adiniz/AU-Veri-Bilimi-Akademisi-2026` adresinde size ait bir kopya olacak.

## 3. Bilgisayara İndirmek (Clone)

Kendi sayfanızdaki yeşil **"Code"** butonuna basıp linki kopyalayın ve terminalde şu komutu çalıştırın:

```bash
git clone https://github.com/kullanici-adiniz/AU-Veri-Bilimi-Akademisi-2026.git
cd AU-Veri-Bilimi-Akademisi-2026
```

## 4. Ödev Teslim Süreci (Her Hafta)

Her hafta ödevinizi göndermek için şu adımları izleyin:

### Adım 1: Dosyalarınızı Hazırlayın
Ödev dosyanızı (Jupyter Notebook veya .py) size ait klasöre kopyalayın:
`02_Mentiler/adiniz_soyadiniz/hafta_01/odev.ipynb`

### Adım 2: Değişiklikleri Kaydedin
```bash
git add .
git commit -m "1. Hafta ödevi tamamlandı"
```

### Adım 3: GitHub'a Gönderin
```bash
git push origin main
```

## 5. Pull Request (PR) Oluşturma

Ödevinizi ana depoya bildirmek için:
1. Kendi GitHub sayfanıza gidin.
2. Üstte çıkan **"Contribute"** -> **"Open Pull Request"** butonuna basın.
3. Başlık olarak "Hafta 1: Ad Soyad" yazarak isteğinizi gönderin.

---
> [!IMPORTANT]
> **DİKKAT:** Sadece kendi klasörünüzün (`02_Mentiler/adiniz_soyadiniz`) içindeki dosyalarda değişiklik yapın. Diğer dosyaları değiştirmemeye özen gösterin.

Anlamadığınız bir yer olursa mentorunuza sormaktan çekinmeyin! Başarılar.
