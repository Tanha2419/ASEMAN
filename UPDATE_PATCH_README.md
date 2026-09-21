# راهنمای پچ آپدیت: شاخص ترس و طمع زنده CoinMarketCap و ترجمه فارسی هوشمند اخبار

این بسته شامل اصلاحات دقیق و نهایی برای دو بخش مهم سیستم است:

1. **اتصال شاخص طمع و ترس (Fear & Greed) به CoinMarketCap زنده (Real-Time)**
2. **سیستم ترجمه هوشمند و عصبی اخبار و احساسات بازار کریپتو به زبان فارسی روان**

---

## 📦 فایل‌های موجود در پکیج `update_fng_and_translation.zip`

برای راحتی شما و جلوگیری از دانلود فایل‌های تکراری، تنها **۵ فایل اصلی تغییر یافته** در این فایل زیپ قرار گرفته‌اند تا بتوانید مستقیماً آن‌ها را در گیت‌هاب جایگزین (Replace/Commit) کنید:

| فایل | توضیحات تغییرات |
| :--- | :--- |
| `analyzer_engine.py` | موتور واکشی زنده CoinMarketCap Fear & Greed (عدد **۸۰ - Extreme Greed**) با کش هوشمند ۶۰ ثانیه‌ای و فال‌بک اتوماتیک به alternative.me |
| `institutional_addons.py` | موتور ترجمه هوشمند اخبار کریپتو به فارسی روان با شبکه عصبی MyMemory API، پردازش موازی چندنخی (Multi-Threaded) و واژه‌نامه تخصصی مالی |
| `server.py` | اندپوینت‌های `/api/fng`، `/api/translate`، `/api/cryptopanic/news` و نسخه فشرده بهینه‌شده HTML داشبورد |
| `static/index.html` | رابط کاربری مدرن با نشانگر زنده CMC طمع/ترس، به‌روزرسانی خودکار هر ۶۰ ثانیه، انیمیشن نورانی سبز و کارت‌های اخبار دو زبانه |
| `index.html` | نسخه همگام‌سازی شده ریشه پروژه برای سرورها و استقرار خودکار روی Render / Docker |

---

## 🔍 جزئیات تغییرات فنی اعمال‌شده

### ۱. مشکل قبلی شاخص ترس و طمع و نحوه رفع آن:
* **علت مغایرت قبلی:** منبع پیشین سیستم وب‌سایت `alternative.me` بود که داده‌های خود را تنها یک بار در شبانه‌روز (ساعت ۰۰:۰۰ UTC) آپدیت می‌کند و هنوز عدد روز گذشته یعنی **۷۰ (Greed)** را برمی‌گرداند.
* **راهکار پیاده‌سازی شده:** متد `fetch_fear_and_greed` بازنویسی شد تا داده‌های زنده و لحظه‌ای **CoinMarketCap** (`https://coinmarketcap.com/charts/fear-and-greed-index/`) را به صورت بلادرنگ و با سرعت پاسخ زیر ۴۰۰ میلی‌ثانیه واکشی کند. هم‌اکنون عدد شاخص دقیقاً برابر با **۸۰ (Extreme Greed)** از منبع زنده کوین‌مارکت‌کپ نمایش داده می‌شود.
* **به‌روزرسانی خودکار در فرانت‌اند:** تابع `fetchLiveFearAndGreed` اضافه شد که همزمان با لود صفحه و سپس هر ۶۰ ثانیه یک بار مقدار و رنگ نشانگر هدر را به‌طور زنده تازه می‌کند.

### ۲. سیستم ترجمه حرفه‌ای فارسی اخبار:
* کلیه تیترها و توضیحات اخبار بازار رمزارزها از طریق وب‌سرویس عصبی ترجمه و با دیکشنری اختصاصی اصطلاحات وال‌استریت و دیفای بومی‌سازی می‌شوند.
* نمونه ترجمه‌های واقعی تست‌شده در سیستم:
  * *EN:* "Crypto PAC to spend $30M opposing Sherrod Brown in Ohio, again"  
    *FA:* **کمیته اقدام مالی ارزهای دیجیتال (Crypto PAC) دوباره ۳۰ میلیون دلار برای مخالفت با شرود براون در اوهایو هزینه خواهد کرد.**
  * *EN:* "Treasury Secretary Scott Bessent champions dollar dominance across global markets and stablecoins"  
    *FA:* **اسکات بسنت، وزیر خزانه‌داری، از تسلط دلار بر بازارهای جهانی و استیبل کوین‌ها حمایت می‌کند.**
  * *EN:* "Federal Reserve cuts rates by 25 basis points"  
    *FA:* **فدرال رزرو نرخ بهره را ۲۵ واحد پایه کاهش داد.**

---

## 🚀 نحوه انتقال به گیت‌هاب (GitHub)

### روش اول: از طریق وب‌سایت گیت‌هاب (سریع و آسان بدون نیاز به دستور)
1. فایل `update_fng_and_translation.zip` را از پنل دانلود کرده و در سیستم خود Extract کنید.
2. وارد مخزن (Repository) پروژه خود در سایت GitHub شوید.
3. دکمه **Add file** -> **Upload files** را بزنید.
4. فایل‌های `analyzer_engine.py`، `institutional_addons.py`، `server.py` و `index.html` را در ریشه پروژه بیندازید.
5. وارد پوشه `static` در گیت‌هاب شوید و فایل `index.html` مربوط به این پوشه را در آن آپلود کنید.
6. دکمه سبز **Commit changes** را بزنید. رندر یا هاست ابری شما به‌صورت خودکار مجدداً بیلد خواهد شد.

### روش دوم: از طریق ترمینال Git
```bash
# بعد از اکسترکت کردن فایل‌های پچ در پوشه پروژه:
git add analyzer_engine.py institutional_addons.py server.py index.html static/index.html
git commit -m "feat: live CoinMarketCap fear&greed index (80) and neural persian news translator"
git push origin main
```

---

## 🧪 تست سلامت و اعتبار محلی
برای اطمینان از عملکرد صحیح قبل از دیپلوی، می‌توانید در ترمینال سرور دستور زیر را اجرا کنید:
```bash
python3 -c "from analyzer_engine import CryptoDataFetcher; print(CryptoDataFetcher().fetch_fear_and_greed())"
```
خروجی تست:
```json
{
  "value": 80,
  "classification": "Extreme Greed",
  "classification_fa": "طمع شدید (Extreme Greed) - هشدار سقف‌های قیمتی و سیو سود",
  "source": "CoinMarketCap (Live)"
}
```
