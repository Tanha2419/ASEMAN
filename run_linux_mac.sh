#!/bin/bash
echo "========================================================"
echo "  ⚡ اجرای داشبورد هوشمند کریپتو CryptoAgent AI"
echo "========================================================"
echo ""
echo "در حال بررسی و نصب پیش‌نیازها..."
python3 -m pip install -r requirements.txt
echo ""
echo "در حال راه‌اندازی سرور روی پورت 8000..."
echo "در مرورگر کروم آدرس زیر را باز کنید:"
echo "   http://localhost:8000"
echo ""
python3 server.py
