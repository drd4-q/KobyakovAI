@echo off
echo ============================================================
echo   OBUCHENIE VSEKH MOZGOV PO OCHEREDI
echo ============================================================
echo.

echo [1/4] Obuchenie mozga: PROGRAMMIROVANIE (code)
echo --------------------------------------------
python train.py --domain code
echo.

echo [2/4] Obuchenie mozga: MATEMATIKA (math)
echo --------------------------------------------
python train.py --domain math
echo.

echo [3/4] Obuchenie mozga: FIZIKA (physics)
echo --------------------------------------------
python train.py --domain physics
echo.

echo [4/4] Obuchenie mozga: OBSHCHENIE (chat)
echo --------------------------------------------
python train.py --domain chat
echo.

echo ============================================================
echo   VSE MOZGI OBUCHENY!
echo   Zapustite: python generate.py
echo ============================================================
pause
