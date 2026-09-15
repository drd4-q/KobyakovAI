@echo off
cls
echo ============================================================
echo   MoE AI ENGINE (C / Zig Compiler + CUDA Acceleration)
echo ============================================================
echo.

if not exist "model.bin" (
    echo [1/3] model.bin not found, generating weights...
    python export_weights.py
)

if not exist "train_tokens.bin" (
    echo [2/3] train_tokens.bin not found, preparing tokens...
    python prepare_train_data.py
)

if not exist "engine_c.exe" (
    echo [3/3] Compiling C engine with zig cc...
    zig cc -O3 engine.c -o engine_c.exe
)

echo.
echo ============================================================
echo   SELECT ACTION:
echo   1. Start Interactive Chat / Prompt Shell (engine_c.exe)
echo   2. Run Fast GPU Training (CUDA RTX 2060, 300 steps)
echo   3. Run Deep GPU Training (CUDA RTX 2060, 1000 steps)
echo   4. Run Native C CPU Training (train_c.exe, 100 steps)
echo   5. Recompile C Engine with zig cc
echo   6. Exit
echo ============================================================
set /p opt="Enter choice (1-6): "

if "%opt%"=="1" (
    echo.
    engine_c.exe model.bin tokenizer.bin
)
if "%opt%"=="2" (
    echo.
    python train_gpu.py --steps 300 --domain code
)
if "%opt%"=="3" (
    echo.
    python train_gpu.py --steps 1000 --domain code
)
if "%opt%"=="4" (
    echo.
    if not exist "train_c.exe" (
        zig cc -O3 train.c -o train_c.exe
    )
    train_c.exe model.bin train_tokens.bin 100 0.001
)
if "%opt%"=="5" (
    echo.
    zig cc -O3 engine.c -o engine_c.exe
    echo Engine recompiled!
)

echo.
pause
