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

:menu
echo.
echo ============================================================
echo   KobyakovAI - Hierarchical MoE (144 Nodes) + Vision
echo   SELECT RUNTIME / ACTION:
echo ============================================================
echo   [1] GPU Runtime [NVIDIA RTX 2060 - CUDA Accelerated]
echo   [2] CPU Runtime [Pure Native C / Cross-Platform Engine]
echo   [3] Fast GPU Training [CUDA RTX 2060, 300 steps]
echo   [4] Deep GPU Training [CUDA RTX 2060, 1000 steps - All Datasets]
echo   [5] Screen and Window Vision Shell [SmolVLM-500M - CUDA Eye]
echo   [6] Cross-Compile for ARM64 [Linux, Raspberry Pi, Apple Silicon]
echo   [7] Cross-Compile for RISC-V [RV64GC Linux]
echo   [8] Recompile Windows C Engine [zig cc]
echo   [9] Exit
echo ============================================================
set /p opt="Enter choice [1-9]: "

if "%opt%"=="1" goto opt1
if "%opt%"=="2" goto opt2
if "%opt%"=="3" goto opt3
if "%opt%"=="4" goto opt4
if "%opt%"=="5" goto opt5
if "%opt%"=="6" goto opt6
if "%opt%"=="7" goto opt7
if "%opt%"=="8" goto opt8
if "%opt%"=="9" goto opt9
echo Invalid choice, please enter 1-9.
goto menu

:opt1
echo.
echo Launching GPU Shell [CUDA]...
python run_gpu.py
goto finish

:opt2
echo.
echo Launching CPU Native C Shell...
engine_c.exe model.bin tokenizer.bin
goto finish

:opt3
echo.
echo Starting Fast GPU Training [300 steps]...
python train_gpu.py --steps 300 --domain code
goto finish

:opt4
echo.
echo Starting Deep GPU Training [1000 steps, All Datasets]...
python train_gpu.py --steps 1000 --domain all
goto finish

:opt5
echo.
echo Launching Screen and Window Vision Shell...
python screen_vision.py -i
goto finish

:opt6
echo.
echo Cross-compiling engine.c for ARM64 [aarch64-linux]...
zig cc -O3 engine.c -target aarch64-linux -o engine_arm64
if exist "engine_arm64" (
    echo [OK] Successfully built ARM64 binary: engine_arm64
) else (
    echo [ERROR] ARM64 compilation failed.
)
goto finish

:opt7
echo.
echo Cross-compiling engine.c for RISC-V [riscv64-linux]...
zig cc -O3 engine.c -target riscv64-linux -o engine_riscv64
if exist "engine_riscv64" (
    echo [OK] Successfully built RISC-V binary: engine_riscv64
) else (
    echo [ERROR] RISC-V compilation failed.
)
goto finish

:opt8
echo.
echo Recompiling engine.c with zig cc for Windows x86_64...
zig cc -O3 engine.c -o engine_c.exe
echo [OK] engine_c.exe updated!
goto finish

:opt9
exit /b 0

:finish
echo.
pause
