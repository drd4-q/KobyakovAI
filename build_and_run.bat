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
echo   [1] Launch Native C/Zig GUI Studio (No Electron, 460 KB)
echo   [2] Interactive CLI Chat (NVIDIA RTX 2060 - CUDA)
echo   [3] Interactive CLI Chat (Pure C99 Engine - CPU)
echo   [4] Fast GPU Training (CUDA RTX 2060, 300 steps)
echo   [5] Deep GPU Training (CUDA RTX 2060, 1000 steps - All Datasets)
echo   [6] Screen and Window Vision CLI (SmolVLM-500M)
echo   [7] Cross-Compile for ARM64 (Linux, Raspberry Pi, Apple Silicon)
echo   [8] Cross-Compile for RISC-V (RV64GC Linux)
echo   [9] Recompile Windows C Engine (zig cc)
echo   [10] Exit
echo ============================================================
set /p opt="Enter choice [1-10]: "

if "%opt%"=="1" goto opt1
if "%opt%"=="2" goto opt2
if "%opt%"=="3" goto opt3
if "%opt%"=="4" goto opt4
if "%opt%"=="5" goto opt5
if "%opt%"=="6" goto opt6
if "%opt%"=="7" goto opt7
if "%opt%"=="8" goto opt8
if "%opt%"=="9" goto opt9
if "%opt%"=="10" goto opt10
echo Invalid choice, please enter 1-10.
goto menu

:opt1
echo.
echo Launching Native C/Zig Studio GUI...
call run_gui.bat
goto finish

:opt2
echo.
echo Launching GPU Shell [CUDA]...
python run_gpu.py
goto finish

:opt3
echo.
echo Launching CPU Native C Shell...
engine_c.exe model.bin tokenizer.bin
goto finish

:opt4
echo.
echo Starting Fast GPU Training [300 steps]...
python train_gpu.py --steps 300 --domain code
goto finish

:opt5
echo.
echo Starting Deep GPU Training [1000 steps, All Datasets]...
python train_gpu.py --steps 1000 --domain all
goto finish

:opt6
echo.
echo Launching Screen and Window Vision CLI...
python screen_vision.py -i
goto finish

:opt7
echo.
echo Cross-compiling engine.c for ARM64 [aarch64-linux]...
zig cc -O3 engine.c -target aarch64-linux -o engine_arm64
if exist "engine_arm64" (
    echo [OK] Successfully built ARM64 binary: engine_arm64
) else (
    echo [ERROR] ARM64 compilation failed.
)
goto finish

:opt8
echo.
echo Cross-compiling engine.c for RISC-V [riscv64-linux]...
zig cc -O3 engine.c -target riscv64-linux -o engine_riscv64
if exist "engine_riscv64" (
    echo [OK] Successfully built RISC-V binary: engine_riscv64
) else (
    echo [ERROR] RISC-V compilation failed.
)
goto finish

:opt9
echo.
echo Recompiling engine.c and kobyakov_gui.c with zig cc...
zig cc -O3 engine.c -o engine_c.exe
zig cc -O3 kobyakov_gui.c -lgdi32 -luser32 -lmsimg32 -lwininet -o KobyakovAI.exe
echo [OK] engine_c.exe and KobyakovAI.exe updated!
goto finish

:opt10
exit /b 0

:finish
echo.
pause
