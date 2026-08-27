@echo off
setlocal
cd /d "%~dp0"

set "PYEXE=.venv\Scripts\python.exe"

if not exist "%PYEXE%" (
    echo [setup] Creating virtualenv...
    python -m venv .venv
    if errorlevel 1 goto :no_python
)

echo [setup] Installing dependencies...
"%PYEXE%" -m pip install -e . >nul 2>&1
if errorlevel 1 echo [setup] WARNING: pip install failed

if not exist "models" mkdir "models"
if not exist "tools" mkdir "tools"
if not exist "logs" mkdir "logs"

set "LLM=models\Qwen3VL-4B-Instruct-Q4_K_M.gguf"
set "MMPROJ=models\mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"

if not exist "%LLM%" (
    echo [models] Downloading %LLM% ...
    curl.exe -L -C - --fail --progress-bar -o "%LLM%" "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3VL-4B-Instruct-Q4_K_M.gguf"
    if errorlevel 1 echo [models] WARNING: failed to download LLM
)

if not exist "%MMPROJ%" (
    echo [models] Downloading %MMPROJ% ...
    curl.exe -L -C - --fail --progress-bar -o "%MMPROJ%" "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"
    if errorlevel 1 echo [models] WARNING: failed to download mmproj
)

rem --- Detect CUDA vs CPU build ---
set "SERVER_EXE=tools\llama-server.exe"
set "NGL="

if exist "tools\cuda\llama-server.exe" if exist "tools\cuda\ggml-cuda.dll" (
    set "SERVER_EXE=tools\cuda\llama-server.exe"
    set "NGL=-ngl 99"
    echo [server] CUDA build detected: tools\cuda\
) else (
    echo [server] Using CPU build: tools\
)

if exist "%SERVER_EXE%" (
    if exist "%LLM%" (
        echo [server] Starting llama.cpp server on :8090 with GPU offload...
        start "llama-server" "%SERVER_EXE%" -m "%LLM%" --mmproj "%MMPROJ%" -c 8192 --port 8090 --embedding --pooling mean %NGL%
    ) else (
        echo [server] Model not found, skipping server start.
    )
) else (
    echo [server] llama-server.exe not found.
    echo   Download the latest llama.cpp release and put it into tools\ or tools\cuda\:
    echo   https://github.com/ggml-org/llama.cpp/releases
    echo   Without the server the GUI runs in mock mode.
)

echo [gui] Launching GroupCOT GUI...
start "GroupCOT GUI" "%PYEXE%" -m groupcot.gui
exit /b 0

:no_python
echo [setup] Python not found. Install Python 3.10+ and try again.
pause
exit /b 1
