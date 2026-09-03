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

rem --- Detect CUDA toolkit + MSVC compiler ---
set "CUDA_PATH="
set "CUDA_BIN="
set "VCVARS="
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
    set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
if not defined VCVARS if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)
if defined CUDA_PATH if defined VCVARS (
    echo [gpu] CUDA toolkit + MSVC found - GPU mode available.
) else (
    echo [gpu] CUDA toolkit or MSVC not found - will run on CPU.
)

rem --- Ensure llama-cpp-python is built WITH CUDA ---
rem Note: this used to be one deeply nested if/else(...) block (3 levels),
rem including an inline "python -c ..." one-liner whose own parentheses
rem (os.path.exists(...), SystemExit(...)) confused cmd.exe's paren-matching
rem for the enclosing block even though they were inside quotes -- it aborted
rem with ") was unexpected at this time." Flat goto-based control flow avoids
rem that whole class of cmd.exe parser gotcha.
set "NEED_CUDA_BUILD=0"
set "LLAMALIB=%~dp0.venv\Lib\site-packages\llama_cpp\lib"
"%PYEXE%" -c "import llama_cpp" >nul 2>&1
if errorlevel 1 set "NEED_CUDA_BUILD=1"
if not exist "%LLAMALIB%\ggml-cuda.dll" set "NEED_CUDA_BUILD=1"

if not "%NEED_CUDA_BUILD%"=="1" goto :cuda_build_done
if not defined CUDA_PATH goto :cuda_build_unavailable
if not defined VCVARS goto :cuda_build_unavailable

echo [gpu] Building llama-cpp-python with CUDA (this takes a few minutes)...
call "%VCVARS%"
set "PATH=%CUDA_BIN%;%PATH%"
set "CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86"
"%PYEXE%" -m pip install --force-reinstall --no-cache-dir --no-binary llama-cpp-python llama-cpp-python==0.3.35
if errorlevel 1 (
    echo [gpu] CUDA build FAILED - falling back to CPU/mock.
    goto :cuda_build_done
)
copy /Y "%CUDA_BIN%\cudart64_12.dll" "%LLAMALIB%\" >nul
copy /Y "%CUDA_BIN%\cublas64_12.dll" "%LLAMALIB%\" >nul
copy /Y "%CUDA_BIN%\cublasLt64_12.dll" "%LLAMALIB%\" >nul
echo [gpu] llama-cpp-python built with CUDA and runtime DLLs copied.
goto :cuda_build_done

:cuda_build_unavailable
echo [gpu] Cannot build CUDA backend (no toolkit/compiler). Running CPU/mock.

:cuda_build_done

rem --- llama-server is no longer used (ARCHITECTURE.md §11): the GUI/CLI
rem talk to LlamaCppEngine in-process (llama-cpp-python), not to a remote
rem llama-server over HTTP. If tools\llama-server.exe / tools\cuda\ still
rem have old binaries from before that decision, they're just unused now --
rem this script no longer starts anything from them.

echo [gui] Launching GroupCOT GUI (GPU if llama-cpp-python built with CUDA)...
if defined CUDA_BIN set "PATH=%CUDA_BIN%;%PATH%"
start "GroupCOT GUI" "%PYEXE%" -m groupcot.gui
exit /b 0

:no_python
echo [setup] Python not found. Install Python 3.10+ and try again.
pause
exit /b 1
