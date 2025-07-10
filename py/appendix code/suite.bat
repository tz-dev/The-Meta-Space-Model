@echo off
cls
setlocal EnableDelayedExpansion

:: Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed. Please download and install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check if pip is available
where pip >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo pip is not installed. Please ensure pip is installed and in your PATH.
    pause
    exit /b 1
)

:: Check Python version (ensure it's Python 3)
for /f "tokens=2 delims= " %%i in ('python --version 2^>nul') do set PYTHON_VERSION=%%i
if "!PYTHON_VERSION:~0,2!" neq "3." (
    echo This script requires Python 3.x. Please download and install Python 3 from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: List of required packages
set "PACKAGES=tkinter tabulate subprocess threading sys io os re csv json logging PIL"

:: Check and install missing packages
set "MISSING_PACKAGES="
for %%p in (%PACKAGES%) do (
    python -c "import %%p" 2>nul
    if !ERRORLEVEL! neq 0 (
        set "MISSING_PACKAGES=!MISSING_PACKAGES! %%p"
    )
)

:: Install missing packages if any
if not "!MISSING_PACKAGES!"=="" (
    echo Installing missing packages: !MISSING_PACKAGES!
    for %%p in (!MISSING_PACKAGES!) do (
        python -m pip install %%p
        if !ERRORLEVEL! neq 0 (
            echo Failed to install %%p. Please install it manually using 'pip install %%p'
            pause
            exit /b 1
        )
    )
)

:: Check if tkinter is available (special case, as it's part of standard library but might not be included in some installations)
python -c "import tkinter" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Tkinter is not available. Please ensure you have a Python installation with Tkinter support.
    echo You may need to install python3-tk (e.g., 'sudo apt-get install python3-tk' on Ubuntu.
    pause
    exit /b 1
)

:: Run the Python script
echo Starting 00_script_suite.py...
python 00_script_suite.py
if %ERRORLEVEL% neq 0 (
    echo Failed to run 00_script_suite.py
    pause
    exit /b 1
)

echo Script executed successfully.
