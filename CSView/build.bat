@echo off
rem Builds cs2-match-viewer with the MSVC toolchain.
rem
rem   build.bat            -> builds into .\build
rem   build.bat <dir>      -> builds into <dir>
rem
rem Why this exists rather than a plain "cmake -B build -S .":
rem   * MSVC is never on PATH; it needs vcvars64.bat first.
rem   * The Visual Studio generator needs the MSBuild VC integration, which a
rem     bare Build Tools install may not have. Ninja only needs cl.exe, and VS
rem     ships one, so this uses that.
setlocal EnableDelayedExpansion

set SRC=%~dp0
if "%SRC:~-1%"=="\" set SRC=%SRC:~0,-1%

set BUILD=%~1
if "%BUILD%"=="" set BUILD=%SRC%\build

set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
if not exist "%VSWHERE%" (
  echo Visual Studio Installer not found.
  echo Install the C++ build tools with:
  echo   winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
  exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VS=%%i
if "%VS%"=="" (
  echo No Visual Studio instance with the C++ toolset was found.
  echo Add it with: winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
  exit /b 1
)

call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 ( echo vcvars64.bat failed & exit /b 1 )

set CMAKEDIR=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake
if exist "%CMAKEDIR%\CMake\bin\cmake.exe" set PATH=%CMAKEDIR%\CMake\bin;%CMAKEDIR%\Ninja;%PATH%

where cmake >nul 2>&1 || ( echo cmake not found. Install it with: winget install --id Kitware.CMake & exit /b 1 )

rem MSVC refuses object paths longer than 250 characters. Deep source trees
rem blow that budget before the build directory is even appended, so say so
rem plainly instead of failing later with "Cannot open compiler generated file".
call :strlen SRCLEN "%SRC%"
if !SRCLEN! GTR 120 (
  echo.
  echo NOTE: this source path is !SRCLEN! characters deep. MSVC caps object
  echo       paths at 250, so building in-tree may fail. Pass a short build
  echo       directory instead, for example:  build.bat C:\cs2build
  echo.
)

cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -S "%SRC%" -B "%BUILD%" || exit /b 1
cmake --build "%BUILD%" || exit /b 1

echo.
echo Built: %BUILD%\cs2mv.exe
echo Tests: "%BUILD%\cs2mv_tests.exe" "%SRC%\tests\data"
exit /b 0

:strlen
set "s=%~2#"
set "len=0"
:strlen_loop
if "!s:~1!"=="" goto :strlen_done
set "s=!s:~1!"
set /a len+=1
goto :strlen_loop
:strlen_done
set "%~1=%len%"
exit /b 0
