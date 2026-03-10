@echo off
set REPO=%~dp0..
set BEST=%REPO%\runs\ball\train\weights\best.pt
set DEST=%REPO%\backend\ball_best.pt

if not exist "%BEST%" (
    echo Trained weights not found at: %BEST%
    echo Wait for training to finish, then run this again.
    exit /b 1
)

copy /Y "%BEST%" "%DEST%"
echo Copied best.pt to backend\ball_best.pt
echo.
echo Set this before starting the backend (CMD):
echo set BALL_DETECT_MODEL=%DEST%
exit /b 0
