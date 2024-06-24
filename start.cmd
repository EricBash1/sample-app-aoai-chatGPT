if not defined in_subprocess (cmd /k set in_subprocess=y ^& %0 %*) & exit )

cd frontend
echo.
echo Building frontend
echo.
call npm run build
if "%errorlevel%" neq "0" (
    echo Failed to build frontend
    exit /B %errorlevel%
)