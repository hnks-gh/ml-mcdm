@echo off
cd /d "C:\Users\hoang\Documents\ml-mcdm"
"C:\Program Files\python\python.exe" -m pytest tests/ --tb=short -q --no-header 2>&1 > test_output.txt
echo Exit: %ERRORLEVEL% >> test_output.txt
