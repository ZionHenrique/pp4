@echo off
echo ========================================
echo Iniciando Aplicacao de Analise Nutricional
echo ========================================
echo.

echo Verificando dependencias...
pip install -r requirements.txt

echo.
echo Iniciando servidor Flask...
python app.py

pause

