@echo off
REM orquestrador_downloads.bat
REM Baixa todos os datasets em sequência para o HendiCode
REM Cada chamada usa baixar_dataset.py

set PYTHON=python
set SCRIPT=C:\Users\HENDI\Desktop\HendiCode\baixar_dataset.py
set LOGS=C:\Users\HENDI\Desktop\dados_5gb
set START=%TIME%

echo ========================================
echo Iniciando download de 5GB de datasets
echo Data: %DATE% %TIME%
echo ========================================

REM ===== AGENTE 1: github-code-2025 parte 1 (Python/JS - arquivos 00-09) =====
echo [1/10] github-code-2025 parte 1 - Python/JS...
%PYTHON% "%SCRIPT%" "nick007x/github-code-2025" "" "content" 500 "github_code_2025_p1"

REM ===== AGENTE 2: github-code-2025 parte 2 =====
echo [2/10] github-code-2025 parte 2...
%PYTHON% "%SCRIPT%" "nick007x/github-code-2025" "" "content" 500 "github_code_2025_p2"

REM ===== AGENTE 3: codeparrot clean train =====
echo [3/10] codeparrot-clean-train...
%PYTHON% "%SCRIPT%" "codeparrot/codeparrot-clean-train" "" "content" 500 "codeparrot_clean"

REM ===== AGENTE 4: code_search_net (todas linguagens) =====
echo [4/10] code_search_net completo...
%PYTHON% "%SCRIPT%" "code-search-net/code_search_net" "all" "func_code_string,func_documentation_string,whole_func_string" 500 "code_search_net_all"

REM ===== AGENTE 5: rStar-Coder =====
echo [5/10] microsoft/rStar-Coder...
%PYTHON% "%SCRIPT%" "microsoft/rStar-Coder" "" "prompt,response" 300 "rstar_coder"

REM ===== AGENTE 6: cc100 português =====
echo [6/10] cc100 lang=pt...
%PYTHON% "%SCRIPT%" "statmt/cc100" "lang=pt" "text" 500 "cc100_portugues"

REM ===== AGENTE 7: CodeAlpaca 20K completo =====
echo [7/10] CodeAlpaca 20K + Evol-Instruct...
%PYTHON% "%SCRIPT%" "sahil2801/CodeAlpaca-20k" "" "instruction,input,output" 50 "codealpaca_20k"

REM ===== AGENTE 8: Evol-Instruct-80k =====
echo [8/10] Evol-Instruct-80k...
%PYTHON% "%SCRIPT%" "nickrosh/Evol-Instruct-Code-80k-v1" "" "instruction,output" 50 "evol_instruct_80k"

REM ===== AGENTE 9: m-a-p/Code-Feedback =====
echo [9/10] Code-Feedback-Ada...
%PYTHON% "%SCRIPT%" "m-a-p/Code-Feedback" "" "prompt,answer" 50 "code_feedback"

REM ===== AGENTE 10: hasankursun/github-code-2025-language-split =====
echo [10/10] github-code-2025 language split...
%PYTHON% "%SCRIPT%" "hasankursun/github-code-2025-language-split" "" "content" 400 "github_code_split"

echo ========================================
echo Download concluido!
echo Inicio: %START%
echo Fim:    %TIME%
echo ========================================
echo.
echo Gerando relatorio...
%PYTHON% -c "
import os, glob
d = r'C:\Users\HENDI\Desktop\dados_5gb'
txts = sorted(glob.glob(os.path.join(d, '*.txt')))
total = 0
nomes = {}
for f in txts:
    sz = os.path.getsize(f)
    name = os.path.basename(f)
    nomes[name] = sz / 1e6
    total += sz
    print(f'  {name}: {sz/1e6:.2f} MB')
print(f'\nTotal: {total/1e6:.2f} MB ({total/1e9:.2f} GB)')
print(f'Arquivos: {len(txts)}')
" 2>&1

pause
