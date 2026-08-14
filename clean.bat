@echo off
:: =============================================================================
:: clean.bat  -  delete files that match .gitignore
:: =============================================================================
::
:: usage
::   double-click in repo root, or:
::     clean.bat
::
:: command actually run  -- after you type Y
::   git clean -fdX
::
:: git clean flags
::   -f    required; actually delete
::   -d    also remove ignored directories
::   -X    ONLY gitignored paths, not other untracked files
::
:: what goes away  -- see .gitignore
::   Data/grid, Data/*.csv, other Data files except the ai_mlp keep-list
::   Data/ai_mlp/ckpts/epoch_*.pt and Data/ai_mlp/epoch_*.pt
::   Data/ai_mlp/test.csv, infer.csv and similar extra outputs
::   Fig/
::   __pycache__/  and  *.pyc
::
:: what stays
::   tracked source, docs, bats
::   untracked files that are NOT ignored, e.g. a new .py you have not added
::   Data/ai_mlp/config.json  scaler.json  history.json
::   Data/ai_mlp/best.pt  last.pt   and other non-epoch *.pt
::   Data/ai_mlp/ckpts/latest.json
::
:: dry-run without deleting
::   git clean -ndX
::
:: notes
::   this is destructive. type Y to confirm, anything else aborts
::   do not use git clean -fdx  -- lowercase x also wipes unignored untracked files
:: =============================================================================

echo git clean -fdX  -- delete gitignored files only
echo keeps source, unignored untracked files, and the ai_mlp keep-list
set /p ans=Type Y to continue: 
if /I not "%ans%"=="Y" (
  echo aborted
  pause
  exit /b 1
)

git clean -fdX
pause
