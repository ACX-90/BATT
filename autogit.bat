@echo off
:: =============================================================================
:: autogit.bat  -  pull, commit, push this repo
:: =============================================================================
::
:: usage
::   double-click in repo root, or:
::     autogit.bat
::
:: commands actually run  -- keep this order
::   git pull
::   git add *
::   git commit -m "auto update"
::   git push
::
:: steps
::   git pull              fetch remote first to cut push conflicts
::   git add *             stage visible files; dotfiles are skipped
::   git commit -m "..."   commit; edit the quoted message on that line
::   git push              push the tracked remote branch
::
:: notes
::   if nothing changed, commit fails and push is skipped. that is ok
::   git add * does not add dotfiles or dot-dirs
::   Data/ and Fig/ are gitignored; sim CSV and plots usually stay local
::   commit message is hard-coded "auto update"
:: =============================================================================

git fetch
git pull
git add *
git commit -m "auto update"
git push
pause
