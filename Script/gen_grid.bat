@echo off
:: =============================================================================
:: Script\gen_grid.bat  -  SOC x temperature grid for training waveforms
:: =============================================================================
::
:: usage
::   double-click this file, or from repo root:
::     Script\gen_grid.bat
::   working directory is forced to the repo root
::
:: what it does
::   same SEQUENCE and noise as nmc100ah_gen.py
::   only start SOC and temperature change per file
::   writes Data/grid/*.csv for train_100.bat / train_1000_resume.bat
::
:: command actually run  -- edit the python line below
::   python .\Src\Sim\nmc100ah_gen_grid.py --n-soc 10 --n-temp 10
::
:: CLI args
::   --n-soc N         start-SOC bins  -- this file: 10; py default 5
::   --n-temp N        temperature bins  -- this file: 10; py default 5
::   --out-dir PATH    output dir  -- default Data/grid
::   --seed N          base noise seed; each case offsets by i*100+j
::   --no-noise        disable meas noise
::   --noise-voltage V voltage meas std / V   (task D: 0.007)
::   --noise-current A current meas std / A   (task D: 0.1)
::   --noise-temp C    temperature meas std / C (task D: 0.5)
::   --noise-soc X     SOC meas std, 1 = 100 pp (task D: 0.005)
::                     raise noise to a NEW directory; do not rewrite Data/grid
::   --dry-run         print grid only; no delete, no sim
::   --pybamm          PyBaMM SPM backend (Chen2020 scaled to 100Ah);
::                     same CSV columns as ECM. use a NEW --out-dir
::                     e.g. Data/grid_pybamm ; do not rewrite Data/grid
::   --thermal         with --pybamm only: lumped thermal (default isothermal)
::
:: scan range: Src/Sim/nmc100ah_gen_grid.py header
::   SOC_MIN / SOC_MAX     default 0.10 to 0.90, inclusive; n=1 uses midpoint
::   T_MIN_C / T_MAX_C     default -10 to 50 C
::   SOC_VALUES / T_VALUES_C
::                         if set as lists, those replace MIN/MAX linspace
::   OUTPUT_DIR / FILE_NAME
::
:: SEQUENCE, noise, dt
::   imported from nmc100ah_gen.py; edit once, both stay in sync
::
:: output  -- default Data/grid
::   nmc100ah_ecm_s{ii}_t{jj}_socXXX_T+YY.csv
::   index.csv   bin, init, end SOC/V, cutoff flag, path
::
:: notes
::   a real run deletes existing *.csv in the out dir, including index.csv
::   so leftover files from a different grid size do not enter training
::   --dry-run does not delete or write
::   this file is 10x10 = 100 cases; set both 10 to 5 for 5x5
::   low SOC + low T may trip cutoff; that is protection, not a crash
::   see Src/Sim/readme.md
:: =============================================================================

python .\..\Src\Sim\nmc100ah_gen_grid.py --n-soc 10 --n-temp 10
pause
