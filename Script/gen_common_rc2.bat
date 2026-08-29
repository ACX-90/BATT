@echo off
:: =============================================================================
:: Script\gen_common.bat  -  one reference trajectory: Data/nmc100ah_ecm_sim.csv
:: =============================================================================
::
:: usage
::   double-click this file, or from repo root:
::     Script\gen_common.bat
::   working directory is forced to the repo root
::
:: what it does
::   time-domain sim using SEQUENCE in nmc100ah_gen.py
::   for test.bat and plots. training grid: use gen_grid.bat
::
:: command actually run  -- edit the python line below
::   python Src/Sim/nmc100ah_gen.py
::
:: CLI args
::   --out PATH        output CSV  -- default Data/nmc100ah_ecm_sim.csv
::   --seed N          override NOISE_SEED in the py header
::   --no-noise        disable meas noise; true columns stay clean
::
:: profile is NOT in this bat. edit Src/Sim/nmc100ah_gen.py header:
::   SOC0 / T_AMBIENT_C / U_P0    initial SOC, temp, polarization
::   DT_S                         step, default 0.1 s
::   ENABLE_CUTOFF                rest for rest of cmd if V/SOC hits limit
::   NOISE_ENABLE / NOISE_SEED / NOISE_STD
::   SEQUENCE                     charge / discharge / rest list
::     mode          charge or discharge or rest
::     duration_s    seconds, or duration_steps, not both
::     c_rate        1.0 = 100 A, or current_a, not both
::
:: output
::   Data/nmc100ah_ecm_sim.csv
::   leading # metadata, then header. pandas: comment="#"
::
:: notes
::   discharge current positive, charge negative
::   defaults are a 100 Ah NMC template, not a commercial cell
::   grid waveforms: gen_grid.bat, same SEQUENCE and noise
::   see Src/Sim/readme.md
:: =============================================================================

python ./../Src/Sim/nmc100ah_gen.py --rc2
pause
