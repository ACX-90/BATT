@echo off
:: =============================================================================
:: Script\gen_long.bat  -  hour-scale traces: Data/long/*.csv (task F)
:: =============================================================================
::
:: usage
::   double-click this file, or from Script\:
::     gen_long.bat
::   python path is .\..\Src\... so cwd must be Script\
::
:: what it does
::   hour-scale negative traces for gate / "do not split R"
::   does NOT edit SEQUENCE in nmc100ah_ecm_gen.py
::   long profiles go in via run_sim(..., sequence=...)
::   not the training grid; not Replay old-set
::
:: command actually run  -- edit the python line below
::   python .\..\Src\Sim\nmc100ah_ecm_gen_long.py --only loop
::
:: this file writes only the loop case. drop --only to write all three:
::   cc_rest   0.3C discharge 2 h + rest 30 min; SOC0=0.70
::   chg_park  1C charge 40 min + park 2 h; SOC0=0.30
::   loop      1C chg/rest x10, long rest, 1C dis/rest x10, whole x3; SOC0=0.30
::
:: CLI args  (nmc100ah_ecm_gen_long.py)
::   --only NAME       cc_rest | chg_park | loop  -- this file: loop
::   --no-noise        disable meas noise
::
:: output  -- Data/long/
::   loop.csv          this file
::   cc_rest.csv       kf_neg.bat (full gen, no --only)
::   chg_park.csv      kf_neg.bat
::
:: notes
::   dt / noise / ECM from nmc100ah_ecm_gen.py; only the current profile changes
::   do not finetune these traces (hour-scale BPTT). filter + gate: kf_neg.bat
::   see Doc/04-a §7.5, Src/Sim/readme.md
:: =============================================================================

python .\..\Src\Sim\nmc100ah_ecm_gen_long.py --only loop
pause
