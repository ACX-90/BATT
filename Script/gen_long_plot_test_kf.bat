@echo off

rem skip if already generated csv
rem python .\..\Src\Sim\nmc100ah_ecm_gen_long.py --only loop

python ./../Src/Plot/plot_sim_waveforms.py --csv Data/long/loop.csv --show

python ./../Src/AI/KF/run.py --soc-error 0.1 --current-bias 1 --show --csv Data/long/loop.csv --export Data/ai_kf/logs/long.csv

pause