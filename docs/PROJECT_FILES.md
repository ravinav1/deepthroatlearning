# Importerte prosjektfiler

15 prosjektfiler ble hentet fra `project materials.zip`. Alle er kontrollert
med SHA-256 mot ZIP-innholdet og er byte for byte uendret. Originalarkivet er
også uendret. Ingen eksisterende filer ble overskrevet. Det ytre
`sintef_project/`-nivået ble fjernet; datagruppene `kernel/` og `extended/`
er beholdt. `.DS_Store` og `__pycache__` ble ikke kopiert inn.

| Plassering | Innhold |
| --- | --- |
| `data/raw/kernel/` | Fem CSV-er: historiske priser, volum, tilsig, unit commitment og syntetiske vannverdier |
| `data/raw/extended/` | `Constraint_min_volume.csv` og `Constraint_min_flow.csv` |
| `data/raw/` | `prediction_mapping.csv` og den utdelte `sample_submission.csv` |
| `configs/Tokke_Vinje_topology.yaml` | Utdelt topologikonfigurasjon |
| `docs/` | `ML_Task_TDT4173.pdf`, `Dataset_definitions_and_explanation.pdf` og `Tokke_topology.pdf` |
| `notebooks/schedule_visualization.ipynb` | Utdelt visualiseringsnotebook |
| `evaluation/kaggle_metric.py` | Utdelt evalueringsmetrikkskript, uendret |

Eksempelinnsendingen er råmateriale i `data/raw/`; senere genererte
innsendingsfiler skal legges i `submissions/`. Eksempelinnsendingen inneholder
725 rader og 2353 kolonner: `Run No` og 2352 prediksjoner.
`prediction_mapping.csv` inneholder 725 rader med `Run No` og `starttime`.

## Notebookens filreferanser

Notebooken er ikke kjørt eller endret. Den refererer til
`data/kernel/Generator_schedules_2015_2023.csv`, som **ikke finnes i ZIP-en**,
og `data/kernel/Historical_day_ahead_price_2015_2025.csv`, som nå ligger i
`data/raw/kernel/`. Fra notebookmappen vil den sistnevnte relative stien være
`../data/raw/kernel/Historical_day_ahead_price_2015_2025.csv`.

ZIP-en inneholder `Unit_commitment_decisions.csv`; det er ikke kontrollert
om denne kan erstatte den manglende schedule-filen. Referansene må avklares
før notebooken kjøres. Ved den første importen var ingen modeller implementert
eller trent. For dagens modellstatus og oppstart, se [README](../README.md).
