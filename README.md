# TDT4173 – vannkraftprediksjon

Vi skal predikere ON/OFF-sannsynligheter for **14 vannkraftaggregater over
168 timer per case**, altså **2352 outputs per case**. Evalueringsmetrikken er
**micro ROC-AUC**. Den utdelte `kaggle_metric.py` er fasiten for evalueringen
og skal beholdes uendret. Output-rekkefølgen må følge konkurransens format.

## Mappestruktur

```text
data/
  raw/           Uendrede, utdelte datasett
  interim/       Midlertidige data fra databehandling
  processed/     Ferdig prosesserte data og features
src/             Python-kode for databehandling og senere modellarbeid
configs/         Konfigurasjonsfiler for senere eksperimenter
notebooks/       Notebooks for utforsking og analyse
evaluation/      Utdelt kaggle_metric.py og eventuell senere evalueringskode
models/          Lagrede modeller og checkpoints
submissions/     Genererte innsendingsfiler
docs/           Prosjektdokumentasjon og oppgavebeskrivelse
reports/
  figures/       Små rapportfigurer som kan versjoneres
```

## Hvor skal prosjektfilene legges?

Ingen prosjektfiler eller datasett fantes lokalt da strukturen ble opprettet.
Legg utdelte trenings-/testdata og en eventuell eksempelinnsending i
`data/raw/`. Legg den utdelte **`kaggle_metric.py` i `evaluation/` uten å endre
innholdet**, og oppgavebeskrivelsen i `docs/`. Eventuelle startnotebooks legges
i `notebooks/`, kode i `src/` og konfigurasjonsfiler i `configs/`.
Genererte innsendingsfiler hører hjemme i `submissions/`.

Data, lagrede modeller, submissions, virtuelle miljøer og notebook-checkpoints
ignoreres av Git. Kode, configs, notebooks, dokumentasjon og små rapportfigurer
kan versjoneres. Hold datasett i `data/`, og fjern store datautskrifter fra
notebooks før de legges i Git. `.gitkeep` bevarer ellers tomme mapper.

Dette er kun prosjektstrukturen; ingen modeller er implementert eller trent.
