# Feilanalyse av 75 % XGBoost + 25 % LightGBM

Analysen bruker kun lagrede valideringsprediksjoner for 365 cases i hvert av årene 2021 og 2022. Konkurransens testdata og leaderboard-labels inngår ikke. Ingen modell er trent i denne analysen.

## Hovedfunn

- Lio og Haukeli har lavest AUC i begge år. De er 2 av 14 aggregater, men står for omtrent 24–25 % av feilene ved sannsynlighetsterskel 0,5.
- Vinje-aggregatene er også relativt svake, særlig i 2022.
- Feil er konsentrert rundt ON/OFF-skifter: omtrent 3,3–3,6 ganger høyere feilrate enn ellers.
- Siste dag i uka har høyere AUC enn første dag i begge år. Resultatene støtter ikke at en lang horisont er hovedproblemet.

## AUC per aggregat

| Aggregat | 2021 | 2022 |
|---|---:|---:|
| Lio_G1 | 0.90105 | 0.92478 |
| Haukeli_G1 | 0.92522 | 0.91934 |
| Vinje_G3 | 0.95064 | 0.93823 |
| Vinje_G2 | 0.95625 | 0.94568 |
| Vinje_G1 | 0.95551 | 0.95010 |
| Tokke_G4 | 0.94373 | 0.97284 |
| Tokke_G2 | 0.95049 | 0.97361 |
| Hogga_G1 | 0.96030 | 0.96564 |
| Tokke_G1 | 0.94957 | 0.97922 |
| Tokke_G3 | 0.95463 | 0.98024 |
| Kjela_G1 | 0.98296 | 0.97749 |
| Vesle Kjela_G1 | 0.99908 | 0.98712 |
| Songa_G1 | 0.99417 | 0.99246 |
| Byrte_G1 | 0.99527 | 0.99790 |

## Skifter

Et skifte ved time t betyr at fasit ved t er forskjellig fra t−1. Nær et skifte betyr innenfor ±2 timer, avgrenset til casets 168 timer. Vi utleder ingen starttilstand før time 0. Denne masken bruker fasit kun til analyse og kan ikke brukes som input ved prediksjon.

| År | Timer nær skifter | Feilrate nær | Feilrate ellers | Andel feil nær skifter |
|---|---:|---:|---:|---:|
| 2021 | 8.0% | 27.8% | 8.3% | 22.6% |
| 2022 | 11.9% | 24.4% | 6.7% | 32.8% |

## Neste forsøk

Prøv først en liten, valgfri gruppe prisfeatures i tre-modellene: prisendring fra forrige time, forskjell til neste time og lokale prisgjennomsnitt for 6/12/24 timer før og etter timen. Beregn dem innenfor den utdelte uka, med avgrensede vinduer ved ukegrensene. Hypotesen er at prisnivået i et sammenhengende tidsrom er mer informativt for start/stopp enn prisen i én time. Dette er ikke bevist av feilanalyse alene.

YAML-filen oppgir p_min, p_max, startcost_const og stopcost_const for aggregatene; disse brukes ikke eksplisitt i dagens features. Et senere forsøk kan kombinere dem med prisvarighet og vannverdi for en grov økonomisk referanse. Faste aggregatparametre alene kan allerede være representert indirekte gjennom aggregat-ID. Det er ikke en eksakt lønnsomhetsgrense: effekt, fallhøyde, virkningsgrad og vannverdi i kaskaden påvirker også beslutningen.

Sammenlign ett slikt featureforsøk om gangen på begge valideringsår, med samme kronologiske splitt og samme eksisterende modeller som referanse.

## Tolkning og filer

- Konkurransemålet er micro ROC-AUC over alle outputs. AUC per aggregat eller dag er diagnostikk og kan ikke gjennomsnittberegnes til den offisielle totalscoren.
- Terskel 0,5 brukes kun for å beskrive feil; en annen terskel eller kalibrering kan endre feilraten uten å forbedre AUC.
- Nærliggende cases overlapper. Disse tallene er beskrivende utviklingsresultater, ikke uavhengige observasjoner eller en ny urørt test.
- Figur: `reports/figures/error_analysis.png`.
- Gruppe- og timemetrikker: `reports/error_analysis_groups.csv` og `reports/error_analysis_hours.csv`.
- De ti mest selvsikre feilene i hvert år: `reports/error_analysis_examples.csv`. Eksemplene er ikke deduplisert etter kalender-time, siden labels er case-spesifikke.
- Reproduser med `python src/analyze_errors.py`.
