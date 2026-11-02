# Earlier experiment notes

Archived from the previous README on 2026-11-02. These notes preserve the earlier experiment history; later rounds and the current best are summarized in [HANDOFF.md](HANDOFF.md). Run command examples from the repository root. Historical commands may need new output names and updated deadlines; use [REPRODUCING.md](REPRODUCING.md) to start.

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

De utdelte filene er plassert i strukturen; se
[docs/PROJECT_FILES.md](../docs/PROJECT_FILES.md) for oversikten. Nye gruppemedlemmer
må hente dataene fra prosjekt-ZIP-en og legge CSV-ene under `data/raw/` med
samme plasseringer. Dataene følger ikke med når man kloner repoet.
Den utdelte **`kaggle_metric.py` i `evaluation/` skal beholdes uendret**.

Data, lagrede modeller, submissions, virtuelle miljøer og notebook-checkpoints
ignoreres av Git. Kode, configs, notebooks, dokumentasjon og små rapportfigurer
kan versjoneres. Hold datasett i `data/`, og fjern store datautskrifter fra
notebooks før de legges i Git. `.gitkeep` bevarer ellers tomme mapper.

## Kjør første baseline

Bruk Python 3.12 fra prosjektroten:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-model.txt
python src/train_baseline.py
```

På macOS kan LightGBM også trenge `brew install libomp`; se
[installasjonsveiledningen](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#macos).

- `src/features.py` leser kernel-data og bygger én rad per aggregat, case og time.
- `src/train_baseline.py` sammenligner aggregatenes historiske ON-andel med én
  felles LightGBM-modell. Begge evalueres med den utdelte micro ROC-AUC-metrikken.
- `configs/baseline.json` bestemmer valideringsdato og modellinnstillinger.

Validering bruker casene som starter i 2022. Trening bruker 2015–2021, med
26.–31. desember 2021 fjernet fordi disse casene overlapper valideringsukene.
Siste treningscase starter 25. desember og slutter ved grensen 1. januar.
2022 brukes også til early stopping, så scoren er en utviklingsscore.

Features inkluderer aggregat, tidspunkt, priser og prisreferanser fra hele
uken, startvolum, fyllingsgrad, terminal vannverdi og naturlig tilsig til
aggregatets nærmeste magasiner. Terminal vannverdi hentes ved start + 168 timer.
For Songa summeres magasinvolum og tilsig, og vannverdiene gjennomsnittberegnes.
Dette er enkle modellfeatures; tilsigssummen er ikke en full vannbalanse.

Modellen lagres i `models/lightgbm_baseline.txt`, valideringsprediksjoner i
`data/processed/baseline_validation_predictions.csv` og en liten resultatfil i
`reports/baseline_metrics.json`. En ny kjøring erstatter disse genererte filene.
## Valgfrie restriksjonsfeatures

`use_extended` i configen slår extended-data på eller av. `baseline.json` har
verdien `false`; `extended.json` har `true` med ellers samme modellinnstillinger.

```bash
python src/train_baseline.py --config configs/extended.json
```

Dette legger til timeverdier og ukesmaksimum for de tre minimumsvolumene og
seks minimumsvannføringene. For minimumsvolumene brukes også startvolum minus
hver times minimumskrav. Det er en referanse til starttilstanden, ikke en
predikert magasinbane. Kravene brukes som modellinputs og tvinger ikke
aggregater ON/OFF. Når bryteren er av, leses ikke extended-filene.
Resultatene lagres med navnet `extended`, separat fra baselinen.

`use_system_context` legger også til startvolum, terminal vannverdi og totalt
naturlig tilsig gjennom uken for alle 17 magasiner, som inputs til hvert aggregat.
Bryteren er av når den ikke er oppgitt. Dette bruker ikke senere observerte
magasinvolumer eller andre aggregaters fasitbeslutninger.

```bash
python src/train_baseline.py --config configs/system_context.json
```

Denne configen har begge bryterne på og ellers samme modellinnstillinger som
`extended.json`. Resultatene lagres separat med navnet `system_context`.

## Valider på et annet år

```bash
python src/train_baseline.py --config configs/validation_2021.json
```

Denne kjøringen trener på cases fra 2015–2020 og validerer bare på cases som
starter i 2021. De seks overlappende casene før 2021 utelates fra trening.
`validation_end` er en eksklusiv grense, så cases som starter i 2022 ikke brukes
i denne kjøringen. Modell og resultater får navnet `validation_2021`.

## Sammenlign lossfunksjoner

`loss` velger `binary` (vanlig binær log loss, standard) eller `focal`.
Focal loss reduserer vekten til enkle eksempler med `focal_gamma`, her 2.
Dette forsøket bruker ingen ekstra klassevekter. Se
[originalartikkelen](https://arxiv.org/abs/1708.02002) for lossfunksjonen.

```bash
python src/train_baseline.py --config configs/focal_2021.json
python src/train_baseline.py --config configs/focal_2022.json
```

Features, modellinnstillinger og datasplitt er de samme som i BCE-kjøringene.
Focal-modellene lagres separat. `src/losses.py` beregner gradient og Hessian;
Hessian får et positivt gulv siden focal loss kan ha negativ krumning.
Egendefinert loss gir rå modellscore, så skriptet bruker sigmoid før det lagrer
sannsynligheter. Ved direkte lasting av en focal-modell må man også bruke
`scipy.special.expit(model.predict(features, raw_score=True))`.

## Valgfrie topologifeatures

`use_topology` slår på ni ekstra features per kraftverk, laget fra forbindelsene
i den utdelte YAML-filen. Oppstrøms brukes alle unike magasiner som kan nå
kraftverket via elver, tunneler og kraftverk. Nedstrøms brukes første mottakende
magasin langs hver utløpsvei; for eksempel finner også Lio Bandak via elva.

Features er oppstrøms startvolum, samlet fyllingsgrad, naturlig uketilsig og
gjennomsnittlig terminal vannverdi; nedstrøms fyllingsgrad, ledig startkapasitet
og gjennomsnittlig vannverdi; forskjellen mellom vannverdiene og en indikator
for om et nedstrøms magasin finnes. Uten mottakende magasin er de nedstrøms
verdiene null. Vannverdiene er enkle aritmetiske snitt. Summene beskriver
magasintilstand langs mulige vannveier, ikke faktisk tapping eller produksjon.

```bash
python src/train_baseline.py --config configs/topology_2021.json
python src/train_baseline.py --config configs/topology_2022.json
```

Begge bruker vanlig BCE og samme modellinnstillinger som tidligere. Bryteren
er av når den mangler; tidligere modeller og resultater beholdes uendret.

## Lag en submission

```bash
python src/make_submission.py
```

`configs/submission_topology.json` bruker vanlig BCE, alle featurebryterne på
og 674 trær valgt fra 2022-valideringen. Modellen trenes på alle 2922 utdelte
treningscases fra 2015–2022. Test-casene fra `prediction_mapping.csv` brukes
bare til prediksjon; det brukes ingen testlabels eller early stopping på test.

CSV-en lagres som `submissions/submission_topology.csv`, med samme ID-er og
kolonnerekkefølge som eksempelinnsendingen. Last den opp på Kaggle for å få
leaderboard-score. Sluttmodellen og kjøringsinformasjon lagres separat som
`models/lightgbm_submission_topology.txt` og `reports/submission_topology_run.json`.
For å bevare en eksisterende submission må en ny kjøring bruke et nytt filnavn,
for eksempel `python src/make_submission.py --output submissions/submission_v2.csv`.

En større modell kan kjøres med:

```bash
python src/train_baseline.py --config configs/larger.json
```

Den bruker opptil 2500 trær med 127 blader, åtte CPU-tråder og samme datasplitt.
Early stopping beholder modellen med best valideringsscore. Filer fra denne
kjøringen får navnet `larger`, slik at den første baselinen beholdes.

## Prøv XGBoost med samme features

```bash
python src/train_xgboost.py --config configs/xgboost_2021.json
python src/train_xgboost.py --config configs/xgboost_2022.json
```

Disse bruker de samme 98 featurene og kronologiske splittene som
`topology_2021.json` og `topology_2022.json`, med seks overlappende cases
fjernet før validering. Aggregat-ID er kategorisk. XGBoost bruker histogramtrær,
vanlig binær log loss og early stopping på AUC. Test-cases brukes ikke.
Modellinnstillinger er egne XGBoost-innstillinger, ikke direkte oversatte
LightGBM-parametre.

Modeller lagres som `models/xgboost_2021.ubj` og `models/xgboost_2022.ubj`;
resultater ligger i `reports/xgboost_2021_metrics.json` og
`reports/xgboost_2022_metrics.json`. `best_iteration` i rapportene er antall
beholdte trær (XGBoosts nullbaserte indeks + 1). LightGBM-filene beholdes.

Lag en XGBoost-submission med 994 trær valgt fra 2022-valideringen:

```bash
python src/make_submission.py --config configs/submission_xgboost.json --output submissions/submission_xgboost.csv
```

Dette trener på alle utdelte treningscases med de samme 98 featurene.
Test-cases brukes bare til prediksjon. Modellen lagres som
`models/xgboost_submission_xgboost.ubj`, og kjøringsinformasjonen som
`reports/submission_xgboost_run.json`. Den tidligere LightGBM-submissionen beholdes.

## Prøv CatBoost

```bash
python src/train_catboost.py --config configs/catboost_2021.json
python src/train_catboost.py --config configs/catboost_2022.json
```

Dette bruker de samme 98 featurene og splittene som XGBoost og LightGBMs
 topologimodeller. Aggregat-ID brukes som kategori, med one-hot-behandling
for de 14 aggregatene. Modellen bruker vanlig binær log loss og early stopping
på validerings-AUC. Test-cases holdes utenfor. Modeller lagres separat som
`models/catboost_2021.cbm` og `models/catboost_2022.cbm`; rapportene ligger i
`reports/catboost_2021_metrics.json` og `reports/catboost_2022_metrics.json`.
`best_iteration` i rapporten er antall beholdte trær.

Lag en CatBoost-submission med 1084 trær valgt fra 2022-valideringen:

```bash
python src/make_submission.py --config configs/submission_catboost.json --output submissions/submission_catboost.csv
```

Den bruker alle utdelte treningscases og samme 98 features. Test-cases brukes
bare til prediksjon. Modell og kjøringsinformasjon lagres separat som
`models/catboost_submission_catboost.cbm` og `reports/submission_catboost_run.json`.

## Bland XGBoost og LightGBM

```bash
python src/blend_submissions.py
```

Dette blander eksisterende `submission_xgboost.csv` (75 %) og
`submission_topology.csv` (25 %) uten ny trening. ID-er og kolonner kontrolleres
mot eksempelinnsendingen. Resultatet lagres i
`submissions/submission_xgb_lgb_75_25.csv`; kilder og vekter dokumenteres i
`reports/submission_xgb_lgb_75_25_run.json`. Eksisterende CSV-er beholdes.

## Temporal CNN

```bash
python src/train_cnn.py --config configs/cnn_2022.json
```

En liten temporal CNN predikerer alle 14 aggregater over 168 timer samtidig.
Den bruker samme inputinformasjon som topologimodellene: systemfeatures lagres
én gang, mens lokale features legges i egne kanaler for hvert aggregat.
Dette gir 305 inputkanaler. Aggregat-ID trengs ikke som feature siden hvert
aggregat har en fast plass i input og output.

Sju residualblokker med symmetriske, dilaterte konvolusjoner kan bruke hele
uken som kontekst. Dette er en CNN, ikke en LSTM. Modellen har omtrent
164 000 parametre og bruker vanlig BCE. Input skaleres med middelverdi og
standardavvik beregnet kun fra trening. Samme 2022-splitt og seks utelatte
cases brukes som før; validering velger beste epoch. Test-cases brukes ikke.

Configen setter maks 40 epochs og early stopping etter åtte epochs uten bedre
AUC. MPS brukes hvis tilgjengelig, ellers CPU med fire tråder. Modellen lagres
med skalering og feature-rekkefølge i `models/cnn_2022.pt`, prediksjoner i
`data/processed/cnn_2022_validation_predictions.csv` og historikk/resultater i
`reports/cnn_2022_metrics.json`.

## Bidireksjonal LSTM

```bash
python src/train_bilstm.py --config configs/bilstm_2022.json
```

Denne modellen leser hver 168-timersuke både forover og bakover. To LSTM-lag
med 64 skjulte verdier per retning kombineres til 14 logits per time.
Den bruker samme 305 inputkanaler, train-only-skalering, BCE og 2022-splitt
som CNN-forsøket. Det brukes ingen tidligere fasitbeslutninger som input.
`src/train_bilstm.py` gjenbruker treningsløkken i `src/train_cnn.py`.

Modell, skalering og kanalrekkefølge lagres i `models/bilstm_2022.pt`,
valideringsprediksjoner i `data/processed/bilstm_2022_validation_predictions.csv`
og resultat/historikk i `reports/bilstm_2022_metrics.json`.

## Feilanalyse av blandingen

```bash
python src/analyze_errors.py
```

Dette analyserer lagrede 75/25-valideringsprediksjoner for 2021 og 2022,
per aggregat, dag/time i uka og nær fasitens ON/OFF-skifter (±2 timer).
AUC er hovedmetrikken; feilrate ved terskel 0,5, Brier og log loss brukes
som diagnostikk. Skiftemasken er kun for analyse, aldri en modellfeature.
Resultater og en liten figur lagres i `reports/error_analysis*` og
`reports/figures/error_analysis.png`. Se `reports/error_analysis.md` for
funnene fra denne kjøringen. Ingen trening eller testdata brukes.

## Valgfri lokal priskontekst

`use_price_context` legger til 14 prisfeatures: endring fra forrige time og
til neste time, gjennomsnitt over foregående og kommende 6/12/24 timer,
og dagens timepris minus hvert gjennomsnitt. Vinduene ekskluderer den aktuelle
timen og bruker bare tilgjengelige timer innenfor casets uke. Ved første/siste
time, der ingen naboer finnes i én retning, brukes aktuell pris som gjennomsnitt
og prisendringen er null. Bryteren er av når den mangler.

```bash
python src/train_xgboost.py --config configs/xgboost_price_2021.json
python src/train_xgboost.py --config configs/xgboost_price_2022.json
```

Disse bruker 112 features, med ellers samme modellinnstillinger og splittene
fra de opprinnelige XGBoost-kjøringene. Modeller og valideringsresultater får
egne navn. Bryteren støttes også av LightGBM, CatBoost og submission-skriptet.
Prisene senere i den utdelte uka er inputs i denne planleggingsoppgaven;
vinduene bruker ingen labels eller fremtidige observerte magasinvolumer.

Første priskontekstforsøk ga XGBoost-AUC 0,965719 i 2021 og 0,971633 i
2022, mot henholdsvis 0,964724 og 0,971527 uten disse featurene. Med
75 % av den nye XGBoost-modellen og 25 % av den eksisterende LightGBM-modellen
ble lokal AUC 0,966169 og 0,971874. Sammenligningen, inkludert diagnostikk
rundt ON/OFF-skifter, ligger i `reports/price_context_comparison.json`.
Dette er utviklingsresultater på de samme valideringsårene, ikke Kaggle-scorer.

Lag submissions med lokal priskontekst:

```bash
python src/make_submission.py --config configs/submission_xgboost_price.json --output submissions/submission_xgboost_price.csv
python src/blend_submissions.py --xgboost submissions/submission_xgboost_price.csv --output submissions/submission_xgb_price_lgb_75_25.csv
```

Den nye XGBoost-modellen bruker 749 trær valgt fra 2022-valideringen og alle
2922 utdelte treningscases. Den andre filen blander 75 % av de nye
XGBoost-prediksjonene med 25 % av den eksisterende LightGBM-submissionen.
LightGBM-delen bruker fortsatt de opprinnelige 98 featurene. Test-cases brukes
bare til prediksjon, og tidligere submissions beholdes.

## Vannkontekst og økonomiske referanser

To valgfrie brytere testes separat fra XGBoost med priskontekst:

- `use_water_context`: 15 features for naturlig tilsig i neste 6/12/24 timer,
  tilsig relativt til magasinkapasitet, startens ledige kapasitet minus tilsig,
  og oppstrøms time-, kumulativt og gjenstående naturlig tilsig. Oppstrøms
  inkluderer alle unike magasiner som kan nå kraftverket, inkludert nærmeste
  magasin. Dette er ikke prognoser for faktisk vannoverføring eller volum.
- `use_economic_context`: åtte features for start+stopp-kostnad per referanse-MW
  og hypotetisk netto margin ved kontinuerlig drift i neste 6/12/24 timer på
  `p_min` eller `p_nom`. Marginen er referanse-MW ganger summen av
  `(pris - lokal terminal vannverdi)` per time, minus start+stopp-kostnaden.
  Enheten er EUR. Dette er ingen eksakt lønnsomhetsgrense: faktisk produksjon,
  fallhøyde, virkningsgrad, kaskadeeffekter og starttilstand er ikke beregnet.

Alle vinduer inkluderer aktuell time og avgrenses til casets uke. Bryterne er
av når de mangler. De nye featurene bruker bare utdelte inputs og initialvolum.

```bash
python src/train_xgboost.py --config configs/xgboost_water_2022.json
python src/train_xgboost.py --config configs/xgboost_water_2021.json
python src/train_xgboost.py --config configs/xgboost_economics_2022.json
python src/train_xgboost.py --config configs/xgboost_economics_2021.json
python src/train_baseline.py --config configs/lightgbm_price_2022.json
python src/train_baseline.py --config configs/lightgbm_price_2021.json
```

Vannforsøket har 127 features, økonomiforsøket 120, og LightGBM-prisforsøket
112. Vann og økonomi er separate eksperimenter med samme XGBoost-parametre
som priskontekst-baselinen. LightGBM bruker samme parametre som opprinnelige
 topologimodeller, med priskontekst slått på. Alle kjøringer har egne filnavn.

Sammenlign de lagrede eksperimentene med
`python src/compare_operating_experiments.py`. Resultater ligger i
`reports/operating_feature_comparison.json`; funn og beslutning i
`reports/operating_feature_experiments.md`. Ingen av de tre endringene
forbedret samlet AUC på begge valideringsår, så eksisterende blend beholdes.

## Audit av prosjektdata og modelloppsett

```bash
python src/audit_project.py
```

Auditen kontrollerer inputdekning, generator/time-rekkefølge, metrikken og
uendrede originalfiler når ZIP-en finnes på den registrerte stien. Den
undersøker også konkrete Lio/Haukeli-feil uten ny trening. Funn ligger i
`reports/project_audit.md`, numeriske kontroller i `reports/project_audit.json`
og eksempler i `reports/project_audit_cases.csv`. En bekreftet utelatelse er
naturlig tilsig til `r_Vest_Vassdraget`, som tidligere manglet som feature.

## Forsøk med elvetilsig

`use_river_context` legger til to features i m³/s: naturlig tilsig til
`r_Vest_Vassdraget` og `max(0, minimumskrav − naturlig tilsig)`.
Bryteren krever `use_extended` og er av som standard. Underskuddet er en
input, ikke en regel som tvinger Lio ON; andre utslipp kan også dekke kravet.

```bash
python src/train_xgboost.py --config configs/xgboost_river_2021.json
python src/train_xgboost.py --config configs/xgboost_river_2022.json
python src/compare_river_experiment.py
```

Forsøkene bruker 114 features og samme parametre og splitter som
XGBoost-priskontekst-modellene. Konkurransens testcaser inngår ikke i trening
eller validering. Resultatene er micro ROC-AUC:

| Modell | 2021 | 2022 |
| --- | --- | --- |
| XGBoost med priskontekst | 0.965719 | 0.971633 |
| XGBoost med elvetilsig | 0.966891 | 0.970898 |
| Eksisterende 75/25-blend | 0.966169 | 0.971874 |
| 75/25-blend med elvetilsig | 0.967090 | 0.971244 |

Lio-AUC i XGBoost øker fra 0.906129 til 0.923175 i 2021, men faller fra
0.918593 til 0.905517 i 2022. Endringen forbedrer ikke begge år, så den
eksisterende submissionen og standardkonfigurasjonene beholdes. Detaljer for
alle generatorer ligger i `reports/river_feature_comparison.json`.

## Antall og kombinasjon av aggregater

`src/train_plant_commitment.py` trener separate modeller for Tokke og Vinje.
Én modell predikerer sannsynligheten for hvert antall aggregater ON; en annen
predikerer aggregatkombinasjoner sett i treningsdata. Kombinasjonene normaliseres
innen hvert antall og vektes med predikert antall-sannsynlighet. Summen av disse
gir ON-sannsynligheten for hvert aggregat. Ingen sann antall-label brukes som
input under prediksjon, og ingen fast startrekkefølge påtvinges.

```bash
python src/train_plant_commitment.py --config configs/plant_commitment_2021.json
python src/train_plant_commitment.py --config configs/plant_commitment_2022.json
```

Forsøket bruker samme priskontekst-inputs og XGBoost-parametre som før, med
111 features per kraftverk (generator-ID fjernes). Early stopping bruker
multiklasse-logloss; sluttresultatet evalueres med offisiell micro ROC-AUC.
De samme tidssplittene og seks utelatte overlappende casene beholdes, og
konkurransens testcaser inngår ikke. Andre kraftverk beholder den eksisterende
75/25-blenden. Modeller, prediksjoner og rapporter får egne filnavn.

| Modell | 2021 | 2022 |
| --- | --- | --- |
| Eksisterende 75/25-blend | 0.966169 | 0.971874 |
| Nye kraftverksmodeller for Tokke/Vinje | 0.967919 | 0.970120 |
| Fast 50/50-blend av disse to | 0.969061 | 0.972153 |

50/50-vekten ble valgt før resultatene forelå. Den forbedrer begge år, men
gevinsten i 2022 er liten. De nye modellene alene forbedrer ikke begge år.
To Vinje-timepunkter i 2022 har kombinasjoner som ikke forekommer i trening;
disse inngår i slutt-AUC, men ikke i kombinasjonsmodellens early-stopping-logloss.
Fullstendige resultater ligger i `reports/plant_commitment_2021_metrics.json`
og `reports/plant_commitment_2022_metrics.json`. Eksisterende submissions
beholdes; forsøket har ikke laget en ny Kaggle-CSV.

## Tilsigprofiler, vannstand og referansebudsjett

Tre uavhengige brytere er av som standard:

- `use_inflow_profiles`: timespesifikt, akkumulert og gjenværende naturlig
  tilsig for hvert av de 17 magasinene. 51 nye features; tidspunkt og
  magasinidentitet beholdes, men dette er ikke en prognose for ankomst nedstrøms.
- `use_hydraulic_context`: startvannstand fra `vol_head`, signert forskjell
  mellom vannstander over magasintunneler og nivå relativt til tunnelåpningene.
  28 nye features; beregner ingen faktiske strømmer eller senere vannstander.
- `use_water_budget`: naturlig tilsig minus kjente, obligatoriske direkte
  utløp, som akkumulert/gjenværende Mm³, referansefylling og margin til
  minimumsvolum. 68 nye features; krever `use_extended`. Produksjon,
  oppstrøms overføringer og spill inngår ikke i dette referansebudsjettet.

Alle tidssummer inkluderer aktuell time og stopper ved casets ukegrense.
Referansevolum er startvolum pluss netto naturlig vann gjennom aktuell time,
ikke et observert eller beregnet faktisk magasinvolum. Negative verdier og
fyllgrad over 1 beholdes som features. Elvekravet for `r_Vest_Vassdraget`
trekkes ikke fra et enkelt magasin, siden flere kilder kan dekke det.

```bash
python src/train_xgboost.py --config configs/xgboost_profiles_2021.json
python src/train_xgboost.py --config configs/xgboost_profiles_2022.json
python src/train_xgboost.py --config configs/xgboost_hydraulic_2021.json
python src/train_xgboost.py --config configs/xgboost_hydraulic_2022.json
python src/train_xgboost.py --config configs/xgboost_budget_2021.json
python src/train_xgboost.py --config configs/xgboost_budget_2022.json
python src/train_xgboost.py --config configs/xgboost_price_2020.json
python src/train_xgboost.py --config configs/xgboost_hydraulic_2020.json
python src/compare_hydrology_experiments.py
```

Forsøkene bruker samme parametre og tidsdeling som pris-XGBoost og egne
filnavn. Sammenligningen viser både XGBoost alene og 75/25 med eksisterende
LightGBM, samlet og per aggregat, i `reports/hydrology_feature_comparison.json`.
Den ekstra 2020-backtesten sammenligner kun pris-XGBoost og hydraulikk.
Bryterne er også tilgjengelige i øvrige trenere og submission-skriptet.

Resultater med XGBoost alene, med uendrede modellparametre:

| Features | 2021 | 2022 |
| --- | --- | --- |
| Eksisterende priskontekst | 0.965719 | 0.971633 |
| + Tilsigprofiler | 0.965813 | 0.971371 |
| + Startvannstand og tunneler | 0.965917 | 0.971828 |
| + Referansebudsjett | 0.964845 | 0.970740 |

Tilsigprofiler gir blandede resultater, og referansebudsjettet er svakere
begge år. Hydraulikk gir omtrent 0.0002 høyere samlet AUC begge år, men
Lio og Haukeli blir svakere i 2021. Med samme 75/25-blend blir hydraulikk
0.966369 / 0.971983; den tidligere kraftverksblenden på 50/50 er fortsatt
bedre samlet med 0.969061 / 0.972153. Eksisterende konfigurasjoner og
submissions beholdes, og de tre nye bryterne forblir av som standard.

Byttes bare XGBoost-delen i den faste 50/50-kraftverksblenden til hydraulikk,
øker samlet AUC litt begge år:

| Kraftverksblend | 2021 | 2022 |
| --- | --- | --- |
| Eksisterende | 0.969061 | 0.972153 |
| Med hydraulikk-XGBoost | 0.969162 | 0.972407 |

Formelen er `0.5 * (0.75 * hydraulic_XGB + 0.25 * LightGBM) +
0.5 * plant_predictions`. Vektene er de samme som før; kraftverksmodellene
er ikke trent på nytt med hydraulikk-inputs. Dette forbedrer samlet AUC,
men beholder avveiningen med svakere Lio/Haukeli i 2021.

Den ekstra backtesten på 2020 bekrefter ikke hydraulikkforbedringen:
prisbaseline får 0.942120, mens hydraulikk får 0.935524. Begge bruker
1820 tidligere treningscaser, 366 valideringscaser og seks purgede caser.
Med uendrede innstillinger stopper hydraulikk etter 179 trær mot 649 for
baselinen. Forsøket gir derfor ikke en stabil forbedring på tvers av de
tre periodene. De nye featuregruppene beholdes som valgfrie eksperimenter;
ingen aktiveres i eksisterende konfigurasjoner.

Alle åtte treningskjøringene ekskluderer konkurransens testcaser. Lagrede
modeller er kontrollert mot valideringsprediksjonene, og alle 15 utdelte
filer, inkludert `kaggle_metric.py`, er fortsatt byte for byte lik ZIP-en.

## Generalisering: parametre, vekting og ensemble

Ni nye XGBoost-kjøringer tester modellinnstillinger med de samme 112
prisfeatures: tre parameterprofiler på 2021/2022 og nyere års vekting på
2020/2021/2022. Kjøringene bruker egne filnavn, fire CPU-tråder og de samme
kronologiske splittene som prisreferansen, med seks caser purget ved grensen.

- `sampling`: 80 % av radene og featurekolonnene per tre.
- `regularized`: dybde 5, `min_child_weight=100`, `reg_lambda=10`.
- `deep_regularized`: dybde 8, samme regularisering og 80 % sampling.
- `recency`: prisreferansens parametre, med fire års halveringstid for
  casevektene. Vektene normaliseres til gjennomsnitt 1. Validering og
  treningsdiagnostikk beregnes uten vekter.

`record_training_auc: true` måler AUC på 256 hele treningscaser valgt med
seed 42, og lagrer case-ID-er. Dette er et treningsutvalg med 602 112
outputs, ikke full trenings-AUC eller et uavhengig valideringssett.
`recency_half_life_years: 4` aktiverer vekting i XGBoost-treneren; uten
feltet har alle treningsrader lik vekt.

```bash
python src/train_xgboost.py --config configs/xgboost_regularized_2022.json
python src/train_xgboost.py --config configs/xgboost_recency_2022.json
python src/compare_generalization_experiments.py
python src/evaluate_generalization_frozen.py
python src/evaluate_group_calibration.py
```

Konfigurasjoner finnes for begge valideringsårene og alle profilene, samt
`xgboost_recency_2020.json` for den ekstra backtesten. Treningskommandoene
lagrer modeller og prediksjoner under konfigurasjonens navn.
Sammenligningsskriptene leser eksisterende resultater.

Resultater med XGBoost alene:

| Oppsett | 2021 | 2022 |
| --- | --- | --- |
| Prisreferanse | 0.965719 | 0.971633 |
| Sampling | 0.964316 | 0.972346 |
| Sterkere regularisering | 0.967331 | 0.969022 |
| Dybde 8 og regularisering | 0.965715 | 0.971548 |
| Nyere års vekting | 0.967307 | 0.971817 |
| Likt gjennomsnitt av alle fem | 0.967138 | 0.973073 |

Ingen av de tre parameterprofilene forbedrer begge årene. Dybde 8 øker
treningsutvalgets AUC til 0.998501 / 0.997649, uten bedre validerings-AUC.
Nyere års vekting forbedrer 2021/2022, men svekker 2020 fra 0.942120 til
0.941082. Den er derfor ikke en stabil forbedring over alle tre periodene.

Det faste ensemblet bruker 20 % fra hver XGBoost-profil og forbedrer
prisreferansen med omtrent 0.0014 begge år. 12/14 aggregater forbedres
i 2021 og 13/14 i 2022; Lio taper litt i 2021. Rapporten inkluderer også
de eksisterende faste LightGBM- og kraftverksblandingene. Med 75/25 mot
LightGBM får ensemblet 0.967138 / 0.972701, mot referansens
0.966169 / 0.971874. Nyere års vekting i den faste kraftverksblenden gir
0.969848 / 0.973066, men denne varianten er ikke backtestet på 2020.

En separat kontroll evaluerer de fem faste 2021-modellene på hele 2022,
uten ny fitting eller tidlig stopping på 2022. Prisreferansen får
0.961388 og ensemblet 0.961662. Gevinsten er dermed mindre når modellen
brukes gjennom et ekstra år. 2022 er allerede brukt til utvikling.

Kalibrering tester én felles positiv logit-slope og 14 regulariserte
aggregat-offsets, med fast `C=0.001`. Koeffisientene læres på 2021 og
evalueres på 2022: prisreferansen går fra 0.971633 til 0.971860, og
ensemblet fra 0.973073 til 0.973395. AUC innen hvert aggregat er uendret;
gevinsten kommer fra rangeringen mellom aggregater. Dette er et
utviklingsforsøk: grunnmodellenes trær er valgt med tidlig stopping på
sine valideringsår, og kalibreringens 2021-score er en in-sample-score.

Resultater, case-ID-er, parametre og AUC per aggregat finnes i
`reports/generalization_experiment_comparison.json`,
`reports/generalization_frozen_comparison.json` og
`reports/generalization_calibration_comparison.json`.
Alle ni nye modellfiler er kontrollert mot konfigurasjon, featurekolonner,
caseutvalg og rapportert beste iterasjon. Konkurransens testcaser er
ekskludert, og tidligere modeller, konfigurasjoner og submissions er bevart.

## XGBoost-ensemble med CNN

En ny CNN-kjøring på 2021 bruker samme oppsett som CNN-en på 2022.
Blandingen er fast: 80 % av gjennomsnittet fra de fem XGBoost-profilene
ovenfor, og 20 % CNN. Vekten ble valgt på 2022 før CNN-kontrollen på 2021.

| Oppsett | 2021 | 2022 |
| --- | --- | --- |
| Fem XGBoost-modeller | 0.967138 | 0.973073 |
| CNN alene | 0.933421 | 0.953495 |
| 80 % XGBoost + 20 % CNN | **0.974018** | **0.977576** |

Blandingen forbedrer AUC innen 13/14 aggregater i 2021 og 10/14 i 2022.
Resultatene og AUC per aggregat finnes i
`reports/sequence_ensemble_comparison.json`. Dette er historisk
utviklingsvalidering med tidlig stopping, ikke en måling av Kaggle-testen.

Sluttmodellene bruker alle 2922 oppgitte treningscaser. De 725 testcasene
fra `prediction_mapping.csv` brukes bare til prediksjon. Sampling,
regularisering, dybde 8 og nyere års vekting bruker henholdsvis 855, 635,
645 og 978 trær, valgt fra 2022-valideringen. Den eksisterende
`submission_xgboost_price.csv` gjenbrukes som den femte modellen.
CNN-en trenes i 29 faste epoker, også valgt fra 2022-valideringen.

```bash
python src/train_cnn.py --config configs/cnn_2021.json
python src/compare_sequence_ensemble.py
python src/make_submission.py --config configs/submission_xgb_sampling.json --output submissions/submission_xgb_sampling.csv
python src/make_submission.py --config configs/submission_xgb_regularized.json --output submissions/submission_xgb_regularized.csv
python src/make_submission.py --config configs/submission_xgb_deep_regularized.json --output submissions/submission_xgb_deep_regularized.csv
python src/make_submission.py --config configs/submission_xgb_recency.json --output submissions/submission_xgb_recency.csv
python src/make_sequence_submission.py
python src/make_ensemble_submission.py --cnn submissions/submission_cnn.csv
```

Submission-skriptene nekter å overskrive eksisterende sluttmodeller,
CSV-er og kjøringsrapporter. De lager tre ensemblefiler:

- `submissions/submission_xgb_cnn_80_20.csv`: den mest lovende blandingen.
- `submissions/submission_xgb_ensemble.csv`: fem XGBoost-modeller med lik vekt.
- `submissions/submission_xgb_ensemble_calibrated.csv`: samme XGBoost-ensemble
  med den tidligere kalibreringen lært på 2021 og kontrollert på 2022.

Kalibreringen bruker én felles positiv logit-slope og ett offset per
aggregat; den er ikke lært på sluttmodellenes treningsprediksjoner eller
testlabels. Modellene og CSV-ene ignoreres av Git, mens configs, kode og
kjøringsrapporter kan versjoneres. Tidligere submissions beholdes.

Brukerrapportert Kaggle-score for `submission_xgb_cnn_80_20.csv` er
**0.98152**, opp fra tidligere beste 0.97898. Scoren er registrert i
`reports/kaggle_scores.json`; den er ikke målt av de lokale skriptene.

## CNN med læringsrateplan

`configs/cnn_scheduler_2021.json` og `configs/cnn_scheduler_2022.json`
tester samme CNN med `ReduceLROnPlateau`: learning rate halveres etter
fire valideringer uten tilstrekkelig AUC-forbedring, ned til minst 0.00001.
Alle øvrige innstillinger er like, inkludert 40 epoker, tidlig stopping
etter åtte epoker uten ny beste AUC, og fire CPU-tråder. Uten
`lr_scheduler` i configen brukes fortsatt en fast learning rate.

```bash
python src/train_cnn.py --config configs/cnn_scheduler_2021.json
python src/train_cnn.py --config configs/cnn_scheduler_2022.json
python src/compare_sequence_ensemble.py --cnn-prefix cnn_scheduler --output reports/sequence_scheduler_comparison.json
```

Learning rate brukt i hver epoke lagres i kjøringsrapportens `history`.
Sammenligningen holder fem XGBoost-modeller og 80/20-vektene fast, og
måler både CNN alene og blandingen mot de tidligere resultatene.

Resultater fra begge scheduler-kjøringene:

| Oppsett | 2021 | 2022 |
| --- | --- | --- |
| CNN med fast learning rate | 0.933421 | 0.953495 |
| CNN med læringsrateplan | 0.929882 | 0.950697 |
| Opprinnelig 80/20-blanding | **0.974018** | **0.977576** |
| 80/20 med læringsrateplan | 0.973898 | 0.977178 |

Denne læringsrateplanen forbedrer verken CNN alene eller blandingen på
noen av årene. Kjøringene stopper etter 18 og 28 epoker, med beste modell
fra epoke 10 og 20. Før første learning rate-endring matcher henholdsvis
åtte og 24 epoker referansens loss og AUC eksakt. Splitt, features og
testekskludering er uendret. Oppsettet beholdes som et valgfritt forsøk;
den innsendte modellen med Kaggle-score 0.98152 beholdes som referanse.

## Kronologisk OOF-stacking

`configs/stacking.json` låser fem XGBoost-profiler til 600 trær og CNN-en
til 20 epoker. Hvert OOF-år bruker grunnmodeller trent på caser som slutter
senest ved årets start. Ingen early stopping eller checkpointvalg bruker
etikettene fra året som predikeres. Eksisterende XGBoost-modeller kan
gjenbrukes ved å ta nøyaktig de første 600 trærne. En eksisterende CNN fra
epoke 20 kan gjenbrukes dersom oppsett og hele treningsforløpet frem til
den epoken matcher det faste oppsettet.

En logistisk regresjon lærer fra OOF-prediksjoner for 2020 og 2021.
Seks logitkolonner standardiseres med statistikk fra meta-treningen, og
14 aggregatindikatorer gir hvert aggregat et biasledd. Modellvektene er
felles for aggregatene. `C=0.001` og øvrige meta-innstillinger er faste.
De seks casene fra 26.–31. desember 2021 purges før meta-fit, slik at
725 caser og 1 705 200 outputs gjenstår. Skalering og meta-modell fryses
før 2022-evalueringen.

```bash
python src/train_stacking_bases.py --year 2020
python src/train_stacking_bases.py --year 2021
python src/train_stacking_bases.py --year 2022
python src/train_stacking.py
```

Treningene kjører sekvensielt med fire CPU-tråder. XGBoost og CNN bruker
separate prosesser for å unngå en lokal konflikt mellom bibliotekene.
Grunnmodellenes prediksjoner ligger i `data/processed/stacking_*`;
bare en eventuell ferdig kombinert submission legges i `submissions`.
Resultatene i `reports/stacking_2022_metrics.json` sammenligner stacking
med 80/20 fra de samme grunnmodellene og den tidligere sterke referansen.
2022 har allerede påvirket tidligere prosjektvalg, så dette er fortsatt
en utviklingsbacktest.

Sluttmodellene og stacking-submissionen lages med:

```bash
python src/train_stacking_bases.py --final
python src/train_stacking.py --final
```

Sluttversjonen refitter samme meta-oppsett på alle 1 096 historiske
OOF-caser gjennom 2022. Grunnmodellene bruker alle oppgitte treningscaser,
og konkurransens testcaser brukes bare til prediksjon. Alle nye modeller,
prediksjonsfiler og rapporter har egne navn; eksisterende filer beholdes.

Resultatet fra den frosne 2022-evalueringen:

| Oppsett | Micro ROC-AUC |
| --- | --- |
| Fast 80/20, samme 600-trærs / 20-epokers grunnmodeller | 0.976999 |
| OOF-stacking fra de samme grunnmodellene | 0.977269 |
| Tidligere 80/20 med valgte treantall / 29 CNN-epoker | **0.977576** |

Stacking forbedrer kontrollen med samme treningsbudsjett med 0.000270
og gir høyere AUC innen 12/14 aggregater. Det slår likevel ikke den
tidligere sterke referansen i denne lokale backtesten. Den lille
forskjellen avgjør ikke hvilken variant som får høyest Kaggle-score.
En egen sluttrefit og `submissions/submission_stacking.csv` er laget
for å teste dette på konkurransens testcaser.
Den eksisterende `submission_xgb_cnn_80_20.csv` med brukerrapportert
Kaggle-score 0.98152 beholdes som referanse.

De lærte koeffisientene gjelder standardiserte logits og trenger ikke
være positive eller summere til én, slik vanlige blandingsvekter gjør.
Meta-modellen konvergerte etter 17 iterasjoner. Dens trenings-AUC er
0.964178; dette er en in-sample-score for kombinasjonsmodellen, med
OOF-inputs fra 2020/2021, og kan ikke sammenlignes direkte med 2022-AUC
som en generaliseringsgap-diagnose. Utviklingsmodellen ligger i
`models/stacking_meta_2022.joblib`, og rapporten inkluderer fit-case-ID-er,
purget case-ID-er, skalering, koeffisienter og AUC per aggregat.

Sluttmodellen ligger i `models/stacking_meta_final.joblib`. Den bruker
1 096 historiske OOF-caser og 2 577 792 outputs til meta-trening, med
samme innstillinger og skalering lært på dette treningssettet.
Grunnmodellene bruker alle 2 922 oppgitte treningscaser: fem 600-trærs
XGBoost-prefixer og en CNN trent i 20 faste epoker. Individuelle
sluttprediksjoner ligger i `data/processed/stacking_*_final_predictions.csv`.
Bare den kombinerte CSV-en legges i `submissions`; den har 725 caser og
2 352 sannsynligheter per case. Kaggle-scoren for denne kandidaten er
foreløpig ukjent. Den frosne 2022-modellen og rapporten beholdes separat.

## CNN: flere seeds og vektgjennomsnitt

`configs/cnn_stability.json` låser en sammenligning av seeds 42, 7 og 21
på 2021 og 2022. Seed 42 gjenbrukes fra referansen; de to andre trenes
med samme features, arkitektur, BCE, learning rate og kronologiske splitter.
Ensemblet bruker likt gjennomsnitt av de tre modellenes sannsynligheter.

EMA og SWA testes separat på seed-42-trajektorien. EMA bruker decay 0.99
etter hvert optimizer-steg fra epoke 6. SWA tar et likt gjennomsnitt av
vekt-snapshots ved slutten av hver epoke fra epoke 10. Learning rate er
fortsatt fast; dette tester vektgjennomsnitt uten en egen SWA-læringsrateplan.
Raw-modellen styrer tidlig stopping, mens raw, EMA og SWA lagrer hver sin
beste checkpoint. Hele raw-historikken og beste raw-vekter er kontrollert
som identiske med referansen for begge år.

```bash
python src/train_cnn.py --config configs/cnn_averaging_2021.json
python src/train_cnn.py --config configs/cnn_averaging_2022.json
python src/train_cnn.py --config configs/cnn_seed7_2021.json
python src/train_cnn.py --config configs/cnn_seed21_2021.json
python src/train_cnn.py --config configs/cnn_seed7_2022.json
python src/train_cnn.py --config configs/cnn_seed21_2022.json
python src/compare_cnn_stability.py
```

Kjøringene bruker fire CPU-tråder og kjøres sekvensielt. Rapporten
`reports/cnn_stability_comparison.json` sammenligner CNN alene og hver
variant i samme 80 % XGBoost / 20 % CNN-blanding, med AUC per aggregat.
Seeds, averaging-innstillinger og blandingsvekter er bestemt før disse
kjøringene; valideringsårene er fortsatt utviklingsdata.

Sluttmodellene bruker faste epoker valgt fra 2022-rapportene. Seed 7 og
21 bruker `make_sequence_submission.py`; EMA/SWA bruker én felles full
treningskjøring i `make_cnn_averaging_submission.py`, med uavhengige
snapshots ved hver metodes valgte epoke. Individuelle prediksjoner lagres
i `data/processed`. `make_cnn_stability_submissions.py` lager tre kombinerte
Kaggle-kandidater med de samme 80/20-vektene: `seeds`, `ema` og `swa`.
Den opprinnelige `submission_xgb_cnn_80_20.csv` beholdes som referanse.

Resultatene for samme 80/20-blanding:

| CNN-variant | 2021 micro ROC-AUC | 2022 micro ROC-AUC |
| --- | --- | --- |
| Referanse, seed 42 | **0.974018** | 0.977576 |
| Gjennomsnitt av seeds 42, 7 og 21 | 0.973793 | **0.978015** |
| EMA, seed 42 | 0.973461 | 0.977172 |
| SWA, seed 42 | 0.973785 | 0.977271 |

Flere seeds forbedrer CNN alene på begge år, men den kombinerte modellen
forbedres bare på 2022. EMA og SWA gir litt lavere AUC enn referansen på
begge år. Kaggle-resultatet må undersøkes separat; ingen av disse
kandidatene erstatter den tidligere innsendte modellen automatisk.

Slutttrening bruker 29 epoker for den eksisterende seed-42-modellen,
17 for seed 7, 13 for seed 21 og 37 for EMA/SWA. Disse budsjettene kommer
fra de historiske 2022-rapportene. Alle 2 922 oppgitte treningscaser brukes;
konkurransens 725 testcaser brukes bare til prediksjon. EMA og SWA deler
én treningskjøring, mens seeds kombineres ved å snitte sannsynligheter.

```bash
python src/make_sequence_submission.py --config configs/submission_cnn_seed7.json --output data/processed/submission_cnn_seed7_predictions.csv
python src/make_sequence_submission.py --config configs/submission_cnn_seed21.json --output data/processed/submission_cnn_seed21_predictions.csv
python src/make_cnn_averaging_submission.py --config configs/submission_cnn_averaging.json
python src/make_cnn_stability_submissions.py
```

De tre nye filene til manuell Kaggle-opplasting er
`submissions/submission_xgb_cnn_seeds_80_20.csv`,
`submissions/submission_xgb_cnn_ema_80_20.csv` og
`submissions/submission_xgb_cnn_swa_80_20.csv`.
`reports/submission_cnn_stability_blends_run.json` dokumenterer kildene og
vektene. Kjøringer har egne artefaktnavn og stopper dersom et slikt
sluttartefakt allerede finnes.

## CNN: stopp på EMA-modellen

Kaggle-resultatene for de tre CNN-variantene er 0.98213 for seed-blandingen,
0.98165 for EMA og 0.98163 for SWA. Seed-blandingen er den nye personlige
rekorden. Vi prioriterer derfor å utvide den til fem seeds. EMA-oppsettene
nedenfor er klargjort for senere forsøk og er foreløpig ikke kjørt.
Den tidligere 2022-kjøringen avsluttet ved epoke 37 fordi raw-modellen
ikke hadde forbedret seg siden epoke 29. EMA fikk sin beste AUC ved
epoke 37, slik at den samme stoppregelen kunne avslutte EMA for tidlig.

`configs/cnn_ema_continuation.json` beskriver to forsøk med EMA som
stoppkriterium: eksisterende decay 0.99 (`long`) og decay 0.995 (`slow`).
Begge bruker maks 60 epoker og tålmodighet 8, med samme seed 42,
arkitektur, dropout, BCE, learning rate, features og kronologiske splitter.
Bare EMA-vekter snittes i disse forsøkene. Standardoppførselen for tidligere
configs er fortsatt raw-stopping med de opprinnelige EMA/SWA-variantene.

```bash
python src/train_cnn.py --config configs/cnn_ema_long_2021.json
python src/train_cnn.py --config configs/cnn_ema_long_2022.json
python src/train_cnn.py --config configs/cnn_ema_slow_2021.json
python src/train_cnn.py --config configs/cnn_ema_slow_2022.json
python src/compare_cnn_ema_continuation.py
```

Forsøkene kjøres sekvensielt med fire CPU-tråder. Sammenligningen bruker
fast 80 % fra det eksisterende XGBoost-ensemblet og 20 % fra hver CNN.
Den sammenligner både med den tidligere EMA-blandingen og raw-referansen.
Treningshistorikken før de gamle stopptidspunktene kontrolleres mot de
tidligere kjøringene. 2021 og 2022 er utviklingsvalidering; testcasene inngår
ikke i fitting, skalering eller epokevalg.

## CNN: fem seeds

`configs/cnn_five_seeds.json` låser seeds 42, 7, 21, 67 og 103 til et
likt gjennomsnitt av sannsynligheter. De første tre modellene gjenbrukes;
67 og 103 trenes på samme 2021-/2022-splitter med samme arkitektur,
features, optimizer og raw-stopping som tidligere. Maks 40 epoker og
tålmodighet 8 beholdes. Hver trening bruker fire CPU-tråder, sekvensielt.

```bash
python src/train_cnn.py --config configs/cnn_seed67_2021.json
python src/train_cnn.py --config configs/cnn_seed67_2022.json
python src/train_cnn.py --config configs/cnn_seed103_2021.json
python src/train_cnn.py --config configs/cnn_seed103_2022.json
python src/compare_cnn_five_seeds.py
```

Sammenligningen måler fem seeds mot de eksisterende tre, både CNN alene
og fast 80 % XGBoost / 20 % CNN. Alle fem beholdes med lik vekt;
ingen seeds velges bort ut fra valideringsresultatet. Rapporten ligger i
`reports/cnn_five_seeds_comparison.json`. Konkurransens testcaser brukes
bare til prediksjon etter slutttrening med historisk valgte epoker.

Resultatene for samme 80/20-blanding:

| CNN-variant | 2021 micro ROC-AUC | 2022 micro ROC-AUC |
| --- | --- | --- |
| Tre seeds | 0.973793 | 0.978015 |
| Fem seeds | **0.974092** | **0.978334** |

Fem seeds gir en liten forbedring på begge utviklingsår. CNN alene øker
fra 0.942243 til 0.947029 på 2021 og fra 0.961331 til 0.964370 på 2022.
Seeds 67 og 103 refittes på alle 2 922 oppgitte treningscaser i henholdsvis
21 og 13 epoker, valgt fra 2022-valideringen. De tre eksisterende
sluttmodellene og det eksisterende XGBoost-ensemblet gjenbrukes.

```bash
python src/make_sequence_submission.py --config configs/submission_cnn_seed67.json --output data/processed/submission_cnn_seed67_predictions.csv
python src/make_sequence_submission.py --config configs/submission_cnn_seed103.json --output data/processed/submission_cnn_seed103_predictions.csv
python src/make_cnn_five_seed_submission.py
```

Kun `submissions/submission_xgb_cnn_5seeds_80_20.csv` er en ny fil til
Kaggle-opplasting. Individuelle prediksjoner ligger i `data/processed`.
`reports/submission_xgb_cnn_5seeds_80_20_run.json` dokumenterer kildene;
hver CNN-seed har vekt 0.04 i sluttblandingen. Den tidligere vinnende
tre-seed-filen med Kaggle-score 0.98213 beholdes uendret.

Brukerrapportert Kaggle-score for fem-seed-filen er **0.98217**, en forbedring på 0.00004 fra tre-seed-filen (0.98213). Fem seeds beholdes som fast referanse for videre forsøk.

## Generalisering: økonomiske regimer og modellblanding

En ny analyse bruker eksisterende valideringsmodeller og prediksjoner.
Ingen modeller er trent eller endret i denne analysen. Den inkluderer
beskrivende sammenligninger av utdelte 2023-/2024-inputs, uten testlabels.

- `reports/cnn_fit_diagnosis_2022.json`: fem-seed-CNN scorer 0.993134 på
  sine 2 551 treningscaser og 0.964370 på 365 valideringscaser. Dette er
  full treningskohort, eval-modus og lagrede treningsscalere.
- `reports/temporal_regime_audit.json`: gjennomsnittspris per case-time
  øker fra 34.25 EUR/MWh i den purgede 2015–2021-kohorten til 211.24 i
  2022. 74.67 % av 2022-prisene ligger utenfor treningens 1.–99. persentil.
  Per-uke-standardisering reduserer den beskrivende KS-avstanden for pris
  fra 0.8753 til 0.0189. Det viser mindre inputskift, ikke dokumentert
  forbedring av en trent modell. Noen enkle normaliserte marginer gir
  høyere AUC, andre lavere; originale nivåer må derfor bevares.
- `reports/cnn_scaling_audit.json`: to konstante Vatjern/Haukeli-kanaler
  får kunstig standardavvik ved float32-akkumulering. De normaliseres
  omtrent til 1 i historiske fits og 0 i fullfit. Trenings- og
  valideringscaser bruker samme scaler innen hver modell; funnet forklarer
  derfor ikke hele generaliseringsgapet. Variabel-kanalenes avrundingsfeil
  har ubetydelig effekt i en diagnostisk kontroll med eksisterende vekter.
- `reports/current_generalization_errors.json`: Lio i mars 2022 har AUC
  0.58574 for XGBoost, 0.95424 for CNN og 0.76542 for fast 80/20-blanding.
  Modellstyrke varierer med driftssituasjonen. Dag 7 har høyere AUC enn
  dag 1 i begge år, og flertallet av terskelfeil ligger utenfor skifter.

Neste forsøk bør først isolere stabil skalering og normalisering av
ukens økonomiske inputs. Normaliser pris, vannverdi og marginer konsistent,
og behold prisnivå, startvolumer og annen fysisk tilstand. Ingen
reservoirtilstand skal forsvinne ved å sentrere en konstant ukekanal.
Per-sekvens-normalisering er en forskningsretning for tidsserier med
distribusjonsskift, blant annet i
[RevIN](https://openreview.net/pdf?id=cGDAkQo1C0p). Oppgaven vår predikerer
binære sannsynligheter, så RevINs inverse transformasjon av numeriske
forecast-outputs skal ikke brukes på ON-sannsynlighetene.

Et separat forsøk kan lære situasjonsavhengige kombinasjonsvekter fra
kronologiske OOF-prediksjoner og oppgitte inputs. Den tidligere lineære
stackingen har globale modellkoeffisienter og aggregat-bias, uten
pris-/tilstandsavhengige modellvekter. Nye vekter må trenes før året som
evalueres, med samme seks-casers purge og sammenlignbare grunnmodeller.
Den label-avhengige oracle-analysen i feilanalyserapporten er bare en
diagnostisk øvre grense og kan ikke brukes som valideringsresultat eller
submission. Overlappende uker gir avhengige observasjoner; KS-avstander
og små AUC-forskjeller er beskrivende, ikke uavhengige signifikanstester.

## Forsøk: ukentlig normalisering og situasjonsavhengig gate

`configs/normalization_experiment.json` låser preprocessing-forsøk på
2021 og 2022, med samme seed 42 og de tidligere CNN-innstillingene.
`cnn_stable_*` bruker float64-akkumulering av treningsmiddel/-standardavvik
og lagrer float32-scalere. `cnn_normalized_*` bruker samme skalering og
legger til 31 økonomiske kanaler. Alle 305 opprinnelige kanaler beholdes.
Prisens uke-zscore bruker standardavvik med gulv 1 EUR/MWh; pris,
lokal terminal vannverdi og marginer deles på ukens gjennomsnittlige
absolutte pris med gulv 10 EUR/MWh. Skalaen beholdes som egen kanal.
Alle ukestatistikker bruker bare casets 168 utdelte priser.

Et oppfølgingsforsøk, `cnn_rescaled_*`, deler de 92 eksisterende pris-/
vannverdikanalene på samme ukeskala, med gulv 10 EUR/MWh, og legger til
logaritmen av skalaen. Dette gir 306 kanaler. De 213 fysiske og øvrige
kanalene beholdes uendret; absolutt økonomisk nivå kan rekonstrueres fra
de skalerte kanalene og log-skalaen. Denne representasjonen brukes som
alternativ til de 31 ekstra kanalene, med samme stabile treningsscaler.

```bash
python src/train_cnn.py --config configs/cnn_stable_2021.json
python src/train_cnn.py --config configs/cnn_normalized_2021.json
python src/train_cnn.py --config configs/cnn_stable_2022.json
python src/train_cnn.py --config configs/cnn_normalized_2022.json
python src/train_cnn.py --config configs/cnn_rescaled_2021.json
python src/train_cnn.py --config configs/cnn_rescaled_2022.json
python src/compare_normalization.py --output reports/normalization_rescaling_comparison.json
```

Rapporten `reports/normalization_comparison.json` bevarer de første tre
variantene; `reports/normalization_rescaling_comparison.json` inkluderer
oppfølgingsforsøket. Rapportene sammenligner CNN alene,
80/20 med ett CNN-medlem og fem-CNN-blandingen der bare seed 42 byttes.
Begge normaliseringsrepresentasjonene endrer inputdimensjon og initialisering;
forsøket tester den samlede representasjonen. Gamle configs bruker
fortsatt den opprinnelige float32-skaleringen og kanalrekkefølgen.

`configs/conditional_gate.json` låser en liten LightGBM-modell med 300
trær og dybde 3. Den bruker 16 inputkanaler med oppgitt fysisk/økonomisk
kontekst og to eksperters sannsynligheter. Ekspertene er fem likevektede
XGBoost-modeller med 600 trær og én CNN med 20 epoker, gjenbrukt fra
de eksisterende kronologiske OOF-filene. Gate-eksperimentet har derfor
andre grunnmodeller enn den nåværende fem-CNN-submissionen.

Gaten lærer ON-sannsynlighet fra historiske labels, med større vekt der
ekspertene er uenige. Estimatet projiseres inn i intervallet mellom
ekspertenes prediksjoner, som gir situasjonsavhengige blandingsvekter.
Det uprojiserte estimatets historiske AUC lagres som diagnostikk.
Dette er et fast forsøk, uten early stopping eller parametersøk.

```bash
python src/train_conditional_gate.py --year 2021
python src/train_conditional_gate.py --year 2022
python src/train_conditional_gate.py --final
```

2021-gaten trenes på 360 OOF-caser fra 2020; 2022-gaten på 725 fra
2020/2021. Seks caser purges før hvert evalueringsår. Sluttversjonen
bruker alle 1 096 historiske OOF-caser gjennom 2022 og de tilsvarende
600-trærs/20-epokers sluttekspertene. Konkurransens testcaser brukes bare
til prediksjon. Rapporter sammenligner med fast 80/20 av de samme
ekspertene og markerer fem-CNN-referansen som et annet treningsbudsjett.
Alle tunge jobber kjøres sekvensielt med fire CPU-tråder.

Gate-resultatene viser en liten forbedring mot de samme ekspertene, men
ligger under den eksisterende fem-CNN-blandingen:

| Micro ROC-AUC | 2021 | 2022 |
|---|---:|---:|
| Fast 80/20, gate-ekspertene | 0.972555 | 0.976999 |
| Situasjonsavhengig gate | 0.972951 | 0.977867 |
| Eksisterende fem-CNN-blanding, annet treningsbudsjett | 0.974092 | 0.978334 |

Det uprojiserte metaestimatet var svakere i begge år. Rapportene er
historiske utviklingsresultater. Brukerrapporterte Kaggle-scores føres i
`reports/kaggle_scores.json`. De tidligere vinnende filene beholdes uendret.

Den konsekvente skaleringen av eksisterende økonomiske kanaler ga den
sterkeste nye faste blandingen på begge år. De første forsøkene med bare
stabil scaler eller ekstra relative kanaler ga ingen konsekvent forbedring
av fem-CNN-blandingen.

| Micro ROC-AUC | 2021 | 2022 |
|---|---:|---:|
| Opprinnelig CNN, seed 42 | 0.933421 | 0.953495 |
| CNN med økonomisk reskalering, seed 42 | 0.955212 | 0.963854 |
| Eksisterende fem-CNN + XGBoost, 80/20 | 0.974092 | 0.978334 |
| Reskalert CNN + XGBoost, 80/20 | **0.974956** | **0.979863** |
| Fem-CNN-blanding der bare seed 42 reskaleres | 0.974492 | 0.979129 |

`reports/cnn_rescaled_fit_diagnosis_2022.json` evaluerer det valgte
checkpointet med lagret treningsscaler. Trenings-AUC er 0.994125 og
validerings-AUC 0.963854; gapet er 0.030271, mot 0.039313 for opprinnelig
seed 42. Generaliseringsgapet er redusert, men fortsatt tydelig.

Sluttmodellen følger `configs/submission_cnn_rescaled.json`: 29 epoker
valgt fra 2022, 2 922 utdelte label-caser og ingen konkurransetestcaser.
Den kombineres med den eksisterende XGBoost-ensemblen ved de samme faste
80/20-vektene. Denne kandidaten bruker én reskalert CNN. Den eksisterende
fem-seed-submissionen beholdes som sammenligningsgrunnlag med Kaggle-score
0.98217.

```bash
python src/make_sequence_submission.py --config configs/submission_cnn_rescaled.json --output data/processed/submission_cnn_rescaled_predictions.csv
python src/blend_submissions.py --first submissions/submission_xgb_ensemble.csv --second data/processed/submission_cnn_rescaled_predictions.csv --first-weight 0.8 --output submissions/submission_xgb_cnn_rescaled_80_20.csv
```

To nye kombinerte filer er kandidater til manuell opplasting:

- `submissions/submission_xgb_cnn_rescaled_80_20.csv`: prioritert kandidat,
  med bedre historiske resultater på begge år.
- `submissions/submission_conditional_gate.csv`: separat gate-forsøk,
  svakere enn nåværende referanse historisk.

Komponentprediksjoner ligger i `data/processed`. Alle nye eksporter
avviser eksisterende modell-, CSV- og rapportfiler i stedet for å
overskrive dem. Brukerrapportert Kaggle-score 13. oktober 2026 for
`submission_xgb_cnn_rescaled_80_20.csv` er **0.98278**, ny personlig beste
med forbedring **0.00061** fra 0.98217. Gaten har foreløpig ingen rapportert
Kaggle-score.

Sluttkontrollen bekrefter 725 rader med 2 352 endelige sannsynligheter,
korrekt malrekkefølge og verdier i [0, 1]. 80/20-formelen er kontrollert
mot komponentene. Første, midterste og siste testcase er reprodusert fra
lagrede checkpoints: CNN innen 1e-6, gate eksakt. Hashene til eksisterende
referansesubmissions og den utdelte `evaluation/kaggle_metric.py` er uendret.

## Reskalert CNN: dropout og nyere treningsuker

`configs/cnn_rescaled_regularization.json` beskriver to separate, faste
forsøk på 2021 og 2022. Begge bruker seed 42, 306 inputkanaler, stabil
treningsscaler, 40 epoker som tak og early stopping med tålmodighet 8.

- `cnn_rescaled_dropout_*`: øker bare dropout fra 0.10 til 0.25.
- `cnn_rescaled_recency_*`: beholder dropout 0.10 og setter
  `recency_half_life_years` til 4.0. En treningsuke får dobbelt så stor
  relativ vekt som en uke fire år eldre. Vektene normaliseres til
  gjennomsnitt 1 og brukes på hver ukes gjennomsnittlige BCE over alle
  14 aggregater og 168 timer.

Utelat `recency_half_life_years` for den opprinnelige, uvektede loss-
funksjonen. Vektene beregnes bare fra treningscasenes startdatoer.
Scalere og micro ROC-AUC er uvektede. De samme kronologiske splittene
med seks purgede caser og de samme lagrede XGBoost-prediksjonene brukes.
Konkurransens testcaser inngår ikke i trening eller modellvalg.

```bash
python src/train_cnn.py --config configs/cnn_rescaled_dropout_2021.json
python src/train_cnn.py --config configs/cnn_rescaled_dropout_2022.json
python src/train_cnn.py --config configs/cnn_rescaled_recency_2021.json
python src/train_cnn.py --config configs/cnn_rescaled_recency_2022.json
python src/compare_sequence_ensemble.py --baseline-prefix cnn_rescaled --cnn-prefix cnn_rescaled_dropout --cnn-prefix cnn_rescaled_recency --average-with-baseline cnn_rescaled_dropout --output reports/cnn_rescaled_regularization_comparison.json
```

Jobbene kjøres sekvensielt med fire CPU-tråder. Forsøkene vurderes etter
uvektet micro ROC-AUC for hele 80/20-blandingen i begge historiske år;
et mindre trenings-/valideringsgap er ikke alene en forbedring.

| Micro ROC-AUC for hele 80/20-blandingen | 2021 | 2022 |
|---|---:|---:|
| Reskalert referanse, dropout 0.10 | 0.974956 | 0.979863 |
| Dropout 0.25 | 0.976171 | 0.979447 |
| Fire års recency-vekting | 0.974147 | 0.979269 |
| Fast 50/50-gjennomsnitt av referanse og dropout-CNN | 0.975879 | 0.979904 |

Dropout alene gir et blandet resultat, og recency svekker begge år.
Et oppfølgingsforsøk bruker ett fast, likevektet gjennomsnitt av de to
CNN-modellene, uten vekt-søk. Det gir forbedring på 0.000922 i 2021 og
0.000041 i 2022. Den siste forskjellen er svært liten; resultatene er
historiske utviklingsmålinger med overlappende caser og tidligere brukt
valideringsdata.

`reports/cnn_rescaled_regularization_diagnosis_2022.json` måler uvektet
trenings- og validerings-AUC med lagrede checkpoints og treningsscalere.
Dropout gir et gap på 0.030154, nesten likt referansens 0.030271, mens
recency øker gapet til 0.032494. CNN-gjennomsnittet gir 0.993922 på
trening og 0.965352 på validering, med gap 0.028570. Disse kontrollene
løser ikke det resterende generaliseringsproblemet.

Den eksperimentelle submissionen bruker 80 % eksisterende XGBoost-ensemble,
10 % eksisterende reskalert CNN og 10 % CNN med dropout 0.25. Bare den siste
modellen får en ny slutt-fit: `configs/submission_cnn_rescaled_dropout.json`
låser 18 epoker valgt fra 2022. Alle 2 922 utdelte label-caser brukes, og
konkurransens testcaser brukes bare til prediksjon. Recency er av.

```bash
python src/make_sequence_submission.py --config configs/submission_cnn_rescaled_dropout.json --output data/processed/submission_cnn_rescaled_dropout_predictions.csv
python src/blend_submissions.py --first data/processed/submission_cnn_rescaled_predictions.csv --second data/processed/submission_cnn_rescaled_dropout_predictions.csv --first-weight 0.5 --output data/processed/submission_cnn_rescaled_dropout_mean_predictions.csv
python src/blend_submissions.py --first submissions/submission_xgb_ensemble.csv --second data/processed/submission_cnn_rescaled_dropout_mean_predictions.csv --first-weight 0.8 --output submissions/submission_xgb_cnn_rescaled_dropout_mean_80_20.csv
```

Det blir én ny kombinert upload-fil:
`submissions/submission_xgb_cnn_rescaled_dropout_mean_80_20.csv`.
Kaggle-score er foreløpig ukjent. Referansen med score **0.98278** beholdes
uendret, sammen med den utdelte `evaluation/kaggle_metric.py`.

## Egne CNN-modeller for Lio og Haukeli

`configs/generator_specialists.json` beskriver fire historiske fits:
én CNN for hvert av `Lio_G1` og `Haukeli_G1`, på 2021 og 2022. Hver modell
får de samme 306 inputkanalene fra alle 14 aggregatene og hele den utdelte
uken, men lærer bare mål-aggregatets 168 ON-sannsynligheter. CNN-bredden,
dropout 0.10, seed 42, BCE, optimizer, 40-epokers tak og tålmodighet 8
beholdes. Et enkelt output-hode endrer også initialiseringen, så forsøket
tester hele spesialistoppsettet.

Sett `target_generator` i en CNN-config for å velge ett aggregat; utelat
nøkkelen for modellen med alle 14 outputs per time. Spesialistenes CSV-er
har bare `Run No` og de 168 målkolonnene og ligger i `data/processed`.
De er komponentprediksjoner. `make_sequence_submission.py` bruker den
valgte historiske epoken til slutt-fit og eksporterer de samme målkolonnene.

```bash
python src/train_cnn.py --config configs/cnn_specialist_lio_2021.json
python src/train_cnn.py --config configs/cnn_specialist_lio_2022.json
python src/train_cnn.py --config configs/cnn_specialist_haukeli_2021.json
python src/train_cnn.py --config configs/cnn_specialist_haukeli_2022.json
python src/compare_generator_specialists.py
```

Alle tunge jobber kjøres sekvensielt med fire CPU-tråder. De vanlige
kronologiske splittene med seks purgede caser brukes, og konkurransens
testcaser inngår ikke i trening. Early stopping bruker mål-aggregatets
AUC; den ferdige løsningen vurderes med micro ROC-AUC over alle 2 352 outputs.

Sammenligningen prøver to faste måter å bruke spesialistene på, for Lio,
Haukeli og begge samtidig: erstatt enten CNN-ens 20 %-bidrag for målet,
eller målets hele endelige sannsynlighet. Øvrige aggregaters prediksjoner
beholdes fra referansen med score 0.98278. Det gjøres ikke et søk over
blandingsvekter. Mål-AUC og global micro ROC-AUC rapporteres separat i
`reports/generator_specialists_comparison.json`.

De to blandede alternativene velger hver sin rute for de to aggregatene.
Med referanse, 20 % CNN-erstatning eller direkte erstatning for hvert mål
er det totalt ni faste alternativer, inkludert referansen.

| CNN-AUC for mål-aggregatet | 2021, delt CNN | 2021, spesialist | 2022, delt CNN | 2022, spesialist |
|---|---:|---:|---:|---:|
| Lio | 0.880741 | 0.932660 | 0.887148 | 0.912653 |
| Haukeli | 0.884271 | 0.947888 | 0.914929 | 0.931916 |

Begge spesialistene rangerer sitt eget mål bedre. Effekten på hele
submissionen er mer begrenset:

| Global micro ROC-AUC | 2021 | 2022 |
|---|---:|---:|
| Reskalert 80/20-referanse | 0.974956 | 0.979863 |
| Lio-spesialist i CNN-bidraget | 0.975195 | 0.979842 |
| Haukeli-spesialist i CNN-bidraget | **0.975334** | **0.980131** |
| Begge spesialister i CNN-bidraget | 0.975571 | 0.980107 |
| Begge brukt direkte | 0.971531 | 0.975460 |

Haukeli alene velges som den enklere kandidaten med forbedring i begge
år og høyest 2022-score blant alternativene som forbedrer begge år.
Lios globale bidrag er blandet. Direkte erstatning svekker global AUC;
for Haukeli forbedres mål-AUC samtidig som rangering mot andre aggregater
svekkes. Derfor beholdes XGBoost-bidraget på 80 %.

`reports/generator_specialists_diagnosis_2022.json` måler hvert måls
trenings-/valideringsgap fra lagrede checkpoints og treningsscalere.
Lios gap faller fra 0.107661 til 0.083661, og Haukelis fra 0.078533 til
0.062077. Disse er mål-spesifikke gap, ikke global AUC over alle aggregater.
Spesialisering hjelper, men gapene er fortsatt store.

Slutt-fit bruker `configs/submission_cnn_specialist_haukeli.json`, med
13 epoker valgt fra 2022 og alle 2 922 utdelte label-caser. Ingen
konkurransetestcaser inngår i trening. Bare Haukelis CNN-bidrag oppdateres;
de andre 13 aggregatene beholdes fra referansen med Kaggle-score 0.98278.

```bash
python src/make_sequence_submission.py --config configs/submission_cnn_specialist_haukeli.json --output data/processed/submission_cnn_specialist_haukeli_predictions.csv
python src/make_specialist_submission.py
```

Det er én ny ferdig upload-fil:
`submissions/submission_xgb_cnn_haukeli_specialist_80_20.csv`.
Brukerrapportert Kaggle-score 13. oktober 2026 er **0.98300**, ny personlig
beste og en forbedring på **0.00022** fra 0.98278. Dette var den nye
referansen med kjent public score. Sluttkontrollen bekrefter riktig
725 × 2 353-format og sannsynligheter i [0, 1]. Alle 2 184 andre outputs
per case er eksakt uendret; Haukeli følger den oppgitte 80/20-formelen.
Checkpoint-replay av 85 testcaser i opprinnelige batcher avviker mindre
enn 3e-8. Referansefiler, modellvekter og `evaluation/kaggle_metric.py`
har uendrede hashes.

## Transformer over hele uken

`src/weekly_transformer.py` bruker fire attention-lag med bredde 64,
fire heads og lærte timeposisjoner. Modellen har 165 326 parametre,
mot CNN-ens 164 366, og predikerer samme 14 × 168 sannsynligheter.
Den bruker de samme 306 reskalerte inputkanalene, BCE, seed 42,
læringsrate og treningsbudsjett som den reskalerte CNN-en. Alle utdelte
inputtimer er tilgjengelige for attention i begge retninger.

Historiske kjøringer bruker de samme tidsdelte 2021/2022-splittene,
med 365 valideringscaser og seks overlappende caser fjernet før hver
valideringsperiode. Treningene kjører sekvensielt med fire CPU-tråder.

```bash
python src/train_cnn.py --config configs/transformer_rescaled_2021.json
python src/train_cnn.py --config configs/transformer_rescaled_2022.json
python src/compare_sequence_ensemble.py --baseline-prefix cnn_rescaled --cnn-prefix transformer_rescaled --fixed-specialist-prefix cnn_specialist_haukeli --output reports/transformer_rescaled_comparison.json
```

| Modell | 2021 micro ROC-AUC | 2022 micro ROC-AUC |
| --- | ---: | ---: |
| Reskalert CNN alene | 0.955212 | 0.963854 |
| Transformer alene | 0.967254 | 0.974015 |
| Nåværende XGBoost/CNN-blanding med Haukeli-spesialist | 0.975334 | 0.980131 |
| XGBoost/Transformer-blanding med samme Haukeli-spesialist | **0.977605** | **0.980857** |

Begge blandinger bruker faste 80/20-vekter og samme fem XGBoost-modeller.
Haukelis spesialistbidrag holdes fast i sammenligningen. Ingen nye features
eller vektsøk inngår. Årene er utviklingsvalidering og brukes også til
early stopping. Transformerens beste epoker er 31 i 2021 og 15 i 2022.

`reports/transformer_rescaled_diagnosis.json` evaluerer de lagrede
checkpointene uten fitting. Inputnavn og scalere er eksakt like CNN-ens;
replay av valideringsprediksjonene avviker mindre enn 3e-8. I 2022 gir
Transformer 0.989436 på trening og 0.974015 på validering, et gap på
0.015421 mot CNN-ens 0.030271.

Slutt-fit bruker 15 faste epoker fra 2022-valideringen og alle 2 922 utdelte
label-caser. De 725 konkurransetestcasene holdes utenfor fitting.
Komponentprediksjoner lagres i `data/processed`; det lages én ferdig
upload-fil, med Haukelis sannsynligheter bevart fra submissionen med
kjent Kaggle-score 0.98300:

```bash
python src/make_sequence_submission.py --config configs/submission_transformer_rescaled.json --output data/processed/submission_transformer_rescaled_predictions.csv
python src/blend_submissions.py --first submissions/submission_xgb_ensemble.csv --second data/processed/submission_transformer_rescaled_predictions.csv --first-weight 0.8 --preserve-generator Haukeli_G1 --preserve-from submissions/submission_xgb_cnn_haukeli_specialist_80_20.csv --output submissions/submission_xgb_transformer_haukeli_80_20.csv
```

Brukerrapportert public Kaggle-score 13. oktober 2026 for
`submissions/submission_xgb_transformer_haukeli_80_20.csv` er **0.98365**,
ny personlig beste og en forbedring på **0.00065** fra 0.98300.
Dette var referansen før forsøket med vekter per kraftverk.

## Blandingsvekter per kraftverk

`src/plant_blend.py` gjenbruker lagrede XGBoost-, Transformer- og
Haukeli-prediksjoner. Ingen modeller trenes på nytt. `configs/plant_blend.json`
låser kandidatvektene til 0, 0.2, 0.5, 0.8 og 1.0 for Transformer-bidraget.
Først velges én felles vekt på 2021; deretter justeres hvert kraftverk én
gang i rekkefølgen fra submission-template. Valget maksimerer global
micro ROC-AUC, inkludert rangering mellom ulike aggregater. Aggregater
i samme kraftverk deler vekt. Haukeli beholdes uendret.

Alle vekter velges på 2021 og fryses før 2022 sjekkes. Ingen vekt endres
etter 2022-resultatet. Begge årene har vært brukt til utvikling og
early stopping tidligere; 2022 er en historisk bekreftelse.

| Blanding | 2021 micro ROC-AUC | 2022 micro ROC-AUC |
| --- | ---: | ---: |
| Tidligere 80 % XGBoost / 20 % Transformer | 0.977605 | 0.980857 |
| Felles 50 % Transformer, valgt på 2021 | 0.981499 | 0.983097 |
| Vekter per kraftverk, valgt på 2021 | **0.986722** | **0.984208** |

Valgte Transformer-vekter er Hogga 50 %, Tokke 100 %, Lio 50 %, Byrte 0 %,
Vinje 80 %, Songa 20 %, Kjela 20 % og Vesle Kjela 0 %. Resten av hvert
kraftverks bidrag kommer fra samme XGBoost-ensemble. Haukeli kopieres
fra referansen med public score 0.98365.

```bash
python src/plant_blend.py
python src/plant_blend.py --make-submission
```

Forsøk og vekter lagres i `reports/plant_blend_comparison.json`.
Det lages én ferdig upload-fil:
`submissions/submission_xgb_transformer_plant_blend.csv`.
Brukerrapportert public Kaggle-score 13. oktober 2026 er **0.98717**,
ny personlig beste og en forbedring på **0.00352** fra 0.98365.
Dette var referansen før finjusteringen av vektene. Endringen bruker
samme lagrede modeller; bare blandingsvektene per kraftverk er endret.

## Stabilitet og finere blandingsvekter

Et omvendt årsforsøk velger vekter på 2022 og sjekker dem på 2021 med
samme grove grid. Fem av åtte kraftverksvekter er identiske med 2021-valget.
De nye vektene og et fast 50/50-gjennomsnitt av de to årvalgene forbedrer
2022, men svekker 2021. Ingen av disse variantene eksporteres. Dette er
en stabilitetssjekk; gjennomsnittet bruker begge årenes labels til valg.

Et separat oppfølgingsforsøk starter fra 2021-vektene og gjør én passering
med vekter i steg på 0.1. Alle justeringer velges på 2021 og sjekkes
uendret på 2022. Det gir en liten forbedring på begge år:

| Vekter | 2021 micro ROC-AUC | 2022 micro ROC-AUC |
| --- | ---: | ---: |
| Referanse med kjent Kaggle-score 0.98717 | 0.986722 | 0.984208 |
| Grovt valg på 2022, sjekket på 2021 | 0.985935 | 0.985086 |
| Fast gjennomsnitt av begge grovvalg | 0.986584 | 0.984997 |
| Finere valg på 2021, sjekket på 2022 | **0.986790** | **0.984273** |

I den finere kandidaten går Hoggas Transformer-vekt fra 50 % til 40 %,
og Songas fra 20 % til 10 %. De øvrige kraftverksvektene beholdes;
Haukelis sannsynligheter kopieres uendret. Ingen nye modeller trenes,
og konkurransetestlabels inngår ikke i vektvalgene. Alle årstallene er
gjenbrukt utviklingsdata, ikke nye urørte holdout-resultater.

```bash
python src/plant_blend.py --config configs/plant_blend_2022.json --reference-report reports/plant_blend_comparison.json --comparison-report reports/plant_blend_stability_comparison.json
python src/plant_blend.py --config configs/plant_blend_refined.json --reference-report reports/plant_blend_comparison.json --comparison-report reports/plant_blend_refined_comparison.json
python src/plant_blend.py --config configs/plant_blend_refined.json --comparison-report reports/plant_blend_refined_comparison.json --make-submission --output submissions/submission_xgb_transformer_plant_refined.csv
```

Det er én ny kandidat: `submissions/submission_xgb_transformer_plant_refined.csv`.
Brukerrapportert public Kaggle-score 13. oktober 2026 er **0.98726**,
ny personlig beste og en forbedring på **0.00009** fra 0.98717.
Den finere blandingen er nå beste kjente referanse.

## Haukeli: egen blandingsvekt

`src/compare_haukeli_blend.py` tester bare Haukelis andel fra den lagrede
spesialist-CNN-en mot samme XGBoost-ensemble. De andre 13 aggregatene
beholdes fra den finere kraftverksblandingen med public score 0.98726.
`configs/haukeli_blend.json` låser kandidatene til steg på 0.1 fra 0 til 1.
Valget maksimerer global micro ROC-AUC på 2021; vekten fryses før 2022
sjekkes. Begge år er gjenbrukt utviklingsdata.

Den valgte Haukeli-blandingen er 70 % XGBoost og 30 % spesialist-CNN,
mot tidligere 80/20. Den gir en liten forbedring på begge år:

| Global blanding | 2021 micro ROC-AUC | 2022 micro ROC-AUC |
| --- | ---: | ---: |
| Finere kraftverksblanding, Haukeli 80/20 | 0.986790 | 0.984273 |
| Samme blanding, bare Haukeli endret til 70/30 | **0.986846** | **0.984383** |

Haukelis egen AUC øker fra 0.941748 til 0.945594 i 2021 og fra 0.935137
til 0.939321 i 2022. Ingen modeller trenes på nytt, og ingen
konkurransetestlabels brukes til å velge vekten.

```bash
python src/compare_haukeli_blend.py
python src/make_specialist_submission.py --config configs/submission_haukeli_blend.json --output submissions/submission_xgb_transformer_haukeli_70_30.csv
```

Forsøket lagres i `reports/haukeli_blend_comparison.json`. Én ny upload-fil
er `submissions/submission_xgb_transformer_haukeli_70_30.csv`, med alle
2 184 andre sannsynligheter per case eksakt bevart. Kaggle-score er
foreløpig ukjent; 0.98726 beholdes som beste kjente score.

## Økonomisk reskalering for tremodeller

`use_economic_rescaling` er en valgfri bryter som er av som standard.
[Forbehandlingen](../src/tree_preprocessing.py) deler priser, vannverdier og prismarginer
på gjennomsnittet av absoluttverdiene til casets 168 utdelte timepriser, minst 10 EUR/MWh.
Fysiske inputs beholdes, og `log_week_price_scale` bevarer informasjon om prisnivået.
Statistikken beregnes separat fra hvert cases inputs og bruker ingen labels.
Se [eksempelconfig](../configs/goal_xgboost_rescaled_2022.json) og
[kjørerapporten](../reports/autonomous_run_20261014.json) for oppsett og målte resultater.
Trenings- og eksportskriptene for disse forsøkene beskytter eksisterende modeller og resultater.
Kopier configen til et nytt filnavn før du kjører et allerede navngitt forsøk på nytt.
Valgfrie R-Drop- og RankNet-brytere finnes også i koden; de er foreløpig **ikke trent** i denne kjøringen.

Den bekreftede kandidaten fra denne kjøringen er [submission_rescaled_xgb_lgb_transformer.csv](../submissions/submission_rescaled_xgb_lgb_transformer.csv).
Den kombinerer reskalert XGBoost og LightGBM med Transformer og oppnådde lokal micro ROC-AUC **0.988272 / 0.987771** på 2021/2022.
Rapportert Kaggle-score er **0.989** (14. oktober 2026), ny personlig rekord. LightGBM brukte en tidsbegrensning på 800 trær; ingen test-labels ble brukt.
