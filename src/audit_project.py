"""Audit supplied data, alignment, and difficult cases without training."""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import HOURS, ROOT, build_features, feeding_reservoirs, flatten_labels, load_data, prediction_frame

sys.path.insert(0, str(ROOT / "evaluation"))
from kaggle_metric import score


def main():
    cases, columns, generators, inputs, topology = load_data(use_extended=True)
    test = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")
    test["starttime"] = pd.to_datetime(test.starttime, utc=True)
    all_cases = pd.concat([cases[["Run No", "starttime"]], test], ignore_index=True)
    assert not cases["Run No"].isin(test["Run No"]).any()
    stamps = pd.DatetimeIndex(np.repeat(all_cases.starttime.array, HOURS)) + pd.to_timedelta(np.tile(np.arange(HOURS), len(all_cases)), unit="h")
    unique_hours = stamps.unique()
    coverage = {}
    for key in ["price", "inflow", "min_volume", "min_flow"]:
        frame = inputs[key]
        assert frame.index.is_unique
        missing = unique_hours.difference(frame.index)
        coverage[key] = {"missing_required_hours": len(missing), "missing_values": int(frame.isna().sum().sum())}
        assert missing.empty
    for key, dates in [("volume", all_cases.starttime), ("water_value", all_cases.starttime + pd.Timedelta(hours=HOURS))]:
        missing = pd.DatetimeIndex(dates).difference(inputs[key].index)
        coverage[key] = {"missing_required_dates": len(missing)}
        assert missing.empty

    selected = cases.iloc[[0, -1]].copy()
    features = build_features(selected, generators, inputs, topology, use_extended=True, use_system_context=True, use_topology=True, use_price_context=True)
    for g, generator in enumerate(generators):
        feeding = feeding_reservoirs(topology, generator.rsplit("_G", 1)[0])
        for c in range(len(selected)):
            start = selected.starttime.iloc[c]
            for hour in [0, 24, 167]:
                row = features.iloc[g*len(selected)*HOURS+c*HOURS+hour]
                stamp = start + pd.Timedelta(hours=hour)
                np.testing.assert_allclose(row.price, inputs['price'].loc[stamp].iloc[0], rtol=1e-6)
                expected = inputs['water_value'].loc[start+pd.Timedelta(hours=HOURS),feeding].mean()
                np.testing.assert_allclose(row.terminal_water_value,expected,rtol=1e-6)
    flat = flatten_labels(selected, columns, generators)
    for g, generator in enumerate(generators):
        for c in range(len(selected)):
            for hour in [0, 24, 167]:
                assert flat[g*len(selected)*HOURS+c*HOURS+hour] == selected.iloc[c][f'result_committed_{generator}_t{hour}']
    restored = prediction_frame(flat, selected, columns, generators)
    np.testing.assert_array_equal(restored[columns], selected[columns])
    assert score(selected[["Run No",*columns]], restored)==1.0

    checksums = {}
    archive = Path('/Users/hallvardgrundt/Downloads/project materials.zip')
    if archive.exists():
        local_files = list((ROOT/'data/raw').glob('**/*')) + list((ROOT/'docs').glob('*')) + list((ROOT/'configs').glob('*')) + list((ROOT/'notebooks').glob('*')) + list((ROOT/'evaluation').glob('*'))
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                if member.is_dir() or any(x in member.filename for x in ['__MACOSX','.DS_Store','__pycache__']):continue
                name=Path(member.filename).name
                candidates=[p for p in local_files if p.is_file() and p.name==name]
                assert len(candidates)==1
                matches=hashlib.sha256(zipped.read(member)).digest()==hashlib.sha256(candidates[0].read_bytes()).digest()
                assert matches
                checksums[name]=matches

    results, examples, graph_cases, unit_priority = {}, [], {}, {}
    for year in [2021,2022]:
        validation=cases[cases.starttime.dt.year==year]
        unit_priority[str(year)] = {}
        for plant, count in [('Tokke', 4), ('Vinje', 3)]:
            values=validation[[f'result_committed_{plant}_G{g}_t{h}' for g in range(1,count+1) for h in range(HOURS)]].to_numpy().reshape(-1,count,HOURS)
            unit_priority[str(year)][plant]={'adjacent_order_violations':int((values[:,1:,:]>values[:,:-1,:]).sum()),'comparisons':int(values[:,1:,:].size)}
        xgb=pd.read_csv(ROOT/f'data/processed/xgboost_price_{year}_validation_predictions.csv').set_index('Run No')
        lgb=pd.read_csv(ROOT/f'data/processed/topology_{year}_validation_predictions.csv').set_index('Run No')
        assert xgb.index.equals(lgb.index)
        predictions=(.75*xgb+.25*lgb).loc[validation['Run No'],columns]
        hourly=pd.DatetimeIndex(np.repeat(validation.starttime.array,HOURS))+pd.to_timedelta(np.tile(np.arange(HOURS),len(validation)),unit='h')
        river=inputs['inflow'].loc[hourly,'r_Vest_Vassdraget'].to_numpy()
        required=inputs['min_flow'].loc[hourly,'r_Vest_Vassdraget'].to_numpy()
        lio_columns=[f'result_committed_Lio_G1_t{t}' for t in range(HOURS)]
        y=validation[lio_columns].to_numpy().ravel();p=predictions[lio_columns].to_numpy().ravel()
        groups={}
        for name,mask in [('natural_inflow_meets_requirement',river>=required),('natural_inflow_below_requirement',river<required)]:
            groups[name]={'outputs':int(mask.sum()),'on_fraction':float(y[mask].mean()),'error_at_0_5':float(((p[mask]>=.5)!=y[mask]).mean())}
        results[str(year)]=groups
        for generator in ['Lio_G1','Haukeli_G1']:
            generator_columns=[f'result_committed_{generator}_t{t}' for t in range(HOURS)]
            target=validation[generator_columns].to_numpy()
            probabilities=predictions[generator_columns].to_numpy()
            wrong=(probabilities>=.5)!=target
            confidence=np.where(wrong,np.abs(probabilities-target),-1)
            for flat_index in np.argsort(confidence.ravel())[-2:][::-1]:
                case_index,hour=np.unravel_index(flat_index,target.shape)
                case=validation.iloc[case_index];stamp=case.starttime+pd.Timedelta(hours=int(hour))
                plant=generator.rsplit('_G',1)[0];feed=feeding_reservoirs(topology,plant)
                record={'year':year,'Run No':int(case['Run No']),'generator':generator,'hour':int(hour),'timestamp':str(stamp),
                        'label':int(target[case_index,hour]),'probability':float(probabilities[case_index,hour]),
                        'price':float(inputs['price'].loc[stamp].iloc[0]),
                        'terminal_local_water_value_mean':float(inputs['water_value'].loc[case.starttime+pd.Timedelta(hours=HOURS),feed].mean()),
                        'initial_feeding_volume_million_m3':float(inputs['volume'].loc[case.starttime,feed].sum()),
                        'local_natural_inflow_m3_s':float(inputs['inflow'].loc[stamp,feed].sum()),
                        'river_natural_inflow_m3_s':float(inputs['inflow'].loc[stamp,'r_Vest_Vassdraget']),
                        'river_minimum_flow_m3_s':float(inputs['min_flow'].loc[stamp,'r_Vest_Vassdraget'])}
                examples.append(record)
                if year==2022 and generator not in graph_cases:graph_cases[generator]=(case,target[case_index],probabilities[case_index],record)

    summary={'coverage':coverage,'train_cases':len(cases),'test_cases':len(test),'original_files_unchanged':checksums,
             'alignment_checks_passed':True,'river_inflow_feature_present':any('r_Vest_Vassdraget' in name and 'inflow' in name for name in features.columns),
             'constant_initial_volumes':{n:{'unique_values':int(inputs['volume'][n].nunique()),'volume_million_m3':float(inputs['volume'][n].iloc[0])} for n in ['Vatjern','Hyljelihyl']},
             'lio_river_diagnostics':results, 'unit_priority_diagnostics':unit_priority,
             'first_subhourly_price_date':str(inputs['price'].index[inputs['price'].index.minute != 0].min())}
    (ROOT/'reports/project_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    pd.DataFrame(examples).to_csv(ROOT/'reports/project_audit_cases.csv',index=False)

    fig,axes=plt.subplots(3,2,figsize=(14,9),sharex=True)
    for col,(generator,(case,target,probabilities,record)) in enumerate(graph_cases.items()):
        hours=pd.date_range(case.starttime,periods=HOURS,freq='h')
        price=inputs['price'].loc[hours].iloc[:,0].to_numpy()
        axes[0,col].plot(np.arange(HOURS),price,label='Price')
        axes[0,col].axhline(record['terminal_local_water_value_mean'],linestyle='--',label='Local terminal water value')
        axes[0,col].set_title(f"{generator}: case {int(case['Run No'])}, {case.starttime.date()}")
        axes[0,col].set_ylabel('EUR/MWh')
        axes[1,col].step(np.arange(HOURS),target,where='post',label='True ON/OFF')
        axes[1,col].plot(np.arange(HOURS),probabilities,label='Blend probability')
        axes[1,col].set_ylim(-.05,1.05)
        if generator=='Lio_G1':
            axes[2,col].plot(np.arange(HOURS),inputs['inflow'].loc[hours,'r_Vest_Vassdraget'],label='Unused natural river inflow')
            axes[2,col].plot(np.arange(HOURS),inputs['min_flow'].loc[hours,'r_Vest_Vassdraget'],label='Minimum river flow')
        else:
            for reservoir in ['Vatjern','Langeidvatn']:
                axes[2,col].plot(np.arange(HOURS),inputs['inflow'].loc[hours,reservoir],label=f'Natural inflow {reservoir}')
        axes[2,col].set_ylabel('m3/s');axes[2,col].set_xlabel('Hour within case')
        for ax in axes[:,col]:
            ax.axvline(record['hour'],color='red',alpha=.4,linestyle=':');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'reports/figures/project_audit_cases.png',dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2))
    print(pd.DataFrame(examples).to_string(index=False))


if __name__=='__main__':
    main()
