
import json, math
from pathlib import Path
import pandas as pd
import numpy as np

class NaiveBayesBN:
    def __init__(self, cpts_dir):
        cpts = Path(cpts_dir)
        self.prior = pd.read_csv(cpts/'prior_location.csv')
        meta = json.loads((cpts/'meta.json').read_text())
        self.locations = meta['locations']; self.features = meta['features']
        self.cpts = {}
        for feat in self.features:
            df = pd.read_csv(cpts/f'cpt_{feat}.csv').set_index('location')
            self.cpts[feat]=df

    def posterior_location(self, reading):
        p = self.prior.set_index('location').loc[self.locations]['P(Location)'].values.astype(float)
        logp = np.log(p + 1e-12)
        for feat,val in reading.items():
            if feat not in self.cpts: continue
            table = self.cpts[feat]; col = str(val)
            if col in table.columns:
                cond = table[col].reindex(self.locations).fillna(1e-9).values
            else:
                cond = np.ones(len(self.locations))/len(self.locations)*1e-6
            logp += np.log(cond + 1e-12)
        logp -= logp.max(); p = np.exp(logp); p /= p.sum()
        return p

def belief(sonar_readings, inner_configuration):
    cpts_dir = inner_configuration.get('cpts_dir','cpts')
    door_states = inner_configuration.get('door_states',['door_passed'])
    bn = NaiveBayesBN(cpts_dir)
    post = bn.posterior_location(sonar_readings)
    p_door = float(sum(post[i] for i,l in enumerate(bn.locations) if l in door_states))
    return {'locations': bn.locations, 'p_location': post.tolist(), 'p_door_passed': p_door}

def door_passed_10cm_ago(history_beliefs, cm_per_step=1.0, door_states=('door_passed',)):
    if not history_beliefs: return 0.0
    k = int(round(10.0/max(1e-6, cm_per_step)))
    idx = max(0, len(history_beliefs)-1-k)
    b = history_beliefs[idx]
    locs = b.get('locations',[]); probs = b.get('p_location',[])
    return float(sum(probs[i] for i,l in enumerate(locs) if l in door_states))

if __name__=='__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('--cpts', default='cpts')
    ap.add_argument('--readings', required=True)
    a = ap.parse_args()
    r = json.loads(a.readings)
    print(json.dumps(belief(r, {'cpts_dir':a.cpts}), indent=2))
