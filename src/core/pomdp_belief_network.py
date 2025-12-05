from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List

def _load_prior(cpts_dir: Path) -> dict[str, float]:
    prior = {}
    with (cpts_dir / "prior_location.csv").open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            prior[r["location"]] = float(r["p"])
    return prior

def _load_cpt(cpts_dir: Path, feature: str):
    path = cpts_dir / f"cpt_{feature}.csv"
    values = []
    table = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        locations = header[1:]
        for row in rdr:
            v = int(row[0])
            values.append(v)
            for i, loc in enumerate(locations):
                table.setdefault(loc, {})[v] = float(row[1+i])
    return locations, values, table

def _normalize(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(d.values())
    if s <= 0:
        n = len(d); return {k: 1.0/n for k in d}
    return {k: v/s for k, v in d.items()}

# Simplified transition model T(s'|s)
# Wall_0 -> Door_Start_1 -> Door_1 -> Door_Passed_1 -> Wall_1 ...

# Shared transition parameters
P_WALL_STAY  = 0.8
P_WALL_START = 0.2
P_START_DOOR = 1.0
P_DOOR_STAY  = 0.8
P_DOOR_PASS  = 0.2
P_PASS_WALL  = 1.0

transition_model = {
    # Sequence 1
    'Wall_0':        {'Wall_0': P_WALL_STAY, 'Door_Start_1': P_WALL_START},
    'Door_Start_1':  {'Door_1': P_START_DOOR},
    'Door_1':        {'Door_1': P_DOOR_STAY, 'Door_Passed_1': P_DOOR_PASS},
    'Door_Passed_1': {'Wall_1': P_PASS_WALL},
    
    # Sequence 2
    'Wall_1':        {'Wall_1': P_WALL_STAY, 'Door_Start_2': P_WALL_START},
    'Door_Start_2':  {'Door_2': P_START_DOOR},
    'Door_2':        {'Door_2': P_DOOR_STAY, 'Door_Passed_2': P_DOOR_PASS},
    'Door_Passed_2': {'Wall_2': P_PASS_WALL},

    # Sequence 3
    'Wall_2':        {'Wall_2': P_WALL_STAY, 'Door_Start_3': P_WALL_START},
    'Door_Start_3':  {'Door_3': P_START_DOOR},
    'Door_3':        {'Door_3': P_DOOR_STAY, 'Door_Passed_3': P_DOOR_PASS},
    'Door_Passed_3': {'Wall_End': P_PASS_WALL},

    # End
    'Wall_End':      {'Wall_End': 1.0}
}

def get_cpt_label(state: str) -> str:
    """Maps specific states (e.g. Door_1) to generic CPT labels (e.g. Door)."""
    if "Wall_End" in state: return "Wall_End" # Keep Wall_End specific if it has unique properties
    if state.startswith("Wall"): return "Wall"
    if state.startswith("Door_Start"): return "Door_Start"
    if state.startswith("Door_Passed"): return "Door_Passed"
    if state.startswith("Door"): return "Door"
    return "Wall"

def pomdp_belief(readings: Dict[str, int], last_belief: Dict[str, float], inner_configuration: Dict) -> Dict[str, float]:
    cpts_dir = Path(inner_configuration.get("cpts_dir", "cpts"))
    door_states = set(inner_configuration.get("door_states", ["Door_Passed_1", "Door_Passed_2", "Door_Passed_3"]))

    prior = _load_prior(cpts_dir)
    
    # If no last_belief, start with the prior (or initialize Wall_0 = 1.0)
    if last_belief is None:
        # Initialize specifically to Wall_0 for the start of a run
        last_belief = {s: 0.0 for s in prior}
        if 'Wall_0' in last_belief:
            last_belief['Wall_0'] = 1.0
        else:
            last_belief = prior

    # Apply transition model: b_t(s') = sum_{s} T(s'|s) * b_{t-1}(s)
    predicted_belief = {s_prime: 0.0 for s_prime in prior}
    for s, prob in last_belief.items():
        if prob == 0: continue
        transitions = transition_model.get(s, {})
        for s_prime, trans_prob in transitions.items():
            if s_prime in predicted_belief:
                predicted_belief[s_prime] += trans_prob * prob
    
    predicted_belief = _normalize(predicted_belief)

    # Apply observation model (Bayes update)
    post = dict(predicted_belief)
    for feat, v in readings.items():
        cpt_file = cpts_dir / f"cpt_{feat}.csv"
        if not cpt_file.exists():
            continue
        locs, values, table = _load_cpt(cpts_dir, feat)
        
        # GENERALIZATION: Use generic labels for observation lookup.
        # The CPTs in 'cpts/' must have columns like 'Wall', 'Door', etc.
        # If you trained with specific labels, you need to retrain or ensure 
        # your training aggregates them.
        # Assuming the user will re-train such that CPTs have generic columns 
        # OR we map current specific columns to generic buckets if we can?
        
        # Actually, the user asked to "generalize transitions".
        # If we want to use data efficiently, we should TRAIN generic CPTs 
        # but RUN with specific states.
        
        # If the CPTs currently have specific columns (Wall_0, Door_1...), 
        # we can't just look up "Door" because it doesn't exist in the table.
        # But if we re-train to produce generic CPTs, then we MUST look up "Door".
        
        for loc in post:
            # If we are using GENERIC CPTs (trained on aggregated data):
            cpt_label = get_cpt_label(loc) 
            
            # Safety: if specific key exists, use it. If not, try generic.
            if loc in table:
                prob = table.get(loc, {}).get(v, 1e-9)
            elif cpt_label in table:
                prob = table.get(cpt_label, {}).get(v, 1e-9)
            else:
                prob = 1e-9
                
            val_to_lookup = int(v) if isinstance(v, bool) else v
            # Re-apply lookup with safe casting if needed (though table keys are ints)
            # The logic above just checked key existence. Now get value.
            
            # Simpler logic:
            if loc in table:
                target_col = loc
            elif cpt_label in table:
                target_col = cpt_label
            else:
                # Fallback to any column that matches the type? No, that's risky.
                # If we can't find the column, we ignore this observation for this state.
                target_col = None
            
            if target_col:
                post[loc] *= table.get(target_col, {}).get(val_to_lookup, 1e-9)

    post = _normalize(post)
    p_door = sum(post.get(s, 0.0) for s in door_states)
    
    # Loads the emission CPT file for one feature from cpts/
    def _table(feat):
        p = cpts_dir / f"cpt_{feat}.csv"
        if not p.exists(): return [], {}
        locs, values, tab = _load_cpt(cpts_dir, feat)
        return values, tab
    
    bins1, tab1 = _table("IR1")
    bins5, tab5 = _table("IR5")
    support = bins1 or bins5
    p_bin = {b: 0.0 for b in support}
    for b in support:
        # Use mapping here too if needed, though distance bins are usually just Wall/Door agnostic
        # But the original code weighted them by 'loc' probability.
        like1 = 0.0
        if bins1:
            for loc in post:
                like1 += post[loc] * tab1.get(loc, {}).get(b, 0.0)
        
        like5 = 0.0
        if bins5:
            for loc in post:
                like5 += post[loc] * tab5.get(loc, {}).get(b, 0.0)
                
        p_bin[b] = (like1 + like5) / (2 if (bins1 and bins5) else 1)
    Z = sum(p_bin.values()) or 1.0
    p_distance_bins = {b: v/Z for b,v in p_bin.items()}
    
    return {
        "posterior": post,
        "p_door_passed": p_door,
        "p_distance_bins": p_distance_bins
    }

def get_expected_reward(belief_state: Dict[str, float], inner_configuration: Dict) -> float:
    cpts_dir = Path(inner_configuration.get("cpts_dir", "cpts"))
    cpt_path = cpts_dir / "cpt_Reward.csv"
    
    if not cpt_path.exists():
        return 0.0
        
    # Load Reward CPT: P(Reward=1 | Location)
    # The CSV format is: value, Loc1, Loc2...
    # rows: 0 -> ...
    #       1 -> ...
    # We want the row for value=1
    
    locs, values, table = _load_cpt(cpts_dir, "Reward")
    
    # table is {Location: {Value: Prob}}
    # We want P(Reward=1 | s)
    
    expected_r = 0.0
    for s, prob in belief_state.items():
        # Use generic label mapping if specific key is not found
        cpt_label = get_cpt_label(s)
        
        if s in table:
            p_reward_1 = table.get(s, {}).get(1, 0.0)
        elif cpt_label in table:
            p_reward_1 = table.get(cpt_label, {}).get(1, 0.0)
        else:
            p_reward_1 = 0.0
            
        expected_r += prob * p_reward_1
        
    return expected_r
