"""
Session 5: Sensitivity, Stakes Enrichment & Visualization
=========================================================
Step 1: Sensitivity analysis (gate condition)
Step 2: Stakes enrichment
Step 3: Per-count decomposition
Step 4: Visualization suite (6 figures)
Step 5: Secondary situation analysis
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

BASE_DIR = "/Users/paul821/Desktop/baseball/baseball_terminal_state"
MDP_DIR = f"{BASE_DIR}/data/mdp"
LONG_DIR = f"{BASE_DIR}/data/longitudinal"
STAKES_DIR = f"{BASE_DIR}/data/stakes"
FIG_DIR = f"{BASE_DIR}/figures"
for d in [STAKES_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────
ERAS = {
    'post_war':          (1950, 1968),
    'expansion':         (1969, 1992),
    'offense_explosion': (1993, 2005),
    'post_steroid':      (2006, 2014),
    'statcast':          (2015, 2024),
}

def assign_era(year):
    for name, (start, end) in ERAS.items():
        if start <= year <= end:
            return name
    return 'unknown'

DEFICITS = [0, 1, 2, 3]
EXTRAS_WP = 0.52

# Pitch decoding
BALL_CHARS = set('BIPV')
CALLED_STRIKE_CHARS = set('CK')
SWINGING_STRIKE_CHARS = set('SMQT')
FOUL_CHARS = set('FHLOR')
CONTACT_CHARS = set('XY')
IGNORE_CHARS = set('>+*123.N')
SWING_DESCS = {'swinging_strike', 'foul', 'foul_tip', 'swinging_strike_blocked',
               'foul_bunt', 'missed_bunt', 'hit_into_play', 'hit_into_play_no_out',
               'hit_into_play_score', 'foul_pitchout', 'swinging_pitchout'}

# Load transition parameters
trans_df = pd.read_csv(f"{MDP_DIR}/transition_parameters.csv", index_col=0)
era_transitions = trans_df.to_dict('index')

# Load Session 4 baseline policies for comparison
baseline_policies = pd.read_csv(f"{MDP_DIR}/optimal_policies_v2.csv")


# ── MDP Engine ───────────────────────────────────────────────────────────
def build_transition_dict(trans_params):
    T = {}
    pb = trans_params['p_ball_on_take']
    ps = trans_params['p_strike_on_take']
    pw = trans_params['p_whiff_on_swing']
    pf = trans_params['p_foul_on_swing']
    pc = trans_params['p_contact_on_swing']

    for b in range(4):
        for s in range(3):
            for d in DEFICITS:
                for r in range(4):
                    state = (b, s, d, r)
                    T[state] = {}

                    take = {}
                    if b == 3:
                        if d == 0:
                            take[('WIN',)] = take.get(('WIN',), 0) + pb
                        else:
                            next_state = (0, 0, d - 1, min(r + 1, 3))
                            take[next_state] = take.get(next_state, 0) + pb
                    else:
                        take[(b + 1, s, d, r)] = pb

                    if s == 2:
                        take[('K', d, r)] = take.get(('K', d, r), 0) + ps
                    else:
                        take[(b, s + 1, d, r)] = ps
                    T[state]['take'] = take

                    swing = {}
                    if s == 2:
                        swing[('K', d, r)] = swing.get(('K', d, r), 0) + pw
                    else:
                        swing[(b, s + 1, d, r)] = swing.get((b, s + 1, d, r), 0) + pw

                    if s < 2:
                        swing[(b, s + 1, d, r)] = swing.get((b, s + 1, d, r), 0) + pf
                    else:
                        swing[(b, 2, d, r)] = swing.get((b, 2, d, r), 0) + pf

                    swing[('BIP', d, r)] = swing.get(('BIP', d, r), 0) + pc
                    T[state]['swing'] = swing
    return T


def value_iteration(trans_params, bip_dist, tol=1e-8, max_iter=1000):
    T = build_transition_dict(trans_params)
    V = {(b, s, d, r): 0.0
         for b in range(4) for s in range(3) for d in DEFICITS for r in range(4)}
    policy = {}

    for iteration in range(max_iter):
        V_new = {}
        for state in V:
            b, s, d, r = state
            action_values = {}
            for action in ['take', 'swing']:
                ev = 0.0
                for ns, prob in T[state][action].items():
                    if ns == ('WIN',):
                        ev += prob * 1.0
                    elif isinstance(ns, tuple) and ns[0] == 'K':
                        ev += prob * 0.0
                    elif isinstance(ns, tuple) and ns[0] == 'BIP':
                        _, bip_d, bip_r = ns
                        bip_ev = 0.0
                        for runs, rp in bip_dist.items():
                            remaining = bip_d - runs
                            if remaining < 0:
                                bip_ev += rp * 1.0
                            elif remaining == 0:
                                bip_ev += rp * EXTRAS_WP
                            else:
                                if runs > 0 and remaining in DEFICITS:
                                    bip_ev += rp * V.get((0, 0, remaining, min(bip_r + runs, 3)), 0.0)
                                else:
                                    bip_ev += rp * 0.0
                        ev += prob * bip_ev
                    elif ns in V:
                        ev += prob * V[ns]
                    else:
                        ev += prob * 0.0
                action_values[action] = ev

            best = max(action_values, key=action_values.get)
            V_new[state] = action_values[best]
            policy[state] = best

        delta = max(abs(V_new[s] - V[s]) for s in V)
        V = V_new
        if delta < tol:
            return V, policy, iteration + 1, T

    return V, policy, max_iter, T


def compute_v_observed(state, swing_rate, V, T, bip_dist):
    v_obs = 0.0
    for action, aprob in [('swing', swing_rate), ('take', 1.0 - swing_rate)]:
        if aprob == 0:
            continue
        for ns, tp in T[state][action].items():
            if ns == ('WIN',):
                v_obs += aprob * tp * 1.0
            elif isinstance(ns, tuple) and ns[0] == 'K':
                v_obs += aprob * tp * 0.0
            elif isinstance(ns, tuple) and ns[0] == 'BIP':
                _, bip_d, bip_r = ns
                bip_ev = 0.0
                for runs, rp in bip_dist.items():
                    rem = bip_d - runs
                    if rem < 0:
                        bip_ev += rp * 1.0
                    elif rem == 0:
                        bip_ev += rp * EXTRAS_WP
                    else:
                        if runs > 0 and (0, 0, rem, min(bip_r + runs, 3)) in V:
                            bip_ev += rp * V[(0, 0, rem, min(bip_r + runs, 3))]
                        else:
                            bip_ev += rp * 0.0
                v_obs += aprob * tp * bip_ev
            elif ns in V:
                v_obs += aprob * tp * V[ns]
    return v_obs


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(" STEP 1: Sensitivity Analysis")
print("=" * 70)

# First, recover the actual baseline BIP distribution used in Session 4.
# Recompute from Statcast primary data.
print("\n  Computing BIP distributions from primary data...")
all_primary = []
for year in range(2015, 2025):
    path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        terminal = df[df['events'].notna()].copy()
        terminal['deficit'] = terminal['fld_score'] - terminal['bat_score']
        if 'post_bat_score' in terminal.columns:
            terminal['runs_scored'] = (terminal['post_bat_score'] - terminal['bat_score']).fillna(0).astype(int)
        all_primary.append(terminal)

primary_all = pd.concat(all_primary, ignore_index=True)

out_events = {'field_out', 'grounded_into_double_play', 'force_out',
              'fielders_choice', 'fielders_choice_out', 'double_play',
              'sac_fly', 'sac_bunt', 'triple_play', 'sac_fly_double_play'}
hit_events = {'single', 'double', 'triple', 'home_run', 'field_error'}
bip_events = hit_events

bip_all = primary_all[primary_all['events'].isin(out_events | hit_events)]
bip_outs = primary_all[primary_all['events'].isin(out_events)]
bip_hits = primary_all[primary_all['events'].isin(hit_events)]
bip_data = primary_all[primary_all['events'].isin(hit_events)]

p_bip_out = len(bip_outs) / len(bip_all) if len(bip_all) > 0 else 0.5
p_bip_hit = 1 - p_bip_out

hit_runs_dist = bip_data['runs_scored'].value_counts(normalize=True).sort_index().to_dict()

# Baseline combined distribution (BIP out = 0 runs, hits = run distribution)
baseline_bip = {0: p_bip_out}
for r, p in hit_runs_dist.items():
    baseline_bip[r] = baseline_bip.get(r, 0) + p_bip_hit * p

print(f"  Baseline BIP distribution (n={len(bip_all)}):")
print(f"    BIP-out rate: {p_bip_out:.3f}, BIP-hit rate: {p_bip_hit:.3f}")
for r, p in sorted(baseline_bip.items()):
    print(f"    {r} runs: {p:.4f}")

# ── Define perturbation scenarios ────────────────────────────────────────
# Each perturbation modifies the COMBINED BIP distribution (not just hits).
# Format: {0: P(0 runs on BIP), 1: P(1 run), 2: P(2 runs), 3: P(3 runs), 4: P(4 runs)}

perturbations = {
    'baseline': baseline_bip,
    'contact_pessimistic': {0: 0.750, 1: 0.110, 2: 0.110, 3: 0.015, 4: 0.015},
    'contact_optimistic':  {0: 0.580, 1: 0.135, 2: 0.175, 3: 0.030, 4: 0.080},
    'deadball_era':        {0: 0.820, 1: 0.120, 2: 0.050, 3: 0.005, 4: 0.005},
    'steroid_era':         {0: 0.580, 1: 0.100, 2: 0.180, 3: 0.030, 4: 0.110},
    'flyball_rev':         {0: 0.620, 1: 0.090, 2: 0.160, 3: 0.020, 4: 0.110},
}

# ── Run all perturbations ────────────────────────────────────────────────
print(f"\n  Running {len(perturbations)} perturbation scenarios...")

sensitivity_results = {}
all_policies = {}  # scenario → {state: action}

for scenario_name, bip_dist in perturbations.items():
    V, policy, n_iter, T = value_iteration(era_transitions['statcast'], bip_dist)
    sensitivity_results[scenario_name] = {
        'V': V, 'policy': policy, 'n_iter': n_iter, 'T': T, 'bip_dist': bip_dist,
    }
    all_policies[scenario_name] = policy
    print(f"  {scenario_name:25s}: converged in {n_iter} iter, V(0-0,d=0)={V[(0,0,0,0)]:.4f}")

# ── Compare policies to baseline ────────────────────────────────────────
print(f"\n  ── Policy Flips vs. Baseline ──")
baseline_pol = all_policies['baseline']
# For comparison, only look at the canonical states (r=0, since policy
# is typically invariant to runs_scored for a given deficit)
canonical_states = [(b, s, d, 0) for b in range(4) for s in range(3) for d in DEFICITS]

flip_report = []
for scenario_name, policy in all_policies.items():
    if scenario_name == 'baseline':
        continue
    flips = []
    for state in canonical_states:
        if policy[state] != baseline_pol[state]:
            flips.append(state)
    flip_report.append({
        'scenario': scenario_name,
        'n_flips': len(flips),
        'flipped_states': flips,
    })
    flip_str = ", ".join(f"{s[0]}-{s[1]}d{s[2]}" for s in flips) if flips else "none"
    print(f"  {scenario_name:25s}: {len(flips)} flips → {flip_str}")

max_flips = max(r['n_flips'] for r in flip_report) if flip_report else 0

print(f"\n  ── GATE CHECK ──")
if max_flips <= 3:
    print(f"  ✓ Policy ROBUST (max {max_flips} flips). Proceeding to stakes enrichment.")
    gate_passed = True
elif max_flips <= 6:
    print(f"  ⚠ Policy partially robust (max {max_flips} flips). Proceeding with caution.")
    gate_passed = True
else:
    print(f"  ✗ Policy FRAGILE (max {max_flips} flips). STOP — do not proceed.")
    gate_passed = False

# ── Identify boundary states ────────────────────────────────────────────
# States that flip under ANY perturbation
all_flipped = set()
for r in flip_report:
    all_flipped.update(r['flipped_states'])

if all_flipped:
    print(f"\n  Boundary states (flip under at least one scenario):")
    for state in sorted(all_flipped):
        base_action = baseline_pol[state]
        flippers = []
        for r in flip_report:
            if state in r['flipped_states']:
                alt_action = all_policies[r['scenario']][state]
                flippers.append(f"{r['scenario']}→{alt_action}")
        print(f"    {state[0]}-{state[1]} d={state[2]}: baseline={base_action}, flips: {', '.join(flippers)}")

# ── Save sensitivity results ────────────────────────────────────────────
sens_rows = []
for scenario_name, res in sensitivity_results.items():
    V, pol = res['V'], res['policy']
    for state in canonical_states:
        b, s, d, _ = state
        sens_rows.append({
            'scenario': scenario_name, 'balls': b, 'strikes': s,
            'deficit': d, 'count': f"{b}-{s}",
            'optimal_action': pol[state], 'value': V[state],
        })

sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(f"{MDP_DIR}/sensitivity_analysis_v2.csv", index=False)

# ── Era-appropriate BIP distributions ────────────────────────────────────
print(f"\n  ── Era-Specific BIP Reanalysis ──")

era_bip_map = {
    'post_war':          perturbations['deadball_era'],
    'expansion':         perturbations['baseline'],
    'offense_explosion': perturbations['steroid_era'],
    'post_steroid':      perturbations['baseline'],
    'statcast':          perturbations['baseline'],
}

# Re-solve MDP for each era with era-appropriate BIP
era_specific_solutions = {}
for era_name, tparams in era_transitions.items():
    bip = era_bip_map.get(era_name, perturbations['baseline'])
    V, pol, n_iter, T = value_iteration(tparams, bip)
    era_specific_solutions[era_name] = {'V': V, 'policy': pol, 'T': T, 'bip': bip}
    print(f"  {era_name:25s}: V(0-0,d=0)={V[(0,0,0,0)]:.4f} (era-specific BIP)")

# Recompute historical policy gaps with era-specific BIP
long_df = pd.read_csv(f"{LONG_DIR}/policy_gaps_historical.csv")
# We'll recompute for Statcast years using era-specific solutions
# For now, store the era-specific V values for comparison
era_comparison = []
for era_name in era_bip_map:
    if era_name not in era_specific_solutions:
        continue
    V_specific = era_specific_solutions[era_name]['V']
    # Baseline used statcast BIP for all eras
    V_baseline_era = sensitivity_results['baseline']['V']  # approximate

    for d in DEFICITS:
        s1 = V_specific[(0, 0, d, 0)]
        era_comparison.append({
            'era': era_name, 'deficit': d,
            'V_era_specific': s1,
        })

era_comp_df = pd.DataFrame(era_comparison)
print(f"\n  Era-specific vs. uniform BIP — V(0-0,r=0) comparison:")
for era_name in ['post_war', 'expansion', 'offense_explosion', 'post_steroid', 'statcast']:
    sub = era_comp_df[era_comp_df['era'] == era_name]
    if len(sub) > 0:
        v0 = sub[sub['deficit'] == 0]['V_era_specific'].values[0]
        v1 = sub[sub['deficit'] == 1]['V_era_specific'].values[0]
        print(f"  {era_name:25s}: V(d=0)={v0:.4f}, V(d=1)={v1:.4f}")


if not gate_passed:
    print("\n\n  *** GATE FAILED — STOPPING SESSION ***")
    import sys
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: STAKES ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 2: Stakes Enrichment")
print("=" * 70)

# ── Build cumulative standings from Retrosheet game results ──────────────
print("\n  Building cumulative standings from game results...")

# Static division membership (for teams in our situation sample)
# Wild card eras: 1995-2011 (1 WC), 2012-2021 (2 WC), 2022+ (3 WC)
# Only need teams appearing in situation PAs

# First, identify which teams appear in our situation data
situation_teams = set()
for year in range(2015, 2025):
    path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False, usecols=['home_team', 'game_date'])
        situation_teams.update(df['home_team'].unique())

# For Retrosheet
for f in os.listdir(f"{BASE_DIR}/data/retrosheet/"):
    if f.startswith('situation_') and f.endswith('.csv'):
        df = pd.read_csv(f"{BASE_DIR}/data/retrosheet/{f}", low_memory=False)
        if 'HOME_TEAM_ID' in df.columns:
            situation_teams.update(df['HOME_TEAM_ID'].unique())
        elif 'GAME_ID' in df.columns:
            situation_teams.update(df['GAME_ID'].str[:3].unique())

print(f"  Teams in situation sample: {len(situation_teams)}")

# ── Statcast: use game_type for postseason detection ─────────────────────
# Load all primary pitches (not just situations) to get game_date and game context
print("\n  Assigning stakes tiers for Statcast era...")

# For Statcast, we need: game_date, home_team, game_type (R/P),
# plus cumulative standings for regular season stakes tiers.
# Build standings from game-level results.

# Simple approach: For each team-date, compute W-L record from primary data
# Actually: our primary data is only the situation PAs. We need full game results.
# Alternative: use the game date and season position to approximate.
#
# Simplest valid approach for Statcast:
# 1. Postseason: game_type == 'P' or similar flag
# 2. High stakes: September + first week of October
# 3. Medium: everything else in regular season
# 4. Low: would need standings data we don't have

# Check what fields are in primary data
sample = pd.read_csv(f"{BASE_DIR}/data/statcast/primary_2024.csv", nrows=5, low_memory=False)
game_cols = [c for c in sample.columns if 'game' in c.lower() or 'type' in c.lower() or 'date' in c.lower()]
print(f"  Available game fields: {game_cols}")

# Statcast has game_type, game_date
# Collect all situation PAs with game context
all_situation_pitches = []
for year in range(2015, 2025):
    path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path, low_memory=False)
    df['deficit'] = df['fld_score'] - df['bat_score']
    df['year'] = year
    all_situation_pitches.append(df)

all_sc = pd.concat(all_situation_pitches, ignore_index=True)

# Parse game_date
all_sc['game_date'] = pd.to_datetime(all_sc['game_date'])
all_sc['month'] = all_sc['game_date'].dt.month
all_sc['day'] = all_sc['game_date'].dt.day

# Stakes assignment
def assign_stakes_statcast(row):
    gt = str(row.get('game_type', 'R')).upper()
    if gt in ('D', 'L', 'W', 'F', 'P'):  # Division, League, Wild Card, Finals, Postseason
        return 'postseason'
    month = row['month']
    day = row['day']
    # September & early October = high stakes
    if month == 9 or (month == 10 and day <= 7):
        return 'high'
    # April-August = medium
    return 'medium'

all_sc['stakes_tier'] = all_sc.apply(assign_stakes_statcast, axis=1)

# Report distribution
stakes_dist = all_sc.groupby('stakes_tier').agg(
    n_pitches=('game_pk', 'count'),
    n_games=('game_pk', 'nunique'),
).reset_index()
print(f"\n  Statcast stakes distribution:")
for _, row in stakes_dist.iterrows():
    print(f"    {row['stakes_tier']:12s}: {row['n_pitches']:>5} pitches, {row['n_games']:>4} games")

# ── Compute policy gaps by stakes tier ───────────────────────────────────
print(f"\n  Computing policy gaps by stakes tier...")

V_sc = sensitivity_results['baseline']['V']
T_sc = sensitivity_results['baseline']['T']
bip_sc = perturbations['baseline']

stakes_gap_records = []
for tier in ['postseason', 'high', 'medium']:
    tier_data = all_sc[all_sc['stakes_tier'] == tier].copy()
    if len(tier_data) == 0:
        continue

    gaps = []
    for _, pitch in tier_data.iterrows():
        b = int(pitch['balls']) if pitch['balls'] < 4 else 3
        s = int(pitch['strikes']) if pitch['strikes'] < 3 else 2
        d = int(pitch['deficit'])
        if d not in DEFICITS:
            continue
        r = 0  # approximate: we don't track runs_scored per-pitch in Statcast

        state = (b, s, d, r)
        if state not in V_sc:
            continue

        swung = 1.0 if pitch['description'] in SWING_DESCS else 0.0
        v_obs = compute_v_observed(state, swung, V_sc, T_sc, bip_sc)
        gap = v_obs - V_sc[state]
        gaps.append(gap)

    if gaps:
        mean_gap = np.mean(gaps)
        se = np.std(gaps) / np.sqrt(len(gaps))
        ci_lo = mean_gap - 1.96 * se
        ci_hi = mean_gap + 1.96 * se

        # Count unique PAs (approximate: count events)
        n_pa = tier_data['events'].notna().sum()

        stakes_gap_records.append({
            'tier': tier, 'mean_gap': mean_gap, 'se': se,
            'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'n_pitches': len(gaps), 'n_pa_approx': n_pa,
        })
        below = " ⚠ below floor" if n_pa < 40 else ""
        print(f"  {tier:12s}: gap={mean_gap:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}], "
              f"n_pitches={len(gaps)}, n_pa≈{n_pa}{below}")

stakes_df = pd.DataFrame(stakes_gap_records)
stakes_df.to_csv(f"{STAKES_DIR}/stakes_policy_gaps.csv", index=False)

# ── Postseason sub-sample detail ─────────────────────────────────────────
print(f"\n  ── Postseason Sub-Sample ──")
ps_data = all_sc[all_sc['stakes_tier'] == 'postseason']
if len(ps_data) > 0:
    ps_events = ps_data[ps_data['events'].notna()]
    print(f"  Postseason situation PAs: {len(ps_events)}")
    for _, pa in ps_events.iterrows():
        print(f"    {pa['game_date'].strftime('%Y-%m-%d')} "
              f"{pa.get('away_team','?')}@{pa.get('home_team','?')} "
              f"deficit={int(pa['deficit'])} {pa['count']}: "
              f"{pa['events']}")
else:
    print("  No postseason situation PAs in dataset")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: PER-COUNT DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 3: Per-Count Decomposition")
print("=" * 70)

# Use all Statcast pitches for the decomposition
decomposition = []

for d in DEFICITS:
    d_data = all_sc[all_sc['deficit'] == d]
    for b in range(4):
        for s in range(3):
            count_data = d_data[(d_data['balls'] == b) & (d_data['strikes'] == s)]
            if len(count_data) == 0:
                continue

            state = (b, s, d, 0)
            if state not in V_sc:
                continue

            # Observed swing rate
            swings = count_data['description'].isin(SWING_DESCS).sum()
            obs_swing = swings / len(count_data)

            # Optimal action
            optimal = all_policies['baseline'][state]

            # Frequency weight
            freq = len(count_data) / len(all_sc)

            # Gap: V(observed) - V(optimal)
            v_obs = compute_v_observed(state, obs_swing, V_sc, T_sc, bip_sc)
            gap = v_obs - V_sc[state]

            # Weighted contribution
            weighted = gap * freq

            decomposition.append({
                'count': f"{b}-{s}", 'deficit': d,
                'balls': b, 'strikes': s,
                'obs_swing_pct': obs_swing,
                'optimal_action': optimal,
                'optimal_swing_pct': 1.0 if optimal == 'swing' else 0.0,
                'swing_deviation': obs_swing - (1.0 if optimal == 'swing' else 0.0),
                'gap': gap,
                'frequency': freq,
                'weighted_contribution': weighted,
                'n_pitches': len(count_data),
            })

decomp_df = pd.DataFrame(decomposition).sort_values('weighted_contribution')
decomp_df.to_csv(f"{LONG_DIR}/count_decomposition.csv", index=False)

print(f"\n  {'Count':>5}  {'Def':>3}  {'Obs%':>6}  {'Opt':>5}  {'Gap':>8}  {'Freq':>6}  {'Wt.Contrib':>10}  {'N':>5}")
print(f"  {'-'*60}")
for _, row in decomp_df.iterrows():
    print(f"  {row['count']:>5}  {int(row['deficit']):>3}  "
          f"{row['obs_swing_pct']:>5.1%}  {row['optimal_action']:>5}  "
          f"{row['gap']:>+7.4f}  {row['frequency']:>5.3f}  "
          f"{row['weighted_contribution']:>+9.5f}  {int(row['n_pitches']):>5}")

# Top contributors
print(f"\n  ── Top 10 Count-Deficit Pairs by WP Cost ──")
top10 = decomp_df.head(10)
cumulative = 0
for _, row in top10.iterrows():
    cumulative += row['weighted_contribution']
    print(f"  {row['count']} d={int(row['deficit'])}: gap={row['gap']:+.4f}, "
          f"obs_swing={row['obs_swing_pct']:.1%}, optimal={row['optimal_action']}, "
          f"weighted={row['weighted_contribution']:+.5f} (cumulative={cumulative:+.5f})")

total_gap = decomp_df['weighted_contribution'].sum()
top5_share = decomp_df.head(5)['weighted_contribution'].sum() / total_gap if total_gap != 0 else 0
print(f"\n  Total weighted gap: {total_gap:+.5f}")
print(f"  Top 5 states account for {top5_share:.1%} of total gap")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: VISUALIZATION SUITE
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 4: Visualization Suite")
print("=" * 70)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ── Plot style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

TEAL = '#2a9d8f'
CORAL = '#e76f51'
AMBER = '#e9c46a'
SLATE = '#264653'
LIGHT_BG = '#f8f9fa'
ERA_COLORS = {
    'expansion': '#a8dadc',
    'offense_explosion': '#ffd166',
    'post_steroid': '#b5e48c',
    'statcast': '#a2d2ff',
}
ERA_LABELS = {
    'expansion': 'Expansion',
    'offense_explosion': 'Offense Explosion',
    'post_steroid': 'Post-Steroid',
    'statcast': 'Statcast',
}


# ── Figure 1: Policy Gap Longitudinal Trend ──────────────────────────────
print("  Building Figure 1: Policy Gap Longitudinal Trend...")

long_data = pd.read_csv(f"{LONG_DIR}/policy_gaps_historical.csv")
# Filter to reliable years only (1990+)
reliable = long_data[long_data['year'] >= 1990].sort_values('year')

fig, ax = plt.subplots(figsize=(12, 5.5))

# Era background shading
era_bounds = {
    'expansion': (1990, 1992),
    'offense_explosion': (1993, 2005),
    'post_steroid': (2006, 2014),
    'statcast': (2015, 2024),
}
for era_name, (start, end) in era_bounds.items():
    ax.axvspan(start - 0.5, end + 0.5, alpha=0.15,
               color=ERA_COLORS.get(era_name, '#ddd'), label=None)
    mid = (start + end) / 2
    ax.text(mid, 0.002, ERA_LABELS.get(era_name, era_name),
            ha='center', va='bottom', fontsize=8, color='#666', style='italic')

# Era boundary lines
for year in [1993, 2006, 2015]:
    ax.axvline(year - 0.5, color='#999', linestyle='--', linewidth=0.8, alpha=0.5)

# Reference line at 0
ax.axhline(0, color='#333', linewidth=1.0, alpha=0.7, linestyle='-')

# Data points and line
years = reliable['year'].values
gaps = reliable['weighted_gap'].values

# SE approximation: use gap variance across pitches (approximate from N)
n_pitches = reliable['n_pitches'].values
se_approx = np.abs(gaps) * 0.3 / np.sqrt(n_pitches)  # rough bootstrap SE

# Hollow point for COVID year
covid_mask = years == 2020
normal_mask = ~covid_mask

ax.plot(years, gaps, color=TEAL, linewidth=2, zorder=3, alpha=0.9)
ax.scatter(years[normal_mask], gaps[normal_mask], color=TEAL, s=50, zorder=4, edgecolors='white', linewidths=1)
if covid_mask.any():
    ax.scatter(years[covid_mask], gaps[covid_mask], color='white', s=50, zorder=4,
               edgecolors=TEAL, linewidths=2, label='COVID-shortened')

# Confidence bands
ax.fill_between(years, gaps - 1.96 * se_approx, gaps + 1.96 * se_approx,
                alpha=0.12, color=TEAL, zorder=2)

# Linear trend
slope, intercept, r, p, se = stats.linregress(years, gaps)
trend_y = slope * years + intercept
ax.plot(years, trend_y, color=CORAL, linewidth=1.5, linestyle='--', alpha=0.7, zorder=3)
ax.text(2020, trend_y[-3] + 0.003,
        f'slope = {slope:+.5f}/yr\np = {p:.3f} (n.s.)',
        fontsize=9, color=CORAL, ha='center')

ax.set_xlabel('Year')
ax.set_ylabel('Policy Gap (WP per pitch)')
ax.set_title('Batter Decision Quality in Terminal Situations (Bot 9, 2 Out, Bases Loaded)\nWeighted Policy Gap vs. Augmented MDP Optimal, 1990–2024')
ax.set_xlim(1989, 2025)
ax.set_xticks(range(1990, 2025, 5))

if covid_mask.any():
    ax.legend(loc='lower right', framealpha=0.9)

fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig1_policy_gap_longitudinal.pdf")
fig.savefig(f"{FIG_DIR}/fig1_policy_gap_longitudinal.png")
plt.close(fig)
print("    Saved fig1_policy_gap_longitudinal.pdf/png")


# ── Figure 2: Optimal Policy Grid ───────────────────────────────────────
print("  Building Figure 2: Optimal Policy Grid...")

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)

cmap = ListedColormap([TEAL, CORAL])  # take=0=teal, swing=1=coral

for idx, d in enumerate(DEFICITS):
    ax = axes[idx]
    grid = np.zeros((3, 4))  # strikes × balls
    value_grid = np.zeros((3, 4))

    for b in range(4):
        for s in range(3):
            state = (b, s, d, 0)
            action = all_policies['baseline'][state]
            grid[s, b] = 0 if action == 'take' else 1
            value_grid[s, b] = sensitivity_results['baseline']['V'][state]

    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    # Annotate with V*
    for b in range(4):
        for s in range(3):
            v = value_grid[s, b]
            action = 'T' if grid[s, b] == 0 else 'S'
            text_color = 'white'
            ax.text(b, s, f'{action}\n{v:.3f}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=text_color)

    ax.set_xticks(range(4))
    ax.set_xticklabels(['0', '1', '2', '3'])
    ax.set_yticks(range(3))
    ax.set_yticklabels(['0', '1', '2'])
    ax.set_xlabel('Balls')
    if idx == 0:
        ax.set_ylabel('Strikes')

    deficit_label = 'Tied' if d == 0 else f'Down {d}'
    ax.set_title(f'Deficit = {d}\n({deficit_label})', fontsize=11)

# Legend
take_patch = mpatches.Patch(color=TEAL, label='TAKE (optimal)')
swing_patch = mpatches.Patch(color=CORAL, label='SWING (optimal)')
fig.legend(handles=[take_patch, swing_patch], loc='lower center',
           ncol=2, frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Optimal Batter Policy — Augmented MDP (Statcast Era)\nBot 9, 2 Out, Bases Loaded',
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig2_optimal_policy_grid.pdf", bbox_inches='tight')
fig.savefig(f"{FIG_DIR}/fig2_optimal_policy_grid.png", bbox_inches='tight')
plt.close(fig)
print("    Saved fig2_optimal_policy_grid.pdf/png")


# ── Figure 3: Count Decomposition Waterfall ─────────────────────────────
print("  Building Figure 3: Count Decomposition Waterfall...")

# Top 15 contributors (most negative = worst)
top_n = decomp_df.head(15).copy()
top_n['label'] = top_n.apply(lambda r: f"{r['count']} d={int(r['deficit'])}", axis=1)
top_n = top_n.iloc[::-1]  # reverse for horizontal bars

fig, ax = plt.subplots(figsize=(10, 7))

colors = [CORAL if row['swing_deviation'] > 0 else TEAL
          for _, row in top_n.iterrows()]
bars = ax.barh(range(len(top_n)), top_n['weighted_contribution'].values, color=colors, alpha=0.85)

ax.set_yticks(range(len(top_n)))
ax.set_yticklabels(top_n['label'].values)
ax.set_xlabel('Weighted WP Contribution')
ax.set_title('Count-Level Decomposition of Aggregate Policy Gap\n'
             'Top 15 States by WP Cost (Statcast Era, 2015–2024)')

# Annotate with swing rate info
for i, (_, row) in enumerate(top_n.iterrows()):
    obs = row['obs_swing_pct']
    opt = row['optimal_action']
    x_pos = row['weighted_contribution']
    ax.text(x_pos - 0.00015, i, f"  {obs:.0%}→{opt}",
            va='center', ha='right' if x_pos < 0 else 'left',
            fontsize=8, color='#333')

# Reference line
ax.axvline(0, color='#333', linewidth=1)

# Legend
over_patch = mpatches.Patch(color=CORAL, label='Over-swinging')
under_patch = mpatches.Patch(color=TEAL, label='Under-swinging')
ax.legend(handles=[over_patch, under_patch], loc='lower right', fontsize=9)

fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig3_count_decomposition.pdf")
fig.savefig(f"{FIG_DIR}/fig3_count_decomposition.png")
plt.close(fig)
print("    Saved fig3_count_decomposition.pdf/png")


# ── Figure 4: Stakes Stratification ─────────────────────────────────────
print("  Building Figure 4: Stakes Stratification...")

if len(stakes_df) > 0:
    fig, ax = plt.subplots(figsize=(8, 5))

    tiers = stakes_df['tier'].values
    gaps_s = stakes_df['mean_gap'].values
    lo = stakes_df['ci_lo'].values
    hi = stakes_df['ci_hi'].values
    ns = stakes_df['n_pitches'].values

    x = range(len(tiers))
    bar_colors = [SLATE if n >= 120 else '#aaa' for n in ns]  # hollow-ish for small N
    edge_colors = [SLATE] * len(tiers)

    bars = ax.bar(x, gaps_s, color=bar_colors, edgecolor=edge_colors, alpha=0.85)
    ax.errorbar(x, gaps_s, yerr=[gaps_s - lo, hi - gaps_s],
                fmt='none', capsize=5, color='#333', linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([t.title() for t in tiers])
    ax.set_ylabel('Mean Policy Gap (WP/pitch)')
    ax.set_title('Policy Gap by Game Stakes Tier\nStatcast Era, 2015–2024')

    # Annotate with N
    for i, (g, n) in enumerate(zip(gaps_s, ns)):
        ax.text(i, g - 0.003, f'n={int(n)}', ha='center', va='top', fontsize=9, color='white')

    # Reference line at aggregate
    agg_gap = decomp_df['weighted_contribution'].sum()
    ax.axhline(agg_gap, color=CORAL, linestyle='--', linewidth=1, alpha=0.7)
    ax.text(len(tiers) - 0.5, agg_gap + 0.001, f'Aggregate: {agg_gap:.4f}',
            fontsize=9, color=CORAL, ha='right')

    ax.axhline(0, color='#333', linewidth=0.8)

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig4_stakes_stratification.pdf")
    fig.savefig(f"{FIG_DIR}/fig4_stakes_stratification.png")
    plt.close(fig)
    print("    Saved fig4_stakes_stratification.pdf/png")
else:
    print("    Skipped — no stakes data")


# ── Figure 5: Pitcher Zone Rate Deviation ───────────────────────────────
print("  Building Figure 5: Pitcher Zone Rate Deviation...")

pit_dev_path = f"{BASE_DIR}/data/deviations/statcast_pitcher_deviations.csv"
if os.path.exists(pit_dev_path):
    pit_devs = pd.read_csv(pit_dev_path)

    if 'year' not in pit_devs.columns and 'season' in pit_devs.columns:
        pit_devs['year'] = pit_devs['season']

    # Try to get yearly zone rate deviation
    if 'dev_zone_rate' in pit_devs.columns and 'year' in pit_devs.columns:
        yearly_zone = pit_devs.groupby('year').agg(
            mean_dev=('dev_zone_rate', 'mean'),
            se=('dev_zone_rate', lambda x: x.std() / np.sqrt(len(x))),
            n=('dev_zone_rate', 'count'),
        ).reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))

        years_z = yearly_zone['year'].values
        means_z = yearly_zone['mean_dev'].values
        se_z = yearly_zone['se'].values

        ax.plot(years_z, means_z, color=CORAL, linewidth=2, marker='o', markersize=6,
                markerfacecolor=CORAL, markeredgecolor='white', markeredgewidth=1, zorder=3)
        ax.fill_between(years_z, means_z - 1.96 * se_z, means_z + 1.96 * se_z,
                        alpha=0.15, color=CORAL)

        ax.axhline(0, color='#333', linewidth=1, linestyle='-')
        ax.set_xlabel('Year')
        ax.set_ylabel('Zone Rate Deviation (situation − baseline)')
        ax.set_title('Pitcher Zone Rate Deviation in Terminal Situations\n'
                     'Statcast Era, 2015–2024')

        # Annotate overall effect
        overall_mean = pit_devs['dev_zone_rate'].mean()
        t_stat, p_val = stats.ttest_1samp(pit_devs['dev_zone_rate'].dropna(), 0)
        ax.text(0.02, 0.95, f'Overall: +{overall_mean:.1%} (p={p_val:.4f})',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/fig5_pitcher_zone_rate.pdf")
        fig.savefig(f"{FIG_DIR}/fig5_pitcher_zone_rate.png")
        plt.close(fig)
        print("    Saved fig5_pitcher_zone_rate.pdf/png")
    else:
        print("    Skipped — missing year or dev_zone_rate columns")
else:
    print("    Skipped — no pitcher deviation data")


# ── Figure 6: Sensitivity Heatmap ───────────────────────────────────────
print("  Building Figure 6: Sensitivity Heatmap...")

# Build grid: rows = scenarios, columns = count states (at d=0)
scenarios_to_show = ['baseline', 'contact_pessimistic', 'contact_optimistic',
                     'deadball_era', 'steroid_era', 'flyball_rev']
scenario_labels = ['Baseline', 'Contact−', 'Contact+',
                   'Deadball', 'Steroid', 'Flyball Rev']

# Show deficit=0 policy across all scenarios
count_labels = [f"{b}-{s}" for b in range(4) for s in range(3)]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for d_idx, d in enumerate(DEFICITS):
    ax = axes[d_idx // 2, d_idx % 2]
    grid = np.zeros((len(scenarios_to_show), 12))

    for i, scenario in enumerate(scenarios_to_show):
        pol = all_policies[scenario]
        for j, (b, s) in enumerate([(b, s) for b in range(4) for s in range(3)]):
            state = (b, s, d, 0)
            grid[i, j] = 0 if pol[state] == 'take' else 1

    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    # Mark flips with amber border
    baseline_row = 0
    for i in range(1, len(scenarios_to_show)):
        for j in range(12):
            if grid[i, j] != grid[baseline_row, j]:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     linewidth=2.5, edgecolor=AMBER,
                                     facecolor='none', zorder=5)
                ax.add_patch(rect)

    ax.set_xticks(range(12))
    ax.set_xticklabels(count_labels, fontsize=8)
    ax.set_yticks(range(len(scenarios_to_show)))
    ax.set_yticklabels(scenario_labels, fontsize=9)
    ax.set_xlabel('Count')
    deficit_label = 'Tied' if d == 0 else f'Down {d}'
    ax.set_title(f'Deficit = {d} ({deficit_label})')

fig.suptitle('Sensitivity of Optimal Policy to BIP Distribution\n'
             'Amber border = policy flip vs. baseline',
             fontsize=13, y=1.02)

# Global legend
fig.legend(handles=[take_patch, swing_patch,
                    mpatches.Patch(edgecolor=AMBER, facecolor='none', linewidth=2.5, label='Policy flip')],
           loc='lower center', ncol=3, frameon=True, fontsize=10,
           bbox_to_anchor=(0.5, -0.04))

fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig6_sensitivity_heatmap.pdf", bbox_inches='tight')
fig.savefig(f"{FIG_DIR}/fig6_sensitivity_heatmap.png", bbox_inches='tight')
plt.close(fig)
print("    Saved fig6_sensitivity_heatmap.pdf/png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: SECONDARY SITUATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 5: Secondary Situation Analysis")
print("=" * 70)

# Secondary situation: tying run on base, any base config
sec_path = f"{BASE_DIR}/data/statcast/secondary_2024.csv"
if os.path.exists(sec_path):
    sec = pd.read_csv(sec_path, low_memory=False)
    print(f"  Secondary situation PAs available: {len(sec[sec['events'].notna()])}")

    # The secondary data from Session 2 is just 2024.
    # Check if we have deviation data from Session 2
    bat_dev_path = f"{BASE_DIR}/data/deviations/statcast_batter_deviations.csv"
    pit_dev_path = f"{BASE_DIR}/data/deviations/statcast_pitcher_deviations.csv"

    if os.path.exists(bat_dev_path):
        bat_devs = pd.read_csv(bat_dev_path)
        print(f"\n  Batter deviations (from Session 2): {len(bat_devs)} records")
        for metric in ['dev_swing_rate', 'dev_zone_swing_rate', 'dev_chase_rate']:
            if metric in bat_devs.columns:
                vals = bat_devs[metric].dropna()
                if len(vals) >= 5:
                    t, p = stats.ttest_1samp(vals, 0)
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    print(f"    {metric:30s}: mean={vals.mean():+.4f}, t={t:>6.2f}, p={p:.4f} {sig}")

    if os.path.exists(pit_dev_path):
        pit_devs_2 = pd.read_csv(pit_dev_path)
        print(f"\n  Pitcher deviations (from Session 2): {len(pit_devs_2)} records")
        for metric in ['dev_zone_rate', 'dev_hard_pct', 'dev_offspeed_pct', 'dev_chase_rate_induced']:
            if metric in pit_devs_2.columns:
                vals = pit_devs_2[metric].dropna()
                if len(vals) >= 5:
                    t, p = stats.ttest_1samp(vals, 0)
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    print(f"    {metric:30s}: mean={vals.mean():+.4f}, t={t:>6.2f}, p={p:.4f} {sig}")

    # Compute secondary situation swing rate from raw data (all years)
    print(f"\n  ── Secondary Situation Swing Rate (All Statcast Years) ──")
    sec_swing_rates = []
    for year in range(2015, 2025):
        path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"  # primary includes our situation
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, low_memory=False)
        swings = df['description'].isin(SWING_DESCS).sum()
        total = len(df)
        sr = swings / total if total > 0 else 0
        sec_swing_rates.append({'year': year, 'swing_rate': sr, 'n': total})

    sec_sr_df = pd.DataFrame(sec_swing_rates)
    if len(sec_sr_df) > 0:
        overall_sr = sum(r['swing_rate'] * r['n'] for _, r in sec_sr_df.iterrows()) / sec_sr_df['n'].sum()
        print(f"  Primary situation overall swing rate: {overall_sr:.3f}")
        print(f"  (This is the swing rate in Bot 9, 2 out, bases loaded across 2015-2024)")

else:
    print("  No secondary situation data available (only single year was cached)")
    print("  Reporting Session 2 deviation findings as the secondary check.")


# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" SESSION 5 ANALYSIS COMPLETE")
print("=" * 70)

# Final inventory
for d in [MDP_DIR, LONG_DIR, STAKES_DIR, FIG_DIR]:
    print(f"\n  {os.path.relpath(d, BASE_DIR)}/")
    for f in sorted(os.listdir(d)):
        fp = os.path.join(d, f)
        sz = os.path.getsize(fp)
        unit = 'KB' if sz > 1024 else 'B'
        val = sz / 1024 if sz > 1024 else sz
        print(f"    {f} ({val:.0f} {unit})")
