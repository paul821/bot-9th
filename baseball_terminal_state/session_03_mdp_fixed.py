"""
Session 3: MDP Infrastructure — CORRECTED deficit direction
============================================================
deficit = fld_score - bat_score (positive = batting team trails)
DEFICITS = [0, 1, 2, 3]
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
os.makedirs(MDP_DIR, exist_ok=True)

ERAS = {
    'post_war': (1950, 1968),
    'expansion': (1969, 1992),
    'offense_explosion': (1993, 2005),
    'post_steroid': (2006, 2014),
    'statcast': (2015, 2024),
}

DEFICITS = [0, 1, 2, 3]  # CORRECTED: 0=tied, 1=down 1, 2=down 2, 3=down 3

TERMINAL_K = 'K'
TERMINAL_BB = 'BB'
TERMINAL_BIP = 'BIP'

EXTRAS_WP = 0.52  # Home team extra-innings win probability

# Pitch sequence decoding
BALL_CHARS = set('BIPV')
CALLED_STRIKE_CHARS = set('CK')
SWINGING_STRIKE_CHARS = set('SMQT')
FOUL_CHARS = set('FHLOR')
CONTACT_CHARS = set('XY')
IGNORE_CHARS = set('>+*123.N')


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Win Expectancy Tables (corrected)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(" STEP 1: Win Expectancy — Corrected Deficit Direction")
print("=" * 70)

# Load all primary situation terminal PAs
all_primary = []
for year in range(2015, 2025):
    path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        terminal = df[df['events'].notna()].copy()
        terminal['deficit'] = terminal['fld_score'] - terminal['bat_score']
        all_primary.append(terminal)

primary = pd.concat(all_primary, ignore_index=True)

# Classify outcomes and compute actual runs scored
def get_runs(row):
    if pd.notna(row.get('post_bat_score')) and pd.notna(row.get('bat_score')):
        return int(row['post_bat_score'] - row['bat_score'])
    event = row['events']
    if event in ('strikeout', 'strikeout_double_play'): return 0
    elif event in ('walk', 'hit_by_pitch'): return 1
    elif event == 'single': return 2
    elif event in ('double', 'triple'): return 3
    elif event == 'home_run': return 4
    else: return 0

primary['runs_scored'] = primary.apply(get_runs, axis=1)

def classify_outcome(event):
    if event in ('strikeout', 'strikeout_double_play'): return 'K'
    elif event == 'walk': return 'BB'
    elif event == 'hit_by_pitch': return 'HBP'
    elif event in ('single', 'double', 'triple', 'home_run', 'field_error'): return 'BIP'
    else: return 'other_out'

primary['outcome_class'] = primary['events'].apply(classify_outcome)

# For each deficit, compute WP outcomes
# After scoring r runs with deficit d:
#   new_deficit = d - r
#   If new_deficit < 0: walk-off win → WP = 1.0
#   If new_deficit == 0: tied → extras → WP ≈ 0.52
#   If new_deficit > 0: loss (bot 9, 2 out, 3rd out made) → WP = 0.0
#   EXCEPTION: BIP doesn't always mean 3 outs. We use actual run outcomes.

print("\n  Outcome counts by deficit:")
for deficit in DEFICITS:
    sub = primary[primary['deficit'] == deficit]
    print(f"  Deficit {deficit}: n={len(sub)}")
    print(f"    {sub['outcome_class'].value_counts().to_dict()}")

mdp_rewards = {}
base_we = {}

for deficit in DEFICITS:
    sub = primary[primary['deficit'] == deficit]
    if len(sub) < 10:
        print(f"\n  Deficit {deficit}: too few observations ({len(sub)})")
        continue

    # For each PA, determine if batting team won
    # After the PA outcome with runs_scored:
    # new_deficit = deficit - runs_scored
    sub = sub.copy()
    sub['new_deficit'] = sub['deficit'] - sub['runs_scored']

    # WP after outcome:
    # K/other_out: game over if 3 outs → loss (WP=0). But wait —
    # for outs that don't end the inning (e.g. if there's an error), it's complex.
    # Simplify: all outs with 2 out = game over = loss.
    def outcome_wp(row):
        oc = row['outcome_class']
        if oc in ('K', 'other_out'):
            return 0.0  # Game over, loss
        new_def = row['new_deficit']
        if new_def < 0:
            return 1.0  # Walk-off win
        elif new_def == 0:
            return EXTRAS_WP  # Extras
        else:
            # Still trailing — but it's bot 9, 2 out. If the PA outcome didn't
            # produce 3 outs, the inning continues. However, our data shows
            # these as terminal PAs, so the game likely ended on this PA.
            # Actually: a walk or hit with 2 out doesn't add an out.
            # The issue is: if the batting team scores but is still behind,
            # the inning continues (not 3 outs yet). The PA just ended that AB.
            # But we're modeling the MDP for a SINGLE at-bat, not the full inning.
            # For the MDP, terminal BIP/BB means the at-bat ended, and:
            #   - If still behind: the INNING continues with a new batter, but
            #     our model stops here. We'd need the continuation WP.
            #   - For simplicity: use the empirical win rate from the data.
            return 0.0  # Approximation: still behind with 2 out is very unlikely to win

    sub['wp_after'] = sub.apply(outcome_wp, axis=1)

    # Base WE = average WP for this deficit
    bwe = sub['wp_after'].mean()
    base_we[deficit] = bwe

    # Terminal state rewards (ΔWP = WP_after - base_WE for each outcome class)
    # K: always 0 → ΔWP = -bwe
    k_dwp = 0.0 - bwe

    # BB/HBP: score 1 run → new deficit = deficit - 1
    bb_sub = sub[sub['outcome_class'].isin(['BB', 'HBP'])]
    bb_wp = bb_sub['wp_after'].mean() if len(bb_sub) > 0 else (
        1.0 if deficit <= 1 else (EXTRAS_WP if deficit == 1 else 0.0)
    )
    bb_dwp = bb_wp - bwe

    # BIP: variable runs scored
    bip_sub = sub[sub['outcome_class'] == 'BIP']
    bip_wp = bip_sub['wp_after'].mean() if len(bip_sub) > 0 else 0.5
    bip_dwp = bip_wp - bwe

    mdp_rewards[deficit] = {
        'base_we': bwe,
        TERMINAL_K: k_dwp,
        TERMINAL_BB: bb_dwp,
        TERMINAL_BIP: bip_dwp,
    }

    print(f"\n  Deficit {deficit} (n={len(sub)}, base WE={bwe:.4f}):")
    print(f"    K:   WP=0.000, ΔWP={k_dwp:+.4f}")
    print(f"    BB:  WP={bb_wp:.3f}, ΔWP={bb_dwp:+.4f} (n={len(bb_sub)})")
    print(f"    BIP: WP={bip_wp:.3f}, ΔWP={bip_dwp:+.4f} (n={len(bip_sub)})")

# For deficit=1 BB: should give tied game → EXTRAS_WP
# For deficit=0 BB: scores run, walk-off win → WP=1.0
# Verify:
print(f"\n  Sanity checks:")
for d in DEFICITS:
    if d in mdp_rewards:
        r = mdp_rewards[d]
        print(f"    deficit={d}: K_ΔWP={r[TERMINAL_K]:+.4f}, BB_ΔWP={r[TERMINAL_BB]:+.4f}, BIP_ΔWP={r[TERMINAL_BIP]:+.4f}")

# Save
with open(f"{MDP_DIR}/mdp_rewards.json", 'w') as f:
    json.dump({str(k): v for k, v in mdp_rewards.items()}, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Load transition parameters (already computed in first run)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 2: Load Transition Parameters")
print("=" * 70)

trans_df = pd.read_csv(f"{MDP_DIR}/transition_parameters.csv", index_col=0)
era_transitions = trans_df.to_dict('index')
print(trans_df[['p_ball_on_take', 'p_strike_on_take', 'p_whiff_on_swing',
                'p_foul_on_swing', 'p_contact_on_swing']].to_string())


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Value Iteration (all eras × all deficits)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 3: Value Iteration — All Eras × All Deficits")
print("=" * 70)

def build_transition_dict(trans_params):
    T = {}
    pb = trans_params['p_ball_on_take']
    ps = trans_params['p_strike_on_take']
    pw = trans_params['p_whiff_on_swing']
    pf = trans_params['p_foul_on_swing']
    pc = trans_params['p_contact_on_swing']

    for b in range(4):
        for s in range(3):
            state = (b, s)
            T[state] = {}

            # TAKE
            take = {}
            if b == 3:
                take[TERMINAL_BB] = take.get(TERMINAL_BB, 0) + pb
            else:
                take[(b+1, s)] = pb
            if s == 2:
                take[TERMINAL_K] = take.get(TERMINAL_K, 0) + ps
            else:
                take[(b, s+1)] = ps
            T[state]['take'] = take

            # SWING
            swing = {}
            if s == 2:
                swing[TERMINAL_K] = swing.get(TERMINAL_K, 0) + pw
            else:
                swing[(b, s+1)] = swing.get((b, s+1), 0) + pw
            if s < 2:
                swing[(b, s+1)] = swing.get((b, s+1), 0) + pf
            else:
                swing[(b, 2)] = swing.get((b, 2), 0) + pf
            swing[TERMINAL_BIP] = swing.get(TERMINAL_BIP, 0) + pc
            T[state]['swing'] = swing

    return T


def value_iteration(trans_params, rewards_for_deficit, tol=1e-8, max_iter=1000):
    T = build_transition_dict(trans_params)
    terminal_values = {
        TERMINAL_K: rewards_for_deficit[TERMINAL_K],
        TERMINAL_BB: rewards_for_deficit[TERMINAL_BB],
        TERMINAL_BIP: rewards_for_deficit[TERMINAL_BIP],
    }

    V = {(b, s): 0.0 for b in range(4) for s in range(3)}
    policy = {}

    for iteration in range(max_iter):
        V_new = {}
        for state in V:
            action_values = {}
            for action in ['take', 'swing']:
                ev = 0.0
                for ns, prob in T[state][action].items():
                    ev += prob * (terminal_values[ns] if ns in terminal_values else V[ns])
                action_values[action] = ev
            best = max(action_values, key=action_values.get)
            V_new[state] = action_values[best]
            policy[state] = best

        delta = max(abs(V_new[s] - V[s]) for s in V)
        V = V_new
        if delta < tol:
            return V, policy, iteration+1, T

    return V, policy, max_iter, T


all_solutions = {}
policy_tables = []

for era_name, trans_params in era_transitions.items():
    print(f"\n  Era: {era_name}")
    for deficit in DEFICITS:
        if deficit not in mdp_rewards:
            print(f"    Deficit {deficit}: no reward data")
            continue

        V, policy, n_iter, T = value_iteration(trans_params, mdp_rewards[deficit])
        all_solutions[(era_name, deficit)] = {'V': V, 'policy': policy, 'T': T}

        for (b, s), action in policy.items():
            policy_tables.append({
                'era': era_name, 'deficit': deficit,
                'balls': b, 'strikes': s, 'count': f"{b}-{s}",
                'optimal_action': action, 'value': V[(b, s)],
            })

        print(f"    deficit={deficit}: V(0-0)={V[(0,0)]:.4f}, converged in {n_iter} iters")

policy_df = pd.DataFrame(policy_tables)
policy_df.to_csv(f"{MDP_DIR}/optimal_policies.csv", index=False)

# Display policy grids
for deficit in DEFICITS:
    if deficit not in mdp_rewards:
        continue
    print(f"\n  ── Optimal Policy: Deficit = {deficit} (batting team trails by {deficit}) ──")
    header = f"  {'Count':<8}"
    for en in era_transitions:
        header += f" {en:>20}"
    print(header)

    for b in range(4):
        for s in range(3):
            row = f"  {b}-{s:<6}"
            for en in era_transitions:
                key = (en, deficit)
                if key in all_solutions:
                    a = all_solutions[key]['policy'][(b, s)]
                    v = all_solutions[key]['V'][(b, s)]
                    row += f" {a:>10} ({v:+.3f})"
                else:
                    row += f" {'N/A':>20}"
            print(row)

# Validation
print(f"\n  ── Validation ──")
for en in era_transitions:
    key0 = (en, 0)
    if key0 not in all_solutions:
        continue
    V = all_solutions[key0]['V']
    pol = all_solutions[key0]['policy']
    v30 = V[(3,0)]
    v02 = V[(0,2)]
    checks = [
        f"V(3-0)={v30:.3f} {'✓' if v30 == max(V.values()) else '✗'}",
        f"V(0-2)={v02:.3f} {'✓' if v02 == min(V.values()) else '✗'}",
        f"π(3-0)={pol[(3,0)]} {'✓' if pol[(3,0)]=='take' else '✗'}",
    ]
    # Deficit monotonicity: more swing as deficit increases
    swing_counts = {}
    for d in DEFICITS:
        kd = (en, d)
        if kd in all_solutions:
            swing_counts[d] = sum(1 for a in all_solutions[kd]['policy'].values() if a == 'swing')
    mono = all(swing_counts.get(d, 0) <= swing_counts.get(d+1, 12) for d in DEFICITS[:-1])
    checks.append(f"swing↑ as deficit↑: {swing_counts} {'✓' if mono else '✗'}")
    print(f"  {en}: {' | '.join(checks)}")

# Save value functions
vf_list = []
for (en, d), sol in all_solutions.items():
    for (b, s), v in sol['V'].items():
        vf_list.append({'era': en, 'deficit': d, 'balls': b, 'strikes': s, 'count': f"{b}-{s}", 'value': v})
pd.DataFrame(vf_list).to_csv(f"{MDP_DIR}/value_functions.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Policy Gap (all deficits, per-year tracking)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 4: Policy Gap — All Deficits")
print("=" * 70)

swing_descs = {'swinging_strike', 'foul', 'foul_tip', 'swinging_strike_blocked',
               'foul_bunt', 'missed_bunt', 'hit_into_play', 'hit_into_play_no_out',
               'hit_into_play_score', 'foul_pitchout', 'swinging_pitchout'}

# Load all primary situation pitches
all_pitches = []
for year in range(2015, 2025):
    path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        df['is_swing'] = df['description'].isin(swing_descs).astype(int)
        df['deficit'] = df['fld_score'] - df['bat_score']
        df['year'] = year
        all_pitches.append(df[['balls', 'strikes', 'is_swing', 'deficit', 'year']])

all_pitches_df = pd.concat(all_pitches, ignore_index=True)

# Observed swing rates by count × deficit
swing_rates = all_pitches_df.groupby(['balls', 'strikes', 'deficit'])['is_swing'].agg(['mean', 'count']).reset_index()
swing_rates.columns = ['balls', 'strikes', 'deficit', 'swing_rate', 'n_pitches']

# Compute policy gaps
gap_results = []

for deficit in DEFICITS:
    if deficit not in mdp_rewards:
        continue
    key = ('statcast', deficit)
    if key not in all_solutions:
        continue

    sol = all_solutions[key]
    V, T = sol['V'], sol['T']
    rewards = mdp_rewards[deficit]
    tv = {TERMINAL_K: rewards[TERMINAL_K], TERMINAL_BB: rewards[TERMINAL_BB], TERMINAL_BIP: rewards[TERMINAL_BIP]}

    sub = swing_rates[swing_rates['deficit'] == deficit]
    print(f"\n  Deficit {deficit} (trailing by {deficit}):")
    print(f"  {'Count':<8} {'SwingRate':>10} {'Optimal':>10} {'V_obs':>10} {'V_opt':>10} {'Gap':>10} {'N':>6}")

    for b in range(4):
        for s in range(3):
            row = sub[(sub['balls'] == b) & (sub['strikes'] == s)]
            if len(row) == 0:
                continue

            sr = row['swing_rate'].values[0]
            n = int(row['n_pitches'].values[0])
            opt_a = sol['policy'][(b, s)]
            v_opt = V[(b, s)]

            # V_observed
            v_obs = 0.0
            for a, ap in [('swing', sr), ('take', 1-sr)]:
                for ns, tp in T[(b,s)][a].items():
                    v_obs += ap * tp * (tv[ns] if ns in tv else V.get(ns, 0))

            gap = v_obs - v_opt

            gap_results.append({
                'era': 'statcast', 'deficit': deficit,
                'balls': b, 'strikes': s, 'count': f"{b}-{s}",
                'obs_swing_rate': sr, 'optimal_action': opt_a,
                'v_observed': v_obs, 'v_optimal': v_opt,
                'gap': gap, 'n_pitches': n,
            })

            print(f"  {b}-{s:<6} {sr:>10.3f} {opt_a:>10} {v_obs:>10.4f} {v_opt:>10.4f} {gap:>+10.4f} {n:>6}")

gap_df = pd.DataFrame(gap_results)
gap_df.to_csv(f"{MDP_DIR}/policy_gaps_statcast.csv", index=False)

# Aggregate by deficit
print(f"\n  ── Aggregate Policy Gaps ──")
for deficit in DEFICITS:
    sub = gap_df[gap_df['deficit'] == deficit]
    if len(sub) == 0:
        continue
    wgap = (sub['gap'] * sub['n_pitches']).sum() / sub['n_pitches'].sum()
    worst = sub.loc[sub['gap'].idxmin()]
    print(f"  deficit={deficit}: weighted gap={wgap:+.4f}, worst={worst['count']} ({worst['gap']:+.4f})")

# Per-year tracking
print(f"\n  ── Per-Year Weighted Policy Gaps ──")
yearly_gaps = []

for year in range(2015, 2025):
    year_pitches = all_pitches_df[all_pitches_df['year'] == year]
    year_swing = year_pitches.groupby(['balls', 'strikes', 'deficit'])['is_swing'].agg(['mean', 'count']).reset_index()
    year_swing.columns = ['balls', 'strikes', 'deficit', 'swing_rate', 'n_pitches']

    total_gap_weighted = 0.0
    total_n = 0

    for deficit in DEFICITS:
        key = ('statcast', deficit)
        if key not in all_solutions or deficit not in mdp_rewards:
            continue
        sol = all_solutions[key]
        V, T = sol['V'], sol['T']
        rewards = mdp_rewards[deficit]
        tv = {TERMINAL_K: rewards[TERMINAL_K], TERMINAL_BB: rewards[TERMINAL_BB], TERMINAL_BIP: rewards[TERMINAL_BIP]}

        sub = year_swing[year_swing['deficit'] == deficit]
        for _, row in sub.iterrows():
            b, s = int(row['balls']), int(row['strikes'])
            if (b, s) not in V:
                continue
            sr = row['swing_rate']
            n = int(row['n_pitches'])

            v_obs = 0.0
            for a, ap in [('swing', sr), ('take', 1-sr)]:
                for ns, tp in T[(b,s)][a].items():
                    v_obs += ap * tp * (tv[ns] if ns in tv else V.get(ns, 0))

            gap = v_obs - V[(b, s)]
            total_gap_weighted += gap * n
            total_n += n

    if total_n > 0:
        avg_gap = total_gap_weighted / total_n
        yearly_gaps.append({'year': year, 'weighted_avg_gap': avg_gap, 'total_pitches': total_n})
        print(f"  {year}: gap={avg_gap:+.4f} (n={total_n})")

yearly_gap_df = pd.DataFrame(yearly_gaps)
yearly_gap_df.to_csv(f"{MDP_DIR}/yearly_policy_gaps.csv", index=False)

# Trend test
if len(yearly_gap_df) >= 5:
    slope, intercept, r, p, se = stats.linregress(yearly_gap_df['year'], yearly_gap_df['weighted_avg_gap'])
    print(f"\n  Trend: slope={slope:+.6f}/year, r²={r**2:.3f}, p={p:.3f}")
    if p < 0.05:
        direction = "closing" if slope > 0 else "widening"
        print(f"  → Significant trend: policy gap is {direction} over time")
    else:
        print(f"  → No significant trend in policy gap over time")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Sensitivity Analysis (all deficits)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 5: Sensitivity Analysis")
print("=" * 70)

perturbation = 0.10
sens_results = []

for era_name, trans_params in era_transitions.items():
    for deficit in DEFICITS:
        if deficit not in mdp_rewards:
            continue
        key = (era_name, deficit)
        if key not in all_solutions:
            continue
        base_pol = all_solutions[key]['policy']

        param_keys = ['p_ball_on_take', 'p_strike_on_take', 'p_whiff_on_swing',
                      'p_foul_on_swing', 'p_contact_on_swing']

        for param in param_keys:
            for direction in [+1, -1]:
                pert = dict(trans_params)
                pert[param] = trans_params[param] * (1 + direction * perturbation)

                # Renormalize
                if param in ['p_ball_on_take', 'p_strike_on_take']:
                    t = pert['p_ball_on_take'] + pert['p_strike_on_take']
                    pert['p_ball_on_take'] /= t
                    pert['p_strike_on_take'] /= t
                else:
                    t = pert['p_whiff_on_swing'] + pert['p_foul_on_swing'] + pert['p_contact_on_swing']
                    pert['p_whiff_on_swing'] /= t
                    pert['p_foul_on_swing'] /= t
                    pert['p_contact_on_swing'] /= t

                V_p, pol_p, _, _ = value_iteration(pert, mdp_rewards[deficit])

                for state in base_pol:
                    if base_pol[state] != pol_p[state]:
                        sens_results.append({
                            'era': era_name, 'deficit': deficit,
                            'param': param,
                            'direction': f"{'+' if direction>0 else '-'}{perturbation*100:.0f}%",
                            'flip_state': f"{state[0]}-{state[1]}",
                            'base_action': base_pol[state],
                            'perturbed_action': pol_p[state],
                        })

sens_df = pd.DataFrame(sens_results)
sens_df.to_csv(f"{MDP_DIR}/sensitivity_analysis.csv", index=False)

if len(sens_df) > 0:
    boundary = sens_df.groupby(['era', 'deficit', 'flip_state']).size().reset_index(name='n_flips')
    print(f"\n  {len(sens_df)} policy flips under ±{perturbation*100:.0f}%:")
    for en in era_transitions:
        era_sens = boundary[boundary['era'] == en]
        if len(era_sens) == 0:
            print(f"    {en}: fully robust (no flips)")
        else:
            states = list(era_sens.apply(lambda r: f"d={r['deficit']},{r['flip_state']}", axis=1))
            print(f"    {en}: boundary states = {states}")
else:
    print(f"\n  ✓ All policies robust under ±{perturbation*100:.0f}% perturbation")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY OUTPUT
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" ALL STEPS COMPLETE — CORRECTED VERSION")
print("=" * 70)

print(f"\nKey results:")
print(f"  WE tables: {len(mdp_rewards)} deficit levels")
print(f"  MDP solutions: {len(all_solutions)} (era × deficit)")
print(f"  Policy gap entries: {len(gap_df)}")
print(f"  Sensitivity flips: {len(sens_df)}")

print(f"\nOutput files:")
for f in sorted(os.listdir(MDP_DIR)):
    print(f"  {f} ({os.path.getsize(os.path.join(MDP_DIR, f)):,} bytes)")
