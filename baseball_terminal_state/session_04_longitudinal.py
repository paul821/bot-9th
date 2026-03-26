"""
Session 4: Historical Pull & Longitudinal Analysis
===================================================

Step 1: Augmented MDP (balls, strikes, deficit, runs_scored_this_inning)
Step 2: Inning run counter for Retrosheet
Step 3: Historical Retrosheet pull
Step 4: Era-level policy gaps
Step 5: 3-0 paradox time series
Step 6: CUSUM changepoint detection
Step 7: Pitcher accountability analysis
"""

import pandas as pd
import numpy as np
import os
import json
import glob
import subprocess
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

BASE_DIR = "/Users/paul821/Desktop/baseball/baseball_terminal_state"
MDP_DIR = f"{BASE_DIR}/data/mdp"
LONG_DIR = f"{BASE_DIR}/data/longitudinal"
RETRO_DIR = f"{BASE_DIR}/retrosheet_data"
os.makedirs(LONG_DIR, exist_ok=True)

# ── Constants ───────────────────────────────────────────────────────────
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

DEFICITS = [0, 1, 2, 3]  # positive = batting team trails
EXTRAS_WP = 0.52

# Transition matrix assignment: year → source year
def get_transition_source(year):
    if year <= 1964: return 1960
    elif year <= 1984: return 1975
    elif year <= 1994: return 1990
    elif year <= 2009: return 2000
    elif year <= 2014: return 2010
    else: return 'statcast'

MIN_PA_FLOOR = 40

TERMINAL_K = 'K'
TERMINAL_BB = 'BB'
TERMINAL_BIP = 'BIP'

# Pitch sequence decoding
BALL_CHARS = set('BIPV')
CALLED_STRIKE_CHARS = set('CK')
SWINGING_STRIKE_CHARS = set('SMQT')
FOUL_CHARS = set('FHLOR')
CONTACT_CHARS = set('XY')
IGNORE_CHARS = set('>+*123.N')

# Statcast swing descriptions
SWING_DESCS = {'swinging_strike', 'foul', 'foul_tip', 'swinging_strike_blocked',
               'foul_bunt', 'missed_bunt', 'hit_into_play', 'hit_into_play_no_out',
               'hit_into_play_score', 'foul_pitchout', 'swinging_pitchout'}


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Augmented MDP
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(" STEP 1: Augmented MDP (balls, strikes, deficit, runs_scored)")
print("=" * 70)

# ── BIP run distribution (from Statcast primary data) ───────────────────
# How many runs does a ball-in-play score with bases loaded?
bip_runs_dist = {}
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

# BIP outcomes: single, double, triple, HR, error
bip_events = {'single', 'double', 'triple', 'home_run', 'field_error'}
bip_data = primary_all[primary_all['events'].isin(bip_events)]
bip_runs_dist = bip_data['runs_scored'].value_counts(normalize=True).sort_index().to_dict()
print(f"  BIP runs distribution (n={len(bip_data)}):")
for r, p in sorted(bip_runs_dist.items()):
    print(f"    {r} runs: {p:.3f}")
print(f"  Mean BIP runs: {bip_data['runs_scored'].mean():.2f}")


def build_augmented_transition_dict(trans_params):
    """
    Build transition dict for augmented state space.
    State: (balls, strikes, deficit, runs_scored_this_inning)

    Transitions:
    - Take+ball: (b+1, s, d, r) or BB terminal
    - Take+strike: (b, s+1, d, r) or K terminal
    - Swing+whiff: (b, s+1, d, r) or K terminal
    - Swing+foul: (b, min(s+1,2), d, r) — fouls don't advance past 2 strikes
    - Swing+contact: BIP terminal

    BB terminal at deficit=0: walk-off win (reward = 1 - base_we)
    BB terminal at deficit>0: next batter starts at (0, 0, d-1, r+1)
    BIP terminal: probabilistic run scoring from bip_runs_dist
    K terminal: game over, loss
    """
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

                    # ── TAKE ──
                    take = {}
                    # Ball
                    if b == 3:
                        # Walk: forces in 1 run
                        if d == 0:
                            # Walk-off win
                            take[('WIN',)] = take.get(('WIN',), 0) + pb
                        else:
                            # Next batter: deficit decreases by 1, runs increase by 1
                            d_new = d - 1
                            r_new = min(r + 1, 3)
                            if d_new == 0 and r_new > 0:
                                # Tied — next batter in extras-like scenario, or walk-off
                                # Actually: deficit was 1, walk ties it. Still bot 9.
                                # Bases still loaded (walked in a run, new runner on 1st,
                                # others advance). Actually: walk with bases loaded —
                                # all runners advance one base, batter to 1st.
                                # Bases remain loaded. Next batter at (0,0,0,r+1).
                                pass
                            next_state = (0, 0, d_new, r_new)
                            take[next_state] = take.get(next_state, 0) + pb
                    else:
                        take[(b + 1, s, d, r)] = pb

                    # Called strike
                    if s == 2:
                        take[('K', d, r)] = take.get(('K', d, r), 0) + ps
                    else:
                        take[(b, s + 1, d, r)] = ps

                    T[state]['take'] = take

                    # ── SWING ──
                    swing = {}
                    # Whiff
                    if s == 2:
                        swing[('K', d, r)] = swing.get(('K', d, r), 0) + pw
                    else:
                        swing[(b, s + 1, d, r)] = swing.get((b, s + 1, d, r), 0) + pw

                    # Foul
                    if s < 2:
                        swing[(b, s + 1, d, r)] = swing.get((b, s + 1, d, r), 0) + pf
                    else:
                        swing[(b, 2, d, r)] = swing.get((b, 2, d, r), 0) + pf

                    # Contact — BIP terminal
                    swing[('BIP', d, r)] = swing.get(('BIP', d, r), 0) + pc

                    T[state]['swing'] = swing

    return T


def augmented_value_iteration(trans_params, bip_runs_dist, tol=1e-8, max_iter=1000):
    """
    Solve the augmented batter MDP via value iteration.

    Terminal state rewards:
    - K: game over → loss → WP = 0
    - WIN: walk-off → WP = 1
    - BB (non-walkoff): transitions to next batter state — handled via re-entry
    - BIP: probabilistic run scoring, then check if walk-off/tie/still-behind
    """
    T = build_augmented_transition_dict(trans_params)

    # Initialize
    V = {}
    policy = {}
    for b in range(4):
        for s in range(3):
            for d in DEFICITS:
                for r in range(4):
                    V[(b, s, d, r)] = 0.0

    for iteration in range(max_iter):
        V_new = {}
        for state in V:
            b, s, d, r = state
            action_values = {}

            for action in ['take', 'swing']:
                ev = 0.0
                for next_state, prob in T[state][action].items():
                    if next_state == ('WIN',):
                        # Walk-off win
                        ev += prob * 1.0
                    elif isinstance(next_state, tuple) and next_state[0] == 'K':
                        # Strikeout — game over, loss
                        ev += prob * 0.0
                    elif isinstance(next_state, tuple) and next_state[0] == 'BIP':
                        # Ball in play — probabilistic run scoring
                        _, bip_d, bip_r = next_state
                        bip_ev = 0.0
                        for runs, run_prob in bip_runs_dist.items():
                            total_runs = bip_r + runs  # runs already scored + new runs
                            remaining_deficit = bip_d - runs
                            if remaining_deficit < 0:
                                # Walk-off win
                                bip_ev += run_prob * 1.0
                            elif remaining_deficit == 0:
                                # Tied — extras
                                bip_ev += run_prob * EXTRAS_WP
                            else:
                                # Still behind — game over (it's an out on BIP with 2 out)
                                # Actually: BIP with 2 out — if the batter is out, game over.
                                # If the batter gets a hit, runs score but it's still 2 out.
                                # Wait — we need to separate BIP-out vs BIP-hit.
                                # Our model treats BIP as a single terminal with
                                # probabilistic run outcomes. This includes both hits
                                # (which keep the inning going) and outs (which end it).
                                # For simplicity: treat BIP as terminal for this PA.
                                # If runs scored but still behind: the hit keeps inning going
                                # but we approximate as game-over since the next batter
                                # would need to continue. Let's use a simple continuation model:
                                # P(winning from here) ≈ V*(0, 0, remaining_deficit, total_runs)
                                # but only if it was a HIT (runs > 0). If runs == 0, game over.
                                if runs > 0:
                                    # Hit — inning continues with next batter
                                    new_r = min(total_runs, 3)
                                    new_d = remaining_deficit
                                    if new_d in DEFICITS:
                                        bip_ev += run_prob * V.get((0, 0, new_d, new_r), 0.0)
                                    else:
                                        bip_ev += run_prob * 0.0
                                else:
                                    # BIP-out — game over
                                    bip_ev += run_prob * 0.0
                        ev += prob * bip_ev
                    elif next_state in V:
                        # Regular count transition
                        ev += prob * V[next_state]
                    else:
                        # Re-entry state from walk (might be new batter state)
                        # Check if it's a valid state
                        if len(next_state) == 4 and next_state in V:
                            ev += prob * V[next_state]
                        else:
                            ev += prob * 0.0  # fallback

                action_values[action] = ev

            best = max(action_values, key=action_values.get)
            V_new[state] = action_values[best]
            policy[state] = best

        delta = max(abs(V_new[s] - V[s]) for s in V)
        V = V_new

        if delta < tol:
            return V, policy, iteration + 1, T

    return V, policy, max_iter, T


# Load transition parameters from Session 3
trans_df = pd.read_csv(f"{MDP_DIR}/transition_parameters.csv", index_col=0)
era_transitions = trans_df.to_dict('index')

# Also handle BIP-out vs BIP-hit split
# From Statcast data: what fraction of BIP results in outs vs hits?
out_events = {'field_out', 'grounded_into_double_play', 'force_out',
              'fielders_choice', 'fielders_choice_out', 'double_play',
              'sac_fly', 'sac_bunt', 'triple_play', 'sac_fly_double_play'}
hit_events = {'single', 'double', 'triple', 'home_run', 'field_error'}

bip_all = primary_all[primary_all['events'].isin(out_events | hit_events | bip_events)]
bip_outs = primary_all[primary_all['events'].isin(out_events)]
bip_hits = primary_all[primary_all['events'].isin(hit_events)]

p_bip_out = len(bip_outs) / len(bip_all) if len(bip_all) > 0 else 0.5
p_bip_hit = 1 - p_bip_out
print(f"\n  BIP outcome split: {p_bip_out:.3f} outs, {p_bip_hit:.3f} hits")

# Refined BIP runs distribution:
# outs score 0 runs (probability = p_bip_out)
# hits score according to bip_runs_dist (probability = p_bip_hit × hit_run_dist)
hit_runs_dist = bip_data['runs_scored'].value_counts(normalize=True).sort_index().to_dict()
# Combine: P(0 runs on BIP) = p_bip_out, P(r runs on BIP) = p_bip_hit × P(r|hit)
combined_bip_dist = {0: p_bip_out}
for r, p in hit_runs_dist.items():
    combined_bip_dist[r] = combined_bip_dist.get(r, 0) + p_bip_hit * p

print(f"  Combined BIP run distribution:")
for r, p in sorted(combined_bip_dist.items()):
    print(f"    {r} runs: {p:.3f}")

# Solve augmented MDP for each era
print(f"\n  Solving augmented MDP for each era...")
augmented_solutions = {}
aug_policy_tables = []

for era_name, tparams in era_transitions.items():
    V, policy, n_iter, T = augmented_value_iteration(tparams, combined_bip_dist)
    augmented_solutions[era_name] = {'V': V, 'policy': policy, 'T': T}

    print(f"\n  {era_name}: converged in {n_iter} iterations")

    # Report key states
    for d in DEFICITS:
        v00 = V[(0, 0, d, 0)]
        print(f"    deficit={d}: V(0-0,r=0)={v00:.4f}")

    # Record policies
    for state, action in policy.items():
        b, s, d, r = state
        aug_policy_tables.append({
            'era': era_name, 'balls': b, 'strikes': s,
            'deficit': d, 'runs_scored': r,
            'count': f"{b}-{s}", 'optimal_action': action,
            'value': V[state],
        })

aug_policy_df = pd.DataFrame(aug_policy_tables)
aug_policy_df.to_csv(f"{MDP_DIR}/optimal_policies_v2.csv", index=False)

# ── Validation checks ──────────────────────────────────────────────────
print(f"\n  ── Augmented MDP Validation ──")

for era_name, sol in augmented_solutions.items():
    V, pol = sol['V'], sol['policy']

    # Check 1: V(0,0,0,0) should be ~0.335 (base WE from Session 3)
    v00_d0 = V[(0, 0, 0, 0)]

    # Check 2: V(0,0,1,1) > V(0,0,1,0) — having scored helps
    v_d1_r1 = V[(0, 0, 1, 1)]
    v_d1_r0 = V[(0, 0, 1, 0)]

    # Check 3: π(3,0,1,0) should be SWING (the 3-0 paradox fix)
    p30_d1 = pol[(3, 0, 1, 0)]

    # Check 4: π(3,0,0,0) should be TAKE (walk-off)
    p30_d0 = pol[(3, 0, 0, 0)]

    # Check 5: V(3,0,d,r) should be highest for any d,r
    v30 = V[(3, 0, 0, 0)]
    v02 = V[(0, 2, 0, 0)]

    checks = [
        f"V(0-0,d=0,r=0)={v00_d0:.3f}",
        f"V(d=1,r=1)>V(d=1,r=0): {v_d1_r1:.3f}>{v_d1_r0:.3f} {'✓' if v_d1_r1 > v_d1_r0 else '✗'}",
        f"π(3-0,d=1)={p30_d1} {'✓ SWING' if p30_d1 == 'swing' else '✗ expected swing'}",
        f"π(3-0,d=0)={p30_d0} {'✓ TAKE' if p30_d0 == 'take' else '✗ expected take'}",
        f"V(3-0)={v30:.3f}>V(0-2)={v02:.3f} {'✓' if v30 > v02 else '✗'}",
    ]
    print(f"  {era_name}: {' | '.join(checks)}")

# ── Display policy grids for deficit=0 and deficit=1 (r=0) ────────────
print(f"\n  ── Policy Grid: deficit=0, runs_scored=0 ──")
header = f"  {'Count':<8}"
for en in era_transitions:
    header += f"  {en:>18}"
print(header)
for b in range(4):
    for s in range(3):
        row = f"  {b}-{s:<6}"
        for en in era_transitions:
            a = augmented_solutions[en]['policy'][(b, s, 0, 0)]
            v = augmented_solutions[en]['V'][(b, s, 0, 0)]
            row += f"  {a:>8}({v:+.3f})"
        print(row)

print(f"\n  ── Policy Grid: deficit=1, runs_scored=0 ──")
header = f"  {'Count':<8}"
for en in era_transitions:
    header += f"  {en:>18}"
print(header)
for b in range(4):
    for s in range(3):
        row = f"  {b}-{s:<6}"
        for en in era_transitions:
            a = augmented_solutions[en]['policy'][(b, s, 1, 0)]
            v = augmented_solutions[en]['V'][(b, s, 1, 0)]
            row += f"  {a:>8}({v:+.3f})"
        print(row)

# ── Compare augmented vs Session 3 policy gaps ─────────────────────────
print(f"\n  ── Augmented vs Session 3 Policy Gaps (Statcast era) ──")

all_pitches = []
for year in range(2015, 2025):
    path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        df['is_swing'] = df['description'].isin(SWING_DESCS).astype(int)
        df['deficit'] = df['fld_score'] - df['bat_score']
        df['year'] = year
        all_pitches.append(df[['balls', 'strikes', 'is_swing', 'deficit', 'year']])

all_pitches_df = pd.concat(all_pitches, ignore_index=True)

# For the augmented MDP, we use r=0 as the default (we don't have per-PA
# runs_scored_this_inning from Statcast yet — that requires pitch-by-pitch
# game state tracking). The primary situation IS the first PA in the terminal
# state most of the time, so r=0 is correct for the vast majority of cases.

swing_rates = all_pitches_df.groupby(['balls', 'strikes', 'deficit'])['is_swing'].agg(['mean', 'count']).reset_index()
swing_rates.columns = ['balls', 'strikes', 'deficit', 'swing_rate', 'n_pitches']

sol = augmented_solutions.get('statcast')
if sol:
    V, T, pol = sol['V'], sol['T'], sol['policy']

    aug_gaps = []
    for _, row in swing_rates.iterrows():
        b, s, d = int(row['balls']), int(row['strikes']), int(row['deficit'])
        if d not in DEFICITS:
            continue
        state = (b, s, d, 0)  # r=0 assumption
        if state not in V:
            continue

        sr = row['swing_rate']
        n = int(row['n_pitches'])
        v_opt = V[state]

        # Compute V_observed
        v_obs = 0.0
        for action, aprob in [('swing', sr), ('take', 1.0 - sr)]:
            for ns, tp in T[state][action].items():
                if ns == ('WIN',):
                    v_obs += aprob * tp * 1.0
                elif isinstance(ns, tuple) and ns[0] == 'K':
                    v_obs += aprob * tp * 0.0
                elif isinstance(ns, tuple) and ns[0] == 'BIP':
                    _, bip_d, bip_r = ns
                    bip_ev = 0.0
                    for runs, rp in combined_bip_dist.items():
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

        gap = v_obs - v_opt
        aug_gaps.append({
            'deficit': d, 'balls': b, 'strikes': s,
            'count': f"{b}-{s}", 'swing_rate': sr,
            'optimal': pol[state], 'v_obs': v_obs, 'v_opt': v_opt,
            'gap': gap, 'n': n,
        })

    aug_gap_df = pd.DataFrame(aug_gaps)
    aug_gap_df.to_csv(f"{MDP_DIR}/policy_gaps_statcast_v2.csv", index=False)

    # Aggregate
    for d in DEFICITS:
        sub = aug_gap_df[aug_gap_df['deficit'] == d]
        if len(sub) == 0:
            continue
        wgap = (sub['gap'] * sub['n']).sum() / sub['n'].sum()
        print(f"  deficit={d}: weighted gap = {wgap:+.4f}")

    total_wgap = (aug_gap_df['gap'] * aug_gap_df['n']).sum() / aug_gap_df['n'].sum()
    print(f"\n  Overall augmented weighted gap: {total_wgap:+.4f}")
    print(f"  (Session 3 simple MDP was approximately -0.026)")

    # Per-year augmented gaps
    yearly_aug_gaps = []
    for year in range(2015, 2025):
        yp = all_pitches_df[all_pitches_df['year'] == year]
        ysr = yp.groupby(['balls', 'strikes', 'deficit'])['is_swing'].agg(['mean', 'count']).reset_index()
        ysr.columns = ['balls', 'strikes', 'deficit', 'swing_rate', 'n_pitches']

        total_wg = 0.0
        total_n = 0
        for _, row in ysr.iterrows():
            b, s, d = int(row['balls']), int(row['strikes']), int(row['deficit'])
            if d not in DEFICITS:
                continue
            state = (b, s, d, 0)
            if state not in V:
                continue

            sr = row['swing_rate']
            n = int(row['n_pitches'])
            v_opt = V[state]

            v_obs = 0.0
            for action, aprob in [('swing', sr), ('take', 1.0 - sr)]:
                for ns, tp in T[state][action].items():
                    if ns == ('WIN',):
                        v_obs += aprob * tp * 1.0
                    elif isinstance(ns, tuple) and ns[0] == 'K':
                        v_obs += aprob * tp * 0.0
                    elif isinstance(ns, tuple) and ns[0] == 'BIP':
                        _, bip_d, bip_r = ns
                        bip_ev = 0.0
                        for runs, rp in combined_bip_dist.items():
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

            gap = v_obs - v_opt
            total_wg += gap * n
            total_n += n

        if total_n > 0:
            yearly_aug_gaps.append({'year': year, 'weighted_gap': total_wg / total_n, 'n': total_n})

    yearly_aug_df = pd.DataFrame(yearly_aug_gaps)
    yearly_aug_df.to_csv(f"{MDP_DIR}/yearly_policy_gaps_v2.csv", index=False)
    print(f"\n  Per-year augmented gaps:")
    for _, row in yearly_aug_df.iterrows():
        print(f"    {int(row['year'])}: gap={row['weighted_gap']:+.4f} (n={int(row['n'])})")


print(f"\n  ✓ Step 1 complete — augmented MDP saved with _v2 suffix")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Inning Run Counter for Retrosheet
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 2: Inning Run Counter")
print("=" * 70)


def compute_inning_run_context(events_df):
    """
    For each play, compute runs_scored_this_inning using score deltas.
    Uses HOME_SCORE_CT / AWAY_SCORE_CT changes between consecutive plays.
    """
    events_df = events_df.sort_values(['GAME_ID', 'INN_CT', 'BAT_HOME_ID']).copy()

    # Score at start of each half-inning
    groups = events_df.groupby(['GAME_ID', 'INN_CT', 'BAT_HOME_ID'])

    runs_this_inning = []
    for _, group in groups:
        group = group.sort_index()
        # Batting team's score: HOME if BAT_HOME_ID=1, AWAY if BAT_HOME_ID=0
        is_home = group['BAT_HOME_ID'].iloc[0] == 1
        if is_home:
            scores = group['HOME_SCORE_CT'].values
        else:
            scores = group['AWAY_SCORE_CT'].values

        start_score = scores[0]
        runs = scores - start_score  # cumulative runs scored this half-inning
        runs_this_inning.extend(runs.tolist())

    events_df['runs_scored_this_inning'] = runs_this_inning
    return events_df


# Validate on 2000 data
retro_2000 = pd.read_csv(f"{BASE_DIR}/data/retrosheet/events_2000.csv", low_memory=False)
retro_2000 = compute_inning_run_context(retro_2000)

# Spot check: runs should be 0 at start of each half-inning
first_plays = retro_2000.groupby(['GAME_ID', 'INN_CT', 'BAT_HOME_ID']).first()
non_zero_starts = (first_plays['runs_scored_this_inning'] != 0).sum()
print(f"  Validation on 2000: {non_zero_starts} half-innings with non-zero starting runs (should be 0)")

# Check that runs_scored_this_inning never decreases within a half-inning
decreases = 0
for _, group in retro_2000.groupby(['GAME_ID', 'INN_CT', 'BAT_HOME_ID']):
    runs = group['runs_scored_this_inning'].values
    if any(runs[i+1] < runs[i] for i in range(len(runs)-1)):
        decreases += 1
print(f"  Half-innings with decreasing run count: {decreases} (should be 0)")

# Find a known big inning for manual validation
big_inning = retro_2000.groupby(['GAME_ID', 'INN_CT', 'BAT_HOME_ID'])['runs_scored_this_inning'].max()
big = big_inning[big_inning >= 5].head(3)
print(f"\n  Sample big innings for manual check:")
for (gid, inn, home), max_runs in big.items():
    sub = retro_2000[(retro_2000['GAME_ID'] == gid) &
                     (retro_2000['INN_CT'] == inn) &
                     (retro_2000['BAT_HOME_ID'] == home)]
    print(f"    {gid}, inn={inn}, home={home}: max_runs={max_runs}, events={len(sub)}")
    for _, row in sub.tail(5).iterrows():
        print(f"      {row.get('BAT_ID','?')} EVENT_CD={row.get('EVENT_CD','?')} "
              f"runs_this_inn={row['runs_scored_this_inning']}")

print(f"\n  ✓ Step 2 complete — inning run counter validated")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Pull Historical Retrosheet Years
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 3: Historical Retrosheet Pull")
print("=" * 70)


def download_retrosheet_year(year):
    """Download Retrosheet event files for a given year."""
    import ssl
    import urllib.request
    import zipfile
    import io

    year_dir = f"{RETRO_DIR}/{year}"
    if os.path.exists(year_dir) and len(os.listdir(year_dir)) > 5:
        print(f"  {year}: already downloaded ({len(os.listdir(year_dir))} files)")
        return True

    os.makedirs(year_dir, exist_ok=True)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    url = f"https://www.retrosheet.org/events/{year}eve.zip"
    print(f"  Downloading {url}...")

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (research project)'})
        resp = urllib.request.urlopen(req, context=ctx, timeout=30)
        data = resp.read()

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(year_dir)
            print(f"    Extracted {len(zf.namelist())} files")
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        return False


def parse_retrosheet_year(year):
    """Parse event files using cwevent, return DataFrame."""
    cache_path = f"{BASE_DIR}/data/retrosheet/events_{year}.csv"
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, low_memory=False)

    year_dir = f"{RETRO_DIR}/{year}"
    ev_files = sorted(glob.glob(f"{year_dir}/{year}*.EV*"))
    team_file = glob.glob(f"{year_dir}/TEAM{year}")

    if not ev_files:
        print(f"  {year}: no event files found")
        return None

    if not team_file:
        print(f"  {year}: no TEAM file found")
        return None

    all_rows = []
    for evf in ev_files:
        try:
            result = subprocess.run(
                ['cwevent', '-y', str(year), '-f', '0-96', evf],
                capture_output=True, text=True, cwd=year_dir, timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                from io import StringIO
                chunk = pd.read_csv(StringIO(result.stdout), header=None)
                all_rows.append(chunk)
        except Exception as e:
            print(f"    Error parsing {os.path.basename(evf)}: {e}")

    if not all_rows:
        return None

    df = pd.concat(all_rows, ignore_index=True)

    # Add headers
    headers = [
        'GAME_ID', 'AWAY_TEAM_ID', 'INN_CT', 'BAT_HOME_ID', 'OUTS_CT',
        'BALLS_CT', 'STRIKES_CT', 'PITCH_SEQ_TX', 'AWAY_SCORE_CT', 'HOME_SCORE_CT',
        'BAT_ID', 'BAT_HAND_CD', 'RES_BAT_ID', 'RES_BAT_HAND_CD', 'PIT_ID',
        'PIT_HAND_CD', 'RES_PIT_ID', 'RES_PIT_HAND_CD', 'POS2_FLD_ID', 'POS3_FLD_ID',
        'POS4_FLD_ID', 'POS5_FLD_ID', 'POS6_FLD_ID', 'POS7_FLD_ID', 'POS8_FLD_ID',
        'POS9_FLD_ID', 'BASE1_RUN_ID', 'BASE2_RUN_ID', 'BASE3_RUN_ID', 'BAT_LINEUP_ID',
        'BAT_FLD_CD', 'EVENT_TX', 'LEADOFF_FL', 'PH_FL', 'BAT_EVENT_FL',
        'AB_FL', 'H_FL', 'SH_FL', 'SF_FL', 'EVENT_OUTS_CT',
        'DP_FL', 'TP_FL', 'RBI_CT', 'WP_FL', 'PB_FL',
        'FLD_CD', 'BATTEDBALL_CD', 'BUNT_FL', 'FOUL_FL', 'BATTEDBALL_LOC_TX',
        'ERR_CT', 'ERR1_FLD_CD', 'ERR1_CD', 'ERR2_FLD_CD', 'ERR2_CD',
        'ERR3_FLD_CD', 'ERR3_CD', 'BAT_DEST_ID', 'RUN1_DEST_ID', 'RUN2_DEST_ID',
        'RUN3_DEST_ID', 'BAT_PLAY_TX', 'RUN1_PLAY_TX', 'RUN2_PLAY_TX', 'RUN3_PLAY_TX',
        'RUN1_SB_FL', 'RUN2_SB_FL', 'RUN3_SB_FL', 'RUN1_CS_FL', 'RUN2_CS_FL',
        'RUN3_CS_FL', 'PO1_FL', 'PO2_FL', 'PO3_FL', 'ASS_CT',
        'PO_CT', 'A_CT', 'E_CT', 'BAT_TEAM_ID', 'FLD_TEAM_ID',
        'HALF_INN_ID', 'START_BAT_SCORE_CT', 'INN_RUNS_CT', 'GAME_NEW_FL', 'GAME_END_FL',
        'PR_RUN1_FL', 'PR_RUN2_FL', 'PR_RUN3_FL', 'REMOVED_FOR_PR_RUN1_ID',
        'REMOVED_FOR_PR_RUN2_ID', 'REMOVED_FOR_PR_RUN3_ID',
        'REMOVED_FOR_PH_BAT_ID', 'REMOVED_FOR_PH_BAT_FLD_CD',
        'PO1_FLD_CD', 'PO2_FLD_CD', 'PO3_FLD_CD',
        'EVENT_CD', 'BAT_SAFE_ERR_FL',
    ]

    if len(df.columns) <= len(headers):
        df.columns = headers[:len(df.columns)]
    else:
        extra = [f'FIELD_{i}' for i in range(len(headers), len(df.columns))]
        df.columns = headers + extra

    df.to_csv(cache_path, index=False)
    print(f"  {year}: parsed {len(df):,} events, cached")
    return df


def extract_situation_data(df, year):
    """Extract primary situation PAs and transition data from parsed events."""
    # Compute inning run context
    df = compute_inning_run_context(df)

    # ── Primary situation filter ──
    situation = df[
        (df['INN_CT'] == 9) &
        (df['BAT_HOME_ID'] == 1) &
        (df['OUTS_CT'] == 2) &
        (df['BASE1_RUN_ID'].notna()) & (df['BASE1_RUN_ID'] != '') &
        (df['BASE2_RUN_ID'].notna()) & (df['BASE2_RUN_ID'] != '') &
        (df['BASE3_RUN_ID'].notna()) & (df['BASE3_RUN_ID'] != '')
    ].copy()

    # Only batting events
    if 'BAT_EVENT_FL' in situation.columns:
        situation = situation[situation['BAT_EVENT_FL'] == 'T']

    # Add deficit
    situation['deficit'] = situation['AWAY_SCORE_CT'] - situation['HOME_SCORE_CT']

    # ── Transition data (league-wide) ──
    bat_events = df[df['BAT_EVENT_FL'] == 'T'].copy() if 'BAT_EVENT_FL' in df.columns else df.copy()
    seqs = bat_events['PITCH_SEQ_TX'].dropna()
    seqs = seqs[(seqs != '') & (seqs != 'nan')]
    coverage = len(seqs) / len(bat_events) if len(bat_events) > 0 else 0

    trans_counts = {'ball': 0, 'called_strike': 0, 'swinging_strike': 0, 'foul': 0, 'contact': 0}
    for seq in seqs:
        for c in str(seq):
            if c in IGNORE_CHARS: continue
            elif c in BALL_CHARS: trans_counts['ball'] += 1
            elif c in CALLED_STRIKE_CHARS: trans_counts['called_strike'] += 1
            elif c in SWINGING_STRIKE_CHARS: trans_counts['swinging_strike'] += 1
            elif c in FOUL_CHARS: trans_counts['foul'] += 1
            elif c in CONTACT_CHARS: trans_counts['contact'] += 1

    # ── 3-0 count tracking ──
    three_zero_records = []
    for _, pa in situation.iterrows():
        seq = str(pa.get('PITCH_SEQ_TX', ''))
        if seq == 'nan' or seq == '':
            continue

        # Decode pitch-by-pitch to find 3-0 count
        balls, strikes = 0, 0
        swung_on_3_0 = False
        reached_3_0 = False

        for c in seq:
            if c in IGNORE_CHARS:
                continue

            if balls == 3 and strikes == 0:
                reached_3_0 = True
                # Check if this pitch is a swing
                if c in SWINGING_STRIKE_CHARS or c in FOUL_CHARS or c in CONTACT_CHARS:
                    swung_on_3_0 = True
                break  # Only care about the first pitch at 3-0

            # Update count
            if c in BALL_CHARS:
                balls += 1
            elif c in CALLED_STRIKE_CHARS:
                strikes += 1
            elif c in SWINGING_STRIKE_CHARS:
                strikes = min(strikes + 1, 2)  # can't strike out on foul at 2 strikes... wait, swinging strike CAN
                if strikes >= 3:
                    break  # K
                strikes = min(strikes + 1, 2) if c in FOUL_CHARS and strikes == 2 else strikes
                # Actually simplify: swinging strike always adds a strike
                # We already incremented, so just check bounds
            elif c in FOUL_CHARS:
                if strikes < 2:
                    strikes += 1
            elif c in CONTACT_CHARS:
                break  # PA over

        if reached_3_0:
            three_zero_records.append({
                'year': year,
                'game_id': pa.get('GAME_ID', ''),
                'batter': pa.get('BAT_ID', ''),
                'deficit': pa['deficit'],
                'runs_scored_this_inning': pa.get('runs_scored_this_inning', 0),
                'swung': 1 if swung_on_3_0 else 0,
                'era': assign_era(year),
            })

    return {
        'situation': situation,
        'trans_counts': trans_counts,
        'pitch_seq_coverage': coverage,
        'three_zero': pd.DataFrame(three_zero_records) if three_zero_records else pd.DataFrame(),
        'total_pas': len(bat_events),
    }


# Phase 1: Decade representatives
phase1_years = [1955, 1965, 1975, 1985, 1995, 2005, 2010]
# Also include years already parsed
existing_years = [1960, 1990, 2000]
all_target_years = sorted(set(phase1_years + existing_years))

historical_results = {}

for year in all_target_years:
    print(f"\n  ── {year} ──")

    # Download if needed
    if year not in existing_years:
        if not download_retrosheet_year(year):
            print(f"  Skipping {year}")
            continue

    # Parse
    df = parse_retrosheet_year(year)
    if df is None:
        print(f"  No data for {year}")
        continue

    print(f"  Events: {len(df):,}")

    # Extract situation data
    result = extract_situation_data(df, year)
    n_sit = len(result['situation'])
    n_3_0 = len(result['three_zero'])
    cov = result['pitch_seq_coverage']

    print(f"  Situation PAs: {n_sit}, 3-0 counts: {n_3_0}, pitch seq coverage: {cov:.1%}")
    print(f"  Transition counts: {result['trans_counts']}")

    # Check PA floor
    below_floor = n_sit < MIN_PA_FLOOR
    if below_floor:
        print(f"  ⚠ Below {MIN_PA_FLOOR} PA floor — included in viz only")

    # Cache
    if n_sit > 0:
        result['situation'].to_csv(f"{BASE_DIR}/data/retrosheet/situation_{year}.csv", index=False)
    if n_3_0 > 0:
        result['three_zero'].to_csv(f"{BASE_DIR}/data/retrosheet/three_zero_{year}.csv", index=False)

    historical_results[year] = {
        'n_sit': n_sit,
        'n_three_zero': n_3_0,
        'coverage': cov,
        'trans_counts': result['trans_counts'],
        'below_floor': below_floor,
        'era': assign_era(year),
    }

print(f"\n  ── Phase 1 Summary ──")
for year in sorted(historical_results.keys()):
    r = historical_results[year]
    flag = " ⚠<floor" if r['below_floor'] else ""
    print(f"  {year} ({r['era']}): {r['n_sit']} PAs, {r['n_three_zero']} 3-0s, coverage={r['coverage']:.0%}{flag}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Compute Era-Level Policy Gaps (Historical)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 4: Historical Policy Gaps")
print("=" * 70)

def decode_pitch_sequence_to_actions(seq):
    """
    Decode Retrosheet PITCH_SEQ_TX to a list of (balls, strikes, action) tuples.
    action: 'swing' or 'take'
    """
    if not isinstance(seq, str) or seq == 'nan':
        return []

    pitches = []
    balls, strikes = 0, 0

    for c in seq:
        if c in IGNORE_CHARS:
            continue

        if c in BALL_CHARS:
            pitches.append((balls, strikes, 'take'))
            balls += 1
            if balls >= 4:
                break  # BB
        elif c in CALLED_STRIKE_CHARS:
            pitches.append((balls, strikes, 'take'))
            strikes += 1
            if strikes >= 3:
                break  # K on called strike
        elif c in SWINGING_STRIKE_CHARS:
            pitches.append((balls, strikes, 'swing'))
            strikes += 1
            if strikes >= 3:
                break  # K on swinging strike
        elif c in FOUL_CHARS:
            pitches.append((balls, strikes, 'swing'))
            if strikes < 2:
                strikes += 1
        elif c in CONTACT_CHARS:
            pitches.append((balls, strikes, 'swing'))
            break  # BIP

    return pitches


def compute_historical_policy_gap(year, situation_df, era_name, augmented_solutions, combined_bip_dist):
    """
    Compute policy gap for a historical year using Retrosheet pitch sequences.
    """
    # Get the MDP solution for this era
    if era_name not in augmented_solutions:
        # Use nearest available
        era_order = ['post_war', 'expansion', 'offense_explosion', 'post_steroid', 'statcast']
        era_name = min(augmented_solutions.keys(),
                       key=lambda e: abs(era_order.index(e) - era_order.index(era_name))
                       if e in era_order and era_name in era_order else 999)

    sol = augmented_solutions[era_name]
    V, T, pol = sol['V'], sol['T'], sol['policy']

    gaps_by_count = {}  # (balls, strikes, deficit) → list of gaps

    for _, pa in situation_df.iterrows():
        seq = str(pa.get('PITCH_SEQ_TX', ''))
        if seq == 'nan' or seq == '':
            continue

        deficit = int(pa.get('deficit', 0))
        if deficit not in DEFICITS:
            continue

        r = int(pa.get('runs_scored_this_inning', 0))
        r = min(r, 3)

        pitches = decode_pitch_sequence_to_actions(seq)

        for balls, strikes, action in pitches:
            state = (balls, strikes, deficit, r)
            if state not in V:
                continue

            # Observed: this batter swung or took
            is_swing = 1 if action == 'swing' else 0
            v_opt = V[state]

            # V_observed for this single pitch
            obs_probs = {'swing': float(is_swing), 'take': 1.0 - float(is_swing)}
            v_obs = 0.0
            for a, ap in obs_probs.items():
                if ap == 0:
                    continue
                for ns, tp in T[state][a].items():
                    if ns == ('WIN',):
                        v_obs += ap * tp * 1.0
                    elif isinstance(ns, tuple) and ns[0] == 'K':
                        v_obs += ap * tp * 0.0
                    elif isinstance(ns, tuple) and ns[0] == 'BIP':
                        _, bip_d, bip_r = ns
                        bip_ev = 0.0
                        for runs, rp in combined_bip_dist.items():
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
                        v_obs += ap * tp * bip_ev
                    elif ns in V:
                        v_obs += ap * tp * V[ns]

            gap = v_obs - v_opt
            key = (balls, strikes, deficit)
            if key not in gaps_by_count:
                gaps_by_count[key] = []
            gaps_by_count[key].append(gap)

    # Aggregate
    all_gaps = []
    for (b, s, d), gap_list in gaps_by_count.items():
        all_gaps.extend(gap_list)

    if all_gaps:
        return np.mean(all_gaps), len(all_gaps)
    return None, 0


# Compute gaps for all historical years
longitudinal_gaps = []

# First: historical Retrosheet years
for year in sorted(historical_results.keys()):
    info = historical_results[year]

    sit_path = f"{BASE_DIR}/data/retrosheet/situation_{year}.csv"
    if not os.path.exists(sit_path):
        continue

    situation_df = pd.read_csv(sit_path, low_memory=False)
    era_name = info['era']

    gap, n_pitches = compute_historical_policy_gap(
        year, situation_df, era_name, augmented_solutions, combined_bip_dist)

    if gap is not None:
        longitudinal_gaps.append({
            'year': year, 'era': era_name,
            'weighted_gap': gap, 'n_pitches': n_pitches,
            'below_floor': info['below_floor'],
            'source': 'retrosheet',
        })
        flag = " ⚠" if info['below_floor'] else ""
        print(f"  {year} ({era_name}): gap={gap:+.4f}, n={n_pitches}{flag}")

# Add Statcast era from yearly_aug_df
if 'yearly_aug_df' in dir() and len(yearly_aug_df) > 0:
    for _, row in yearly_aug_df.iterrows():
        longitudinal_gaps.append({
            'year': int(row['year']),
            'era': 'statcast',
            'weighted_gap': row['weighted_gap'],
            'n_pitches': int(row['n']),
            'below_floor': int(row['n']) < MIN_PA_FLOOR * 4,  # pitches, not PAs
            'source': 'statcast',
        })

long_df = pd.DataFrame(longitudinal_gaps).sort_values('year')
long_df.to_csv(f"{LONG_DIR}/policy_gaps_historical.csv", index=False)

print(f"\n  ── Full Longitudinal Series ──")
for _, row in long_df.iterrows():
    flag = " ⚠" if row['below_floor'] else ""
    print(f"  {int(row['year'])} ({row['era']}): gap={row['weighted_gap']:+.4f}, n={int(row['n_pitches'])}, src={row['source']}{flag}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: 3-0 Paradox Time Series
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 5: 3-0 Paradox Time Series")
print("=" * 70)

# Collect 3-0 data from all years
all_three_zero = []

# Retrosheet years
for year in sorted(historical_results.keys()):
    path = f"{BASE_DIR}/data/retrosheet/three_zero_{year}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        all_three_zero.append(df)

# Statcast years
for year in range(2015, 2025):
    path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    if not os.path.exists(path):
        continue

    df = pd.read_csv(path, low_memory=False)
    df['deficit'] = df['fld_score'] - df['bat_score']
    df['is_swing'] = df['description'].isin(SWING_DESCS).astype(int)

    # Find 3-0 count pitches
    three_zero = df[(df['balls'] == 3) & (df['strikes'] == 0)].copy()
    if len(three_zero) > 0:
        records = three_zero[['deficit', 'is_swing']].copy()
        records['year'] = year
        records['era'] = 'statcast'
        records.columns = ['deficit', 'swung', 'year', 'era']
        all_three_zero.append(records)

if all_three_zero:
    three_zero_all = pd.concat(all_three_zero, ignore_index=True)

    # Time series by year and deficit
    ts_records = []
    for year in sorted(three_zero_all['year'].unique()):
        yr_data = three_zero_all[three_zero_all['year'] == year]

        # Overall
        overall = yr_data['swung'].mean()
        ts_records.append({
            'year': year, 'deficit': 'all',
            'swing_rate': overall, 'n': len(yr_data),
            'era': assign_era(year),
        })

        # By deficit
        for d in DEFICITS:
            sub = yr_data[yr_data['deficit'] == d]
            if len(sub) >= 3:
                ts_records.append({
                    'year': year, 'deficit': d,
                    'swing_rate': sub['swung'].mean(), 'n': len(sub),
                    'era': assign_era(year),
                })

    ts_df = pd.DataFrame(ts_records)
    ts_df.to_csv(f"{LONG_DIR}/three_zero_time_series.csv", index=False)

    print(f"\n  3-0 swing rates in terminal situation:")
    print(f"  {'Year':<6} {'All':>8} {'d=0':>8} {'d=1':>8} {'d=2':>8} {'d=3':>8}")
    for year in sorted(ts_df['year'].unique()):
        yr = ts_df[ts_df['year'] == year]
        vals = {'all': '', 0: '', 1: '', 2: '', 3: ''}
        for _, row in yr.iterrows():
            d = row['deficit']
            vals[d] = f"{row['swing_rate']:.2f}({int(row['n'])})"
        print(f"  {year:<6} {vals['all']:>8} {vals.get(0,''):>8} {vals.get(1,''):>8} {vals.get(2,''):>8} {vals.get(3,''):>8}")
else:
    print("  No 3-0 count data found")
    ts_df = pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: CUSUM Changepoint Detection
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 6: CUSUM Changepoint Detection")
print("=" * 70)


def cusum_detection(series, years, target_mean=None, k=0.01, h=0.05):
    """
    CUSUM control chart for detecting shifts in the policy gap series.

    k: allowance (half the minimum shift to detect)
    h: decision threshold
    """
    if target_mean is None:
        # Use pre-2015 mean as baseline
        pre_statcast = [(y, v) for y, v in zip(years, series) if y < 2015]
        if pre_statcast:
            target_mean = np.mean([v for _, v in pre_statcast])
        else:
            target_mean = np.mean(series[:3])

    C_plus = [0.0]
    C_minus = [0.0]
    signals_up = []
    signals_down = []

    for i, (y, x) in enumerate(zip(years, series)):
        c_p = max(0, C_plus[-1] + (x - target_mean) - k)
        c_m = max(0, C_minus[-1] - (x - target_mean) - k)
        C_plus.append(c_p)
        C_minus.append(c_m)

        if c_p > h:
            signals_up.append((y, i))
        if c_m > h:
            signals_down.append((y, i))

    return C_plus[1:], C_minus[1:], signals_up, signals_down, target_mean


def ewma_chart(series, years, lambda_=0.2, L=3.0):
    """EWMA control chart."""
    if len(series) < 3:
        return [], [], [], []

    mu_0 = np.mean(series[:max(3, len(series)//3)])
    sigma = np.std(series[:max(3, len(series)//3)])

    if sigma == 0:
        sigma = 0.01

    z = [series[0]]
    ucl = []
    lcl = []
    signals = []

    for i in range(1, len(series)):
        z_new = lambda_ * series[i] + (1 - lambda_) * z[-1]
        z.append(z_new)
        cl_width = L * sigma * np.sqrt(lambda_ / (2 - lambda_) * (1 - (1 - lambda_)**(2*(i+1))))
        ucl.append(mu_0 + cl_width)
        lcl.append(mu_0 - cl_width)

        if z_new > mu_0 + cl_width or z_new < mu_0 - cl_width:
            signals.append((years[i], i))

    return z, ucl, lcl, signals


if len(long_df) >= 5:
    # Sort by year
    long_sorted = long_df.sort_values('year')
    years = long_sorted['year'].values
    gaps = long_sorted['weighted_gap'].values

    # CUSUM
    C_plus, C_minus, sig_up, sig_down, target = cusum_detection(gaps, years)
    print(f"\n  CUSUM (target mean = {target:.4f}):")
    if sig_up:
        print(f"    Upward shifts detected at: {[y for y, _ in sig_up]}")
    if sig_down:
        print(f"    Downward shifts detected at: {[y for y, _ in sig_down]}")
    if not sig_up and not sig_down:
        print(f"    No significant changepoints detected")

    # EWMA
    z, ucl, lcl, ewma_signals = ewma_chart(gaps, years)
    print(f"\n  EWMA (λ=0.2, L=3.0):")
    if ewma_signals:
        print(f"    Signals at: {[y for y, _ in ewma_signals]}")
    else:
        print(f"    No significant signals detected")

    # Agreement check
    cusum_years = set([y for y, _ in sig_up + sig_down])
    ewma_years = set([y for y, _ in ewma_signals])
    agreement = cusum_years & ewma_years
    if agreement:
        print(f"\n  ✓ CUSUM and EWMA agree on changepoints at: {sorted(agreement)}")
    elif not cusum_years and not ewma_years:
        print(f"\n  ✓ Both methods agree: no structural breaks detected")
    else:
        print(f"\n  ⚠ Methods disagree. CUSUM: {sorted(cusum_years) if cusum_years else 'none'}, EWMA: {sorted(ewma_years) if ewma_years else 'none'}")

    # Save
    changepoint_df = pd.DataFrame({
        'year': years,
        'gap': gaps,
        'cusum_plus': C_plus,
        'cusum_minus': C_minus,
    })
    changepoint_df.to_csv(f"{LONG_DIR}/changepoint_analysis.csv", index=False)

    # Linear trend test on full series
    if len(years) >= 5:
        slope, intercept, r, p, se = stats.linregress(years, gaps)
        print(f"\n  Linear trend (full series): slope={slope:+.6f}/year, r²={r**2:.3f}, p={p:.4f}")
else:
    print("  Insufficient data for changepoint analysis")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Pitcher Accountability Analysis
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 7: Pitcher Accountability Analysis (Statcast era)")
print("=" * 70)

# Use the pitcher deviation data from Session 2
pit_dev_path = f"{BASE_DIR}/data/deviations/statcast_pitcher_deviations.csv"
if os.path.exists(pit_dev_path):
    pit_devs = pd.read_csv(pit_dev_path)

    # For pitcher accountability, we need to know if the pitcher allowed the
    # baserunners or inherited them. With Statcast, we can check if the pitcher
    # was in the game when the runners reached base.
    #
    # Simplified proxy: use the number of pitches thrown in the situation.
    # A pitcher who entered mid-inning (inherited runners) will have fewer
    # pitches than one who started the inning. This is a rough proxy.
    #
    # More directly: check if the situation data has 'if_fielding_alignment'
    # or track pitcher changes.

    # For now, use a descriptive split based on zone rate deviation
    print(f"  Loaded {len(pit_devs)} pitcher deviation records")

    # Key metrics
    for metric in ['dev_zone_rate', 'dev_hard_pct', 'dev_offspeed_pct', 'dev_fps_rate']:
        if metric in pit_devs.columns:
            vals = pit_devs[metric].dropna()
            if len(vals) > 0:
                t, p = stats.ttest_1samp(vals, 0)
                print(f"  {metric}: mean={vals.mean():+.4f}, t={t:.2f}, p={p:.4f}, n={len(vals)}")
else:
    print("  No pitcher deviation data found")


# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" SESSION 4 COMPLETE")
print("=" * 70)

print(f"\nOutput files:")
for d in [MDP_DIR, LONG_DIR]:
    for f in sorted(os.listdir(d)):
        path = os.path.join(d, f)
        print(f"  {os.path.relpath(path, BASE_DIR)} ({os.path.getsize(path):,} bytes)")
