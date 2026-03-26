"""
Session 3: MDP Infrastructure for Terminal Game State Analysis
=============================================================

Components:
  Step 1: Win expectancy tables by era (reward function)
  Step 2: Transition probability matrices by era
  Step 3: Value iteration solver
  Step 4: Policy gap metric
  Step 5: Sensitivity analysis
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/Users/paul821/Desktop/baseball/baseball_terminal_state"
MDP_DIR = f"{BASE_DIR}/data/mdp"
os.makedirs(MDP_DIR, exist_ok=True)

# ── Era definitions ─────────────────────────────────────────────────────
ERAS = {
    'post_war':          (1950, 1968),
    'expansion':         (1969, 1992),
    'offense_explosion': (1993, 2005),
    'post_steroid':      (2006, 2014),
    'statcast':          (2015, 2024),
}

# Deficit levels to model
DEFICITS = [0, -1, -2, -3]

# Count states: (balls, strikes)
COUNT_STATES = [(b, s) for b in range(4) for s in range(3)]

# Terminal states
TERMINAL_K = 'K'
TERMINAL_BB = 'BB'
TERMINAL_BIP = 'BIP'  # Ball in play

# Retrosheet event codes
EVENT_K = 3
EVENT_BB = 14
EVENT_IBB = 15
EVENT_HBP = 16
EVENT_SINGLE = 20
EVENT_DOUBLE = 21
EVENT_TRIPLE = 22
EVENT_HR = 23

# Pitch sequence decoding
BALL_CHARS = set('BIPV')      # Ball, intentional ball, pitchout, called ball on pitchout
CALLED_STRIKE_CHARS = set('CK')  # Called strike, strike (unknown type)
SWINGING_STRIKE_CHARS = set('SMQT')  # Swinging miss, missed bunt, swinging on pitchout, foul tip (2-strike out)
FOUL_CHARS = set('FHLOR')     # Foul, foul bunt, foul bunt attempt, foul tip of bunt, foul on pitchout
CONTACT_CHARS = set('XY')     # Ball in play, ball in play (bunt)
IGNORE_CHARS = set('>+*123.N')  # Prefixes, pickoffs, no pitch


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Win Expectancy Tables
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(" STEP 1: Win Expectancy Tables by Era")
print("=" * 70)


def compute_win_expectancy_from_retrosheet(events_df, era_name):
    """
    Compute WE for bot 9, 2 out, bases loaded situations from game outcomes.

    For each game in this situation, we know:
    - The batting team's deficit (fld_score - bat_score mapped from AWAY/HOME scores)
    - The PA outcome (EVENT_CD)
    - Whether the batting team ultimately won (from GAME_END_FL or final score)

    Returns: dict of {deficit: {outcome: delta_wp}}
    """
    # Filter to bot 9 batting events
    bot9 = events_df[
        (events_df['INN_CT'] >= 9) &
        (events_df['BAT_HOME_ID'] == 1)
    ].copy()

    # Compute score differential from batting team perspective
    # BAT_HOME_ID=1 means home team batting, so:
    # home_score = HOME_SCORE_CT (before this event)
    # away_score = AWAY_SCORE_CT
    bot9['bat_score'] = bot9['HOME_SCORE_CT']
    bot9['fld_score'] = bot9['AWAY_SCORE_CT']
    bot9['deficit'] = bot9['fld_score'] - bot9['bat_score']

    return bot9


def compute_statcast_win_expectancy(year_range):
    """
    Compute WE from Statcast data for the statcast era.
    Uses all bot 9+ situations to build a win expectancy model.
    """
    we_data = []

    for year in range(year_range[0], year_range[1] + 1):
        # Load full situation data (primary + secondary have the fields we need)
        # But we need broader data — use the baseline files which have all bot-9 context
        # Actually, for WE we need game outcomes. Use primary/secondary files which
        # have game_pk and events.
        for prefix in ['primary', 'secondary']:
            path = f"{BASE_DIR}/data/statcast/{prefix}_{year}.csv"
            if os.path.exists(path):
                df = pd.read_csv(path, low_memory=False)
                # Get terminal PAs (events not null)
                terminal = df[df['events'].notna()].copy()
                if len(terminal) > 0:
                    terminal['deficit'] = terminal['fld_score'] - terminal['bat_score']
                    terminal['year'] = year
                    we_data.append(terminal)

    if we_data:
        return pd.concat(we_data, ignore_index=True)
    return pd.DataFrame()


def build_we_table_from_outcomes():
    """
    Build win expectancy using theoretical/empirical estimates for
    bot 9, 2 out, bases loaded at each deficit.

    This uses the observed outcome frequencies and known win probabilities.
    """
    # For bot 9, 2 out, bases loaded, the key insight is:
    # - Any out (K, groundout, flyout, etc.) → game over, batting team loses
    # - Walk/HBP → forces in 1 run
    # - Single → typically scores 2 runs (runners from 2nd and 3rd)
    # - Double → typically scores 3 runs (clears bases)
    # - Triple/HR → scores all runners + batter

    # Base win expectancies for bot 9, 2 out, bases loaded
    # Derived from historical data: batting team wins roughly 35-45% depending on deficit

    # We'll compute these empirically from Statcast data where we have outcomes
    we_tables = {}

    # Load Statcast primary situation data with outcomes
    all_primary = []
    for year in range(2015, 2025):
        path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            terminal = df[df['events'].notna()].copy()
            if len(terminal) > 0:
                terminal['deficit'] = terminal['fld_score'] - terminal['bat_score']
                all_primary.append(terminal)

    if not all_primary:
        return we_tables

    primary_all = pd.concat(all_primary, ignore_index=True)

    # Classify outcomes
    def classify_outcome(event):
        if event in ('strikeout', 'strikeout_double_play'):
            return 'K'
        elif event in ('walk'):
            return 'BB'
        elif event in ('hit_by_pitch'):
            return 'HBP'
        elif event in ('single'):
            return 'single'
        elif event in ('double'):
            return 'double'
        elif event in ('triple'):
            return 'triple'
        elif event in ('home_run'):
            return 'HR'
        elif event in ('field_error'):
            return 'error'
        else:
            return 'other_out'  # groundout, flyout, etc.

    primary_all['outcome_class'] = primary_all['events'].apply(classify_outcome)

    print(f"\n  Outcome distribution in primary situation (n={len(primary_all)}):")
    print(primary_all['outcome_class'].value_counts().to_string())

    # Compute runs scored by outcome type (from delta_run_exp or post_bat_score)
    # Use the at-bat result to estimate runs scored
    if 'post_bat_score' in primary_all.columns:
        primary_all['runs_scored'] = primary_all['post_bat_score'] - primary_all['bat_score']
    else:
        # Estimate from RBI or outcome type
        runs_map = {
            'K': 0, 'other_out': 0,
            'BB': 1, 'HBP': 1,
            'single': 2,  # average — runners from 2nd/3rd score
            'double': 3,  # average — clears most bases
            'triple': 3,  # runners + batter to 3rd
            'HR': 4,      # grand slam
            'error': 1,   # conservative estimate
        }
        primary_all['runs_scored_est'] = primary_all['outcome_class'].map(runs_map)

    # For each deficit, compute whether batting team wins after each outcome
    # Bot 9, 2 out: after outcome, if batting team has more runs → walk-off win
    # If tied → extra innings (roughly 50% win probability)
    # If still behind → loss

    for deficit in DEFICITS:
        subset = primary_all[primary_all['deficit'] == deficit]
        if len(subset) < 5:
            print(f"\n  Deficit {deficit}: n={len(subset)}, too few for WE estimation")
            continue

        print(f"\n  Deficit {deficit}: n={len(subset)}")

        we = {}
        for outcome in subset['outcome_class'].unique():
            outcome_rows = subset[subset['outcome_class'] == outcome]
            n = len(outcome_rows)

            if 'runs_scored' in outcome_rows.columns:
                runs = outcome_rows['runs_scored'].values
            else:
                runs = outcome_rows['runs_scored_est'].values

            # After scoring 'runs' runs with deficit 'deficit':
            # New score diff = deficit + runs (from batting team perspective, deficit is negative)
            # Actually deficit = fld_score - bat_score, so after scoring:
            # new_deficit = deficit - runs_scored
            # If new_deficit < 0 → batting team ahead → win
            # If new_deficit == 0 → tied → extra innings (~50% win for home team)
            # If new_deficit > 0 → still behind → loss (game over in bot 9, 2 out, 3rd out)

            # For outs: game over, loss
            if outcome in ('K', 'other_out'):
                win_pct = 0.0
            else:
                new_deficit = deficit - runs
                wins = (new_deficit < 0).sum()
                ties = (new_deficit == 0).sum()
                losses = (new_deficit > 0).sum()
                # Ties go to extras — home team wins ~52% historically
                win_pct = (wins + 0.52 * ties) / n if n > 0 else 0.0

            we[outcome] = {'win_pct': win_pct, 'n': n}
            print(f"    {outcome}: n={n}, win_pct={win_pct:.3f}")

        we_tables[deficit] = we

    return we_tables


print("\n  Building WE tables from Statcast primary situation outcomes...")
we_tables = build_we_table_from_outcomes()

# Also compute base WE (pre-outcome) for each deficit
# This is the batting team's win probability AT the start of the PA
# Base WE = weighted average of outcome WEs weighted by outcome frequency
base_we = {}
for deficit in DEFICITS:
    if deficit in we_tables:
        total_n = sum(v['n'] for v in we_tables[deficit].values())
        if total_n > 0:
            we = sum(v['win_pct'] * v['n'] for v in we_tables[deficit].values()) / total_n
            base_we[deficit] = we
            print(f"\n  Base WE for deficit={deficit}: {we:.4f} (n={total_n})")

# Compute delta_WP for each outcome
# delta_WP = outcome_WP - base_WP
print("\n  Delta WP by outcome:")
delta_wp_table = {}
for deficit in DEFICITS:
    if deficit not in we_tables or deficit not in base_we:
        continue
    delta_wp_table[deficit] = {}
    bwe = base_we[deficit]
    print(f"\n  Deficit {deficit} (base WE = {bwe:.4f}):")
    for outcome, data in we_tables[deficit].items():
        dwp = data['win_pct'] - bwe
        delta_wp_table[deficit][outcome] = dwp
        print(f"    {outcome}: ΔWP = {dwp:+.4f}")

# For the MDP, we need ΔWP for three terminal outcome classes:
# K (any out), BB (walk/HBP), BIP (ball in play)
print("\n  Simplified ΔWP for MDP terminal states:")
mdp_rewards = {}
for deficit in DEFICITS:
    if deficit not in we_tables or deficit not in base_we:
        continue

    bwe = base_we[deficit]
    we = we_tables[deficit]

    # Out: K + other_out
    out_n = sum(we.get(o, {}).get('n', 0) for o in ['K', 'other_out'])
    out_wp = 0.0  # All outs end the game in a loss

    # BB: walk + HBP
    bb_outcomes = {o: we[o] for o in ['BB', 'HBP'] if o in we}
    bb_n = sum(v['n'] for v in bb_outcomes.values())
    bb_wp = sum(v['win_pct'] * v['n'] for v in bb_outcomes.values()) / bb_n if bb_n > 0 else 0.0

    # BIP: single + double + triple + HR + error
    bip_outcomes = {o: we[o] for o in ['single', 'double', 'triple', 'HR', 'error'] if o in we}
    bip_n = sum(v['n'] for v in bip_outcomes.values())
    bip_wp = sum(v['win_pct'] * v['n'] for v in bip_outcomes.values()) / bip_n if bip_n > 0 else 0.0

    mdp_rewards[deficit] = {
        'base_we': bwe,
        TERMINAL_K: out_wp - bwe,    # Always negative (loss)
        TERMINAL_BB: bb_wp - bwe,    # Positive (forcing in a run helps)
        TERMINAL_BIP: bip_wp - bwe,  # Positive on average (contact produces runs)
    }

    print(f"  Deficit {deficit}: K={out_wp-bwe:+.4f}, BB={bb_wp-bwe:+.4f}, BIP={bip_wp-bwe:+.4f}")

# Save
rewards_df = pd.DataFrame(mdp_rewards).T
rewards_df.index.name = 'deficit'
rewards_df.to_csv(f"{MDP_DIR}/mdp_rewards.csv")
with open(f"{MDP_DIR}/mdp_rewards.json", 'w') as f:
    json.dump(mdp_rewards, f, indent=2)

print("\n  ✓ Win expectancy tables saved to data/mdp/")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Transition Probability Matrices
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 2: Transition Probability Matrices by Era")
print("=" * 70)


def decode_pitch_sequence(seq):
    """
    Decode a Retrosheet PITCH_SEQ_TX string into a list of pitch outcomes.
    Returns list of ('ball', 'called_strike', 'swinging_strike', 'foul', 'contact').
    """
    if not isinstance(seq, str) or seq == 'nan':
        return []

    pitches = []
    for c in seq:
        if c in IGNORE_CHARS:
            continue
        elif c in BALL_CHARS:
            pitches.append('ball')
        elif c in CALLED_STRIKE_CHARS:
            pitches.append('called_strike')
        elif c in SWINGING_STRIKE_CHARS:
            pitches.append('swinging_strike')
        elif c in FOUL_CHARS:
            pitches.append('foul')
        elif c in CONTACT_CHARS:
            pitches.append('contact')
        # else: unknown character, skip

    return pitches


def estimate_transition_probs_from_retrosheet(events_df, era_name):
    """
    Estimate ball/strike/foul/contact/whiff rates from Retrosheet pitch sequences.
    These are league-wide rates, NOT situation-specific.

    Returns dict with:
    - p_ball_on_take: P(ball | batter takes)
    - p_strike_on_take: P(called strike | batter takes)
    - p_whiff_on_swing: P(swinging strike | batter swings)
    - p_foul_on_swing: P(foul | batter swings)
    - p_contact_on_swing: P(ball in play | batter swings)
    """
    # Only batting events with pitch sequences
    bat = events_df[events_df['BAT_EVENT_FL'] == 'T'].copy()
    seqs = bat['PITCH_SEQ_TX'].dropna()
    seqs = seqs[(seqs != '') & (seqs != 'nan')]

    coverage = len(seqs) / len(bat) if len(bat) > 0 else 0
    print(f"  {era_name}: {len(bat):,} PAs, pitch seq coverage = {coverage:.1%}")

    if len(seqs) < 100:
        print(f"    WARNING: Too few pitch sequences for reliable estimation")
        return None

    # Count pitch outcomes across all sequences
    total_balls = 0
    total_called_strikes = 0
    total_swinging_strikes = 0
    total_fouls = 0
    total_contacts = 0

    for seq in seqs:
        pitches = decode_pitch_sequence(seq)
        for p in pitches:
            if p == 'ball':
                total_balls += 1
            elif p == 'called_strike':
                total_called_strikes += 1
            elif p == 'swinging_strike':
                total_swinging_strikes += 1
            elif p == 'foul':
                total_fouls += 1
            elif p == 'contact':
                total_contacts += 1

    total_pitches = total_balls + total_called_strikes + total_swinging_strikes + total_fouls + total_contacts

    if total_pitches == 0:
        return None

    # Compute rates
    # "Take" outcomes: balls and called strikes
    takes = total_balls + total_called_strikes
    p_ball_on_take = total_balls / takes if takes > 0 else 0.5
    p_strike_on_take = total_called_strikes / takes if takes > 0 else 0.5

    # "Swing" outcomes: swinging strikes, fouls, and contact
    swings = total_swinging_strikes + total_fouls + total_contacts
    p_whiff_on_swing = total_swinging_strikes / swings if swings > 0 else 0.33
    p_foul_on_swing = total_fouls / swings if swings > 0 else 0.33
    p_contact_on_swing = total_contacts / swings if swings > 0 else 0.33

    result = {
        'p_ball_on_take': p_ball_on_take,
        'p_strike_on_take': p_strike_on_take,
        'p_whiff_on_swing': p_whiff_on_swing,
        'p_foul_on_swing': p_foul_on_swing,
        'p_contact_on_swing': p_contact_on_swing,
        'total_pitches': total_pitches,
        'pitch_seq_coverage': coverage,
        'total_pas': len(bat),
    }

    print(f"    P(ball|take)={p_ball_on_take:.3f}, P(Cstrike|take)={p_strike_on_take:.3f}")
    print(f"    P(whiff|swing)={p_whiff_on_swing:.3f}, P(foul|swing)={p_foul_on_swing:.3f}, P(contact|swing)={p_contact_on_swing:.3f}")

    return result


def estimate_transition_probs_from_statcast():
    """
    Estimate transition probabilities from Statcast pitch-level data.
    Uses all league data (baseline files), not just situation sample.
    """
    all_takes_ball = 0
    all_takes_strike = 0
    all_swing_whiff = 0
    all_swing_foul = 0
    all_swing_contact = 0
    total_pitches = 0
    total_pas = 0

    for year in range(2015, 2025):
        if year == 2020:
            continue  # Skip COVID season

        # Use batter baseline files (all pitches by situation batters, large sample)
        path = f"{BASE_DIR}/data/statcast/baseline_batter_{year}.csv"
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path, low_memory=False)
        total_pitches += len(df)
        total_pas += df['events'].notna().sum()

        for _, row in df.iterrows():
            desc = str(row.get('description', ''))

            # Classify: is this a swing or take?
            swing_descs = {'swinging_strike', 'foul', 'foul_tip', 'swinging_strike_blocked',
                          'foul_bunt', 'missed_bunt', 'hit_into_play', 'hit_into_play_no_out',
                          'hit_into_play_score', 'foul_pitchout', 'swinging_pitchout'}
            take_descs = {'called_strike', 'ball', 'blocked_ball', 'pitchout',
                         'hit_by_pitch', 'intent_ball'}

            if desc in swing_descs:
                if desc in {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt', 'swinging_pitchout'}:
                    all_swing_whiff += 1
                elif desc in {'foul', 'foul_tip', 'foul_bunt', 'foul_pitchout'}:
                    all_swing_foul += 1
                elif desc in {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}:
                    all_swing_contact += 1
            elif desc in take_descs:
                if desc == 'called_strike':
                    all_takes_strike += 1
                else:
                    all_takes_ball += 1

    takes = all_takes_ball + all_takes_strike
    swings = all_swing_whiff + all_swing_foul + all_swing_contact

    result = {
        'p_ball_on_take': all_takes_ball / takes if takes > 0 else 0.5,
        'p_strike_on_take': all_takes_strike / takes if takes > 0 else 0.5,
        'p_whiff_on_swing': all_swing_whiff / swings if swings > 0 else 0.33,
        'p_foul_on_swing': all_swing_foul / swings if swings > 0 else 0.33,
        'p_contact_on_swing': all_swing_contact / swings if swings > 0 else 0.33,
        'total_pitches': total_pitches,
        'total_pas': total_pas,
        'pitch_seq_coverage': 1.0,
    }

    print(f"  statcast: {total_pitches:,} pitches, {total_pas:,} PAs (excl. 2020)")
    print(f"    P(ball|take)={result['p_ball_on_take']:.3f}, P(Cstrike|take)={result['p_strike_on_take']:.3f}")
    print(f"    P(whiff|swing)={result['p_whiff_on_swing']:.3f}, P(foul|swing)={result['p_foul_on_swing']:.3f}, P(contact|swing)={result['p_contact_on_swing']:.3f}")

    return result


# Estimate transitions for each available era
era_transitions = {}

# Retrosheet eras (use available years)
retro_year_map = {
    'post_war': [1960],
    'expansion': [1990],
    'offense_explosion': [2000],
}

for era_name, years in retro_year_map.items():
    for year in years:
        path = f"{BASE_DIR}/data/retrosheet/events_{year}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            trans = estimate_transition_probs_from_retrosheet(df, f"{era_name} ({year})")
            if trans is not None:
                era_transitions[era_name] = trans
                break

# Statcast era — use pitch-level data directly
# This is slow for the full baseline files, so let's use a sampled approach
print(f"\n  Computing Statcast era transitions from pitch-level data...")
# Instead of iterating row by row through millions of pitches,
# use vectorized operations on the Statcast baseline files
all_desc_counts = {}
total_sc_pitches = 0
total_sc_pas = 0

for year in range(2015, 2025):
    if year == 2020:
        continue
    path = f"{BASE_DIR}/data/statcast/baseline_batter_{year}.csv"
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path, usecols=['description', 'events'], low_memory=False)
    total_sc_pitches += len(df)
    total_sc_pas += df['events'].notna().sum()
    counts = df['description'].value_counts()
    for desc, n in counts.items():
        all_desc_counts[desc] = all_desc_counts.get(desc, 0) + n

# Classify descriptions
swing_whiff_descs = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt', 'swinging_pitchout'}
swing_foul_descs = {'foul', 'foul_tip', 'foul_bunt', 'foul_pitchout'}
swing_contact_descs = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
take_ball_descs = {'ball', 'blocked_ball', 'pitchout', 'hit_by_pitch', 'intent_ball'}
take_strike_descs = {'called_strike'}

takes_ball = sum(all_desc_counts.get(d, 0) for d in take_ball_descs)
takes_strike = sum(all_desc_counts.get(d, 0) for d in take_strike_descs)
swings_whiff = sum(all_desc_counts.get(d, 0) for d in swing_whiff_descs)
swings_foul = sum(all_desc_counts.get(d, 0) for d in swing_foul_descs)
swings_contact = sum(all_desc_counts.get(d, 0) for d in swing_contact_descs)

takes_total = takes_ball + takes_strike
swings_total = swings_whiff + swings_foul + swings_contact

statcast_trans = {
    'p_ball_on_take': takes_ball / takes_total if takes_total > 0 else 0.5,
    'p_strike_on_take': takes_strike / takes_total if takes_total > 0 else 0.5,
    'p_whiff_on_swing': swings_whiff / swings_total if swings_total > 0 else 0.33,
    'p_foul_on_swing': swings_foul / swings_total if swings_total > 0 else 0.33,
    'p_contact_on_swing': swings_contact / swings_total if swings_total > 0 else 0.33,
    'total_pitches': total_sc_pitches,
    'total_pas': total_sc_pas,
    'pitch_seq_coverage': 1.0,
}
era_transitions['statcast'] = statcast_trans

print(f"  statcast: {total_sc_pitches:,} pitches, {total_sc_pas:,} PAs (excl. 2020)")
print(f"    P(ball|take)={statcast_trans['p_ball_on_take']:.3f}, P(Cstrike|take)={statcast_trans['p_strike_on_take']:.3f}")
print(f"    P(whiff|swing)={statcast_trans['p_whiff_on_swing']:.3f}, P(foul|swing)={statcast_trans['p_foul_on_swing']:.3f}, P(contact|swing)={statcast_trans['p_contact_on_swing']:.3f}")

# Save transition parameters
trans_df = pd.DataFrame(era_transitions).T
trans_df.index.name = 'era'
trans_df.to_csv(f"{MDP_DIR}/transition_parameters.csv")
print(f"\n  Transition parameters across eras:")
print(trans_df[['p_ball_on_take', 'p_strike_on_take', 'p_whiff_on_swing',
                'p_foul_on_swing', 'p_contact_on_swing']].to_string())


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Value Iteration Solver
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 3: Value Iteration Solver")
print("=" * 70)


def build_transition_dict(trans_params):
    """
    Build the full MDP transition dictionary from estimated parameters.

    State: (balls, strikes) — 12 live states (0-3 balls × 0-2 strikes)
    Actions: 'take', 'swing'
    Terminal states: 'K', 'BB', 'BIP'

    Returns: transitions[state][action] = {next_state: probability}
    """
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

            # Action: TAKE
            take_trans = {}
            if b == 3:
                # 3 balls: another ball → walk
                take_trans[TERMINAL_BB] = take_trans.get(TERMINAL_BB, 0) + pb
            else:
                take_trans[(b + 1, s)] = pb

            if s == 2:
                # 2 strikes: called strike → strikeout
                take_trans[TERMINAL_K] = take_trans.get(TERMINAL_K, 0) + ps
            else:
                take_trans[(b, s + 1)] = ps

            T[state]['take'] = take_trans

            # Action: SWING
            swing_trans = {}
            # Swinging strike (whiff)
            if s == 2:
                swing_trans[TERMINAL_K] = swing_trans.get(TERMINAL_K, 0) + pw
            else:
                swing_trans[(b, s + 1)] = swing_trans.get((b, s + 1), 0) + pw

            # Foul ball
            if s < 2:
                swing_trans[(b, s + 1)] = swing_trans.get((b, s + 1), 0) + pf
            else:
                # Foul with 2 strikes → stays at 2 strikes
                swing_trans[(b, 2)] = swing_trans.get((b, 2), 0) + pf

            # Contact → ball in play (terminal)
            swing_trans[TERMINAL_BIP] = swing_trans.get(TERMINAL_BIP, 0) + pc

            T[state]['swing'] = swing_trans

    return T


def value_iteration(trans_params, rewards_for_deficit, tol=1e-8, max_iter=1000):
    """
    Solve the batter MDP via value iteration.

    trans_params: dict with p_ball_on_take, p_strike_on_take, etc.
    rewards_for_deficit: {TERMINAL_K: ΔWP, TERMINAL_BB: ΔWP, TERMINAL_BIP: ΔWP}

    Returns: (V, policy) where V[state] = optimal value, policy[state] = optimal action
    """
    T = build_transition_dict(trans_params)

    # Initialize value function
    # Terminal states have fixed values (the ΔWP rewards)
    terminal_values = {
        TERMINAL_K: rewards_for_deficit[TERMINAL_K],
        TERMINAL_BB: rewards_for_deficit[TERMINAL_BB],
        TERMINAL_BIP: rewards_for_deficit[TERMINAL_BIP],
    }

    V = {(b, s): 0.0 for b in range(4) for s in range(3)}
    policy = {(b, s): None for b in range(4) for s in range(3)}

    for iteration in range(max_iter):
        V_new = {}
        for state in V:
            action_values = {}
            for action in ['take', 'swing']:
                ev = 0.0
                for next_state, prob in T[state][action].items():
                    if next_state in terminal_values:
                        ev += prob * terminal_values[next_state]
                    else:
                        ev += prob * V[next_state]
                action_values[action] = ev

            best_action = max(action_values, key=action_values.get)
            V_new[state] = action_values[best_action]
            policy[state] = best_action

        delta = max(abs(V_new[s] - V[s]) for s in V)
        V = V_new

        if delta < tol:
            return V, policy, iteration + 1, T

    return V, policy, max_iter, T


# Solve for each era × deficit combination
all_solutions = {}
policy_tables = []

for era_name, trans_params in era_transitions.items():
    print(f"\n  Era: {era_name}")

    for deficit in DEFICITS:
        if deficit not in mdp_rewards:
            print(f"    Deficit {deficit}: no reward data, skipping")
            continue

        rewards = mdp_rewards[deficit]
        V, policy, n_iter, T = value_iteration(trans_params, rewards)

        key = (era_name, deficit)
        all_solutions[key] = {'V': V, 'policy': policy, 'T': T, 'n_iter': n_iter}

        # Record policy for output
        for (b, s), action in policy.items():
            policy_tables.append({
                'era': era_name,
                'deficit': deficit,
                'balls': b,
                'strikes': s,
                'count': f"{b}-{s}",
                'optimal_action': action,
                'value': V[(b, s)],
            })

        print(f"    Deficit {deficit}: converged in {n_iter} iterations, V(0-0)={V[(0,0)]:.4f}")

policy_df = pd.DataFrame(policy_tables)
policy_df.to_csv(f"{MDP_DIR}/optimal_policies.csv", index=False)

# Display policy grids
print(f"\n  ── Optimal Policy Grids ──")
for deficit in DEFICITS:
    if deficit not in mdp_rewards:
        continue
    print(f"\n  Deficit = {deficit}:")
    header = f"  {'Count':<8}"
    for era_name in era_transitions:
        header += f" {era_name:>20}"
    print(header)

    for b in range(4):
        for s in range(3):
            count = f"{b}-{s}"
            row = f"  {count:<8}"
            for era_name in era_transitions:
                key = (era_name, deficit)
                if key in all_solutions:
                    action = all_solutions[key]['policy'][(b, s)]
                    value = all_solutions[key]['V'][(b, s)]
                    row += f" {action:>10} ({value:+.3f})"
                else:
                    row += f" {'N/A':>20}"
            print(row)

# Validation checks
print(f"\n  ── Validation Checks ──")
for era_name in era_transitions:
    key_0 = (era_name, 0)
    if key_0 not in all_solutions:
        continue
    V = all_solutions[key_0]['V']
    policy = all_solutions[key_0]['policy']

    checks = []
    # V(0-0) should be ~0.35-0.50
    v00 = V[(0, 0)]
    checks.append(f"V(0-0)={v00:.4f} ({'✓' if 0.1 < v00 < 0.7 else '✗'} expected 0.15-0.60)")

    # V(3-0) should be highest
    v30 = V[(3, 0)]
    max_v = max(V.values())
    checks.append(f"V(3-0)={v30:.4f} ({'✓' if v30 == max_v else '✗'} should be highest)")

    # V(0-2) should be lowest
    v02 = V[(0, 2)]
    min_v = min(V.values())
    checks.append(f"V(0-2)={v02:.4f} ({'✓' if v02 == min_v else '✗'} should be lowest)")

    # 3-0 with deficit=0 should be TAKE (walk wins the game)
    p30 = policy[(3, 0)]
    checks.append(f"π(3-0, deficit=0)={p30} ({'✓' if p30 == 'take' else '✗'} should be take)")

    print(f"  {era_name}: {' | '.join(checks)}")

# Check: deficit=-3, should swing more
for era_name in era_transitions:
    key_3 = (era_name, -3)
    if key_3 not in all_solutions:
        continue
    policy = all_solutions[key_3]['policy']
    swing_count = sum(1 for a in policy.values() if a == 'swing')
    key_0 = (era_name, 0)
    if key_0 in all_solutions:
        policy_0 = all_solutions[key_0]['policy']
        swing_count_0 = sum(1 for a in policy_0.values() if a == 'swing')
        print(f"  {era_name}: swing states at deficit=0: {swing_count_0}/12, deficit=-3: {swing_count}/12 ({'✓' if swing_count >= swing_count_0 else '✗'} should be ≥)")


# Save value functions
vf_tables = []
for (era_name, deficit), sol in all_solutions.items():
    for (b, s), val in sol['V'].items():
        vf_tables.append({
            'era': era_name, 'deficit': deficit,
            'balls': b, 'strikes': s, 'count': f"{b}-{s}",
            'value': val,
        })
vf_df = pd.DataFrame(vf_tables)
vf_df.to_csv(f"{MDP_DIR}/value_functions.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Policy Gap Metric
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 4: Policy Gap Metric")
print("=" * 70)


def compute_policy_gap(observed_swing_rate, optimal_action, V, T, rewards_terminal):
    """
    Compute the WP cost of observed behavior vs optimal at a given state.

    observed_swing_rate: fraction of pitches swung at (0.0 to 1.0)
    optimal_action: 'take' or 'swing'
    V: value function
    T: transition dict
    rewards_terminal: {K: ΔWP, BB: ΔWP, BIP: ΔWP}

    Returns: gap (negative = suboptimal, 0 = optimal)
    """
    terminal_values = {
        TERMINAL_K: rewards_terminal[TERMINAL_K],
        TERMINAL_BB: rewards_terminal[TERMINAL_BB],
        TERMINAL_BIP: rewards_terminal[TERMINAL_BIP],
    }

    def expected_value(state, action_probs):
        ev = 0.0
        for action, prob in action_probs.items():
            for next_state, trans_prob in T[state][action].items():
                if next_state in terminal_values:
                    ev += prob * trans_prob * terminal_values[next_state]
                else:
                    ev += prob * trans_prob * V[next_state]
        return ev

    return None  # Placeholder — computed inline below


# Compute observed swing rates by count from Statcast situation data
print("\n  Computing observed swing rates by count in primary situation...")

swing_descs = {'swinging_strike', 'foul', 'foul_tip', 'swinging_strike_blocked',
               'foul_bunt', 'missed_bunt', 'hit_into_play', 'hit_into_play_no_out',
               'hit_into_play_score', 'foul_pitchout', 'swinging_pitchout'}

all_pitches = []
for year in range(2015, 2025):
    path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        df['is_swing'] = df['description'].isin(swing_descs).astype(int)
        df['deficit'] = df['fld_score'] - df['bat_score']
        df['year'] = year
        all_pitches.append(df[['balls', 'strikes', 'description', 'is_swing', 'deficit', 'year']])

all_pitches_df = pd.concat(all_pitches, ignore_index=True)

# Compute swing rate by (balls, strikes, deficit)
swing_rates = all_pitches_df.groupby(['balls', 'strikes', 'deficit'])['is_swing'].agg(['mean', 'count']).reset_index()
swing_rates.columns = ['balls', 'strikes', 'deficit', 'swing_rate', 'n_pitches']

print(f"\n  Observed swing rates (primary situation, 2015-2024):")
for deficit in DEFICITS:
    sub = swing_rates[swing_rates['deficit'] == deficit]
    if len(sub) == 0:
        continue
    print(f"\n  Deficit {deficit}:")
    for _, row in sub.iterrows():
        print(f"    {int(row['balls'])}-{int(row['strikes'])}: swing_rate={row['swing_rate']:.3f} (n={int(row['n_pitches'])})")

# Compute policy gaps
print(f"\n  ── Policy Gaps (Statcast era, primary situation) ──")

terminal_values_map = {
    TERMINAL_K: 'K',
    TERMINAL_BB: 'BB',
    TERMINAL_BIP: 'BIP',
}

policy_gap_results = []

for deficit in DEFICITS:
    if deficit not in mdp_rewards:
        continue

    key = ('statcast', deficit)
    if key not in all_solutions:
        continue

    sol = all_solutions[key]
    V = sol['V']
    T = sol['T']
    rewards = mdp_rewards[deficit]

    terminal_vals = {
        TERMINAL_K: rewards[TERMINAL_K],
        TERMINAL_BB: rewards[TERMINAL_BB],
        TERMINAL_BIP: rewards[TERMINAL_BIP],
    }

    sub = swing_rates[swing_rates['deficit'] == deficit]

    print(f"\n  Deficit {deficit}:")
    print(f"  {'Count':<8} {'Obs Swing%':>12} {'Optimal':>10} {'V_obs':>10} {'V_opt':>10} {'Gap':>10} {'N':>6}")

    for b in range(4):
        for s in range(3):
            state = (b, s)
            row = sub[(sub['balls'] == b) & (sub['strikes'] == s)]

            if len(row) == 0:
                continue

            obs_swing = row['swing_rate'].values[0]
            n_pitches = int(row['n_pitches'].values[0])
            optimal_action = sol['policy'][state]
            v_optimal = V[state]

            # Compute V_observed
            obs_action_probs = {'swing': obs_swing, 'take': 1.0 - obs_swing}
            v_observed = 0.0
            for action, aprob in obs_action_probs.items():
                for next_state, trans_prob in T[state][action].items():
                    if next_state in terminal_vals:
                        v_observed += aprob * trans_prob * terminal_vals[next_state]
                    else:
                        v_observed += aprob * trans_prob * V[next_state]

            gap = v_observed - v_optimal

            policy_gap_results.append({
                'era': 'statcast',
                'deficit': deficit,
                'balls': b,
                'strikes': s,
                'count': f"{b}-{s}",
                'obs_swing_rate': obs_swing,
                'optimal_action': optimal_action,
                'v_observed': v_observed,
                'v_optimal': v_optimal,
                'gap': gap,
                'n_pitches': n_pitches,
            })

            print(f"  {b}-{s:<6} {obs_swing:>12.3f} {optimal_action:>10} {v_observed:>10.4f} {v_optimal:>10.4f} {gap:>+10.4f} {n_pitches:>6}")

gap_df = pd.DataFrame(policy_gap_results)
gap_df.to_csv(f"{MDP_DIR}/policy_gaps_statcast.csv", index=False)

# Aggregate gap statistics
print(f"\n  ── Aggregate Policy Gap ──")
for deficit in DEFICITS:
    sub = gap_df[gap_df['deficit'] == deficit]
    if len(sub) == 0:
        continue
    # Weighted by n_pitches
    weighted_gap = (sub['gap'] * sub['n_pitches']).sum() / sub['n_pitches'].sum()
    mean_gap = sub['gap'].mean()
    worst_state = sub.loc[sub['gap'].idxmin()]
    print(f"  Deficit {deficit}: mean gap={mean_gap:.4f}, weighted gap={weighted_gap:.4f}, worst={worst_state['count']} ({worst_state['gap']:+.4f})")


# Also compute per-year policy gaps for longitudinal tracking
print(f"\n  ── Per-Year Policy Gaps (tracking longitudinal trend) ──")
yearly_gaps = []

for year in range(2015, 2025):
    path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    if not os.path.exists(path):
        continue

    df = pd.read_csv(path, low_memory=False)
    df['is_swing'] = df['description'].isin(swing_descs).astype(int)
    df['deficit'] = df['fld_score'] - df['bat_score']

    year_swing = df.groupby(['balls', 'strikes', 'deficit'])['is_swing'].agg(['mean', 'count']).reset_index()
    year_swing.columns = ['balls', 'strikes', 'deficit', 'swing_rate', 'n_pitches']

    year_total_gap = 0.0
    year_total_n = 0

    for deficit in DEFICITS:
        key = ('statcast', deficit)
        if key not in all_solutions or deficit not in mdp_rewards:
            continue

        sol = all_solutions[key]
        V = sol['V']
        T = sol['T']
        rewards = mdp_rewards[deficit]
        terminal_vals = {TERMINAL_K: rewards[TERMINAL_K], TERMINAL_BB: rewards[TERMINAL_BB], TERMINAL_BIP: rewards[TERMINAL_BIP]}

        sub = year_swing[year_swing['deficit'] == deficit]

        for _, row in sub.iterrows():
            b, s = int(row['balls']), int(row['strikes'])
            state = (b, s)
            if state not in V:
                continue

            obs_swing = row['swing_rate']
            n = int(row['n_pitches'])

            obs_probs = {'swing': obs_swing, 'take': 1.0 - obs_swing}
            v_obs = sum(
                obs_probs[a] * sum(
                    T[state][a].get(ns, 0) * (terminal_vals.get(ns, 0) if ns in terminal_vals else V.get(ns, 0))
                    for ns in T[state][a]
                )
                for a in obs_probs
            )

            gap = v_obs - V[state]
            year_total_gap += gap * n
            year_total_n += n

    if year_total_n > 0:
        avg_gap = year_total_gap / year_total_n
        yearly_gaps.append({'year': year, 'weighted_avg_gap': avg_gap, 'total_pitches': year_total_n})
        print(f"  {year}: weighted gap = {avg_gap:+.4f} (n={year_total_n})")

yearly_gap_df = pd.DataFrame(yearly_gaps)
yearly_gap_df.to_csv(f"{MDP_DIR}/yearly_policy_gaps.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 5: Sensitivity Analysis")
print("=" * 70)

perturbation = 0.10  # ±10%
sensitivity_results = []

for era_name, trans_params in era_transitions.items():
    for deficit in DEFICITS:
        if deficit not in mdp_rewards:
            continue

        key = (era_name, deficit)
        if key not in all_solutions:
            continue

        base_policy = all_solutions[key]['policy']

        # Perturb each parameter ±10% and check if policy changes
        param_keys = ['p_ball_on_take', 'p_strike_on_take', 'p_whiff_on_swing',
                      'p_foul_on_swing', 'p_contact_on_swing']

        for param in param_keys:
            for direction in [+1, -1]:
                perturbed = trans_params.copy()
                perturbed[param] = trans_params[param] * (1 + direction * perturbation)

                # Renormalize: take probs must sum to 1, swing probs must sum to 1
                if param in ['p_ball_on_take', 'p_strike_on_take']:
                    total = perturbed['p_ball_on_take'] + perturbed['p_strike_on_take']
                    perturbed['p_ball_on_take'] /= total
                    perturbed['p_strike_on_take'] /= total
                else:
                    total = perturbed['p_whiff_on_swing'] + perturbed['p_foul_on_swing'] + perturbed['p_contact_on_swing']
                    perturbed['p_whiff_on_swing'] /= total
                    perturbed['p_foul_on_swing'] /= total
                    perturbed['p_contact_on_swing'] /= total

                V_pert, policy_pert, _, _ = value_iteration(perturbed, mdp_rewards[deficit])

                # Count policy flips
                flips = []
                for state in base_policy:
                    if base_policy[state] != policy_pert[state]:
                        flips.append(state)

                if flips:
                    for flip_state in flips:
                        sensitivity_results.append({
                            'era': era_name, 'deficit': deficit,
                            'param': param, 'direction': f"{'+' if direction > 0 else '-'}{perturbation*100:.0f}%",
                            'flip_state': f"{flip_state[0]}-{flip_state[1]}",
                            'base_action': base_policy[flip_state],
                            'perturbed_action': policy_pert[flip_state],
                        })

sens_df = pd.DataFrame(sensitivity_results)
sens_df.to_csv(f"{MDP_DIR}/sensitivity_analysis.csv", index=False)

if len(sens_df) > 0:
    print(f"\n  Found {len(sens_df)} policy flips under ±{perturbation*100:.0f}% perturbation:")

    # Identify boundary states (states that flip under any perturbation)
    boundary_states = sens_df.groupby(['era', 'deficit', 'flip_state']).size().reset_index(name='n_flips')
    print(f"\n  Boundary states (policy sensitive to parameter uncertainty):")
    for _, row in boundary_states.iterrows():
        print(f"    {row['era']}, deficit={row['deficit']}, count={row['flip_state']}: flips in {row['n_flips']} scenarios")

    # For reporting: which counts have robust policies?
    print(f"\n  Per-era robustness:")
    for era_name in era_transitions:
        era_sens = sens_df[sens_df['era'] == era_name]
        if len(era_sens) == 0:
            print(f"    {era_name}: All policies robust (no flips)")
        else:
            boundary = era_sens['flip_state'].unique()
            print(f"    {era_name}: Boundary states = {list(boundary)}")
else:
    print(f"\n  ✓ All policies are robust under ±{perturbation*100:.0f}% perturbation — no flips detected")

print(f"\n{'='*70}")
print(" ALL STEPS COMPLETE")
print("=" * 70)
print(f"\nOutput files in data/mdp/:")
for f in sorted(os.listdir(MDP_DIR)):
    path = os.path.join(MDP_DIR, f)
    size = os.path.getsize(path)
    print(f"  {f} ({size:,} bytes)")
