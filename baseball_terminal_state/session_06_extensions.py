"""
Session 6: Model Extensions, Stakes Fix & Visualization
========================================================
Step 1: Batter pitch history state augmentation (768-state MDP)
Step 3: Stakes enrichment with proper standings data
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

BASE_DIR = "/Users/paul821/Desktop/baseball/baseball_terminal_state"
MDP_DIR = f"{BASE_DIR}/data/mdp"
LONG_DIR = f"{BASE_DIR}/data/longitudinal"
STAKES_DIR = f"{BASE_DIR}/data/stakes"
EXT_DIR = f"{BASE_DIR}/data/extensions"
FIG_DIR = f"{BASE_DIR}/figures"
for d in [EXT_DIR, STAKES_DIR]:
    os.makedirs(d, exist_ok=True)

DEFICITS = [0, 1, 2, 3]
EXTRAS_WP = 0.52
FASTBALL_TYPES = {'FF', 'SI', 'FC'}  # four-seam, sinker, cutter

SWING_DESCS = {'swinging_strike', 'foul', 'foul_tip', 'swinging_strike_blocked',
               'foul_bunt', 'missed_bunt', 'hit_into_play', 'hit_into_play_no_out',
               'hit_into_play_score', 'foul_pitchout', 'swinging_pitchout'}

# Load transition parameters and baseline BIP distribution from Session 5
trans_df = pd.read_csv(f"{MDP_DIR}/transition_parameters.csv", index_col=0)
statcast_trans = trans_df.loc['statcast'].to_dict()

# Recover baseline BIP dist
all_primary = []
for year in range(2015, 2025):
    path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        all_primary.append(df)

primary_all = pd.concat(all_primary, ignore_index=True)

out_events = {'field_out', 'grounded_into_double_play', 'force_out',
              'fielders_choice', 'fielders_choice_out', 'double_play',
              'sac_fly', 'sac_bunt', 'triple_play', 'sac_fly_double_play'}
hit_events = {'single', 'double', 'triple', 'home_run', 'field_error'}

terminal = primary_all[primary_all['events'].notna()].copy()
terminal['deficit'] = terminal['fld_score'] - terminal['bat_score']
if 'post_bat_score' in terminal.columns:
    terminal['runs_scored'] = (terminal['post_bat_score'] - terminal['bat_score']).fillna(0).astype(int)

bip_all = terminal[terminal['events'].isin(out_events | hit_events)]
bip_outs = terminal[terminal['events'].isin(out_events)]
bip_hits = terminal[terminal['events'].isin(hit_events)]

p_bip_out = len(bip_outs) / len(bip_all) if len(bip_all) > 0 else 0.5
p_bip_hit = 1 - p_bip_out
hit_runs_dist = bip_hits['runs_scored'].value_counts(normalize=True).sort_index().to_dict()

baseline_bip = {0: p_bip_out}
for r, p in hit_runs_dist.items():
    baseline_bip[r] = baseline_bip.get(r, 0) + p_bip_hit * p


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: PITCH HISTORY STATE AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(" STEP 1: Pitch History State Augmentation")
print("=" * 70)

# ── Estimate P(fastball | count, fastballs_seen) from ALL Statcast data ──
# Use the full situation sample across all years.
# Group pitches by at-bat, sort by pitch_number, compute fastballs_seen.

print("\n  Computing P(fastball | count, fastballs_seen) from Statcast...")

# Build pitch sequences per PA
pa_sequences = []
for year in range(2015, 2025):
    path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path, low_memory=False)
    df['year'] = year

    # Group by game + at-bat
    for (gpk, abn), group in df.groupby(['game_pk', 'at_bat_number']):
        group = group.sort_values('pitch_number')
        fb_count = 0
        for _, pitch in group.iterrows():
            pa_sequences.append({
                'balls': int(pitch['balls']),
                'strikes': int(pitch['strikes']),
                'fastballs_seen': min(fb_count, 3),
                'is_fastball': 1 if pitch['pitch_type'] in FASTBALL_TYPES else 0,
                'pitch_type': pitch['pitch_type'],
                'description': pitch['description'],
                'deficit': int(pitch['fld_score'] - pitch['bat_score']),
            })
            if pitch['pitch_type'] in FASTBALL_TYPES:
                fb_count += 1

pa_seq_df = pd.DataFrame(pa_sequences)
print(f"  Total pitch observations: {len(pa_seq_df):,}")

# Compute P(fastball | count, fastballs_seen)
fb_prob = pa_seq_df.groupby(['balls', 'strikes', 'fastballs_seen']).agg(
    p_fastball=('is_fastball', 'mean'),
    n=('is_fastball', 'count'),
).reset_index()

print(f"\n  P(fastball | count, fastballs_seen) — key states:")
print(f"  {'Count':>5} {'FB_seen':>7} {'P(FB)':>7} {'N':>6}")
for _, row in fb_prob.sort_values(['balls', 'strikes', 'fastballs_seen']).iterrows():
    if row['n'] >= 10:
        print(f"  {int(row['balls'])}-{int(row['strikes']):>1}   {int(row['fastballs_seen']):>5}   "
              f"{row['p_fastball']:>5.3f}  {int(row['n']):>5}")

# ── Build transition dict for FB-augmented state space ───────────────────
# Now: the key question is HOW fastballs_seen affects transitions.
# Approach: fastballs_seen changes the pitch MIX the batter faces, which
# in turn affects p(ball), p(strike), p(whiff), p(foul), p(contact).
#
# Estimate these conditional on fastball vs. offspeed:
print(f"\n  Estimating outcome probabilities by pitch category...")

# Classify pitch outcomes
def classify_outcome(desc):
    if desc in {'called_strike'}:
        return 'called_strike'
    elif desc in {'ball', 'blocked_ball', 'pitchout', 'intent_ball'}:
        return 'ball'
    elif desc in {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt', 'swinging_pitchout'}:
        return 'whiff'
    elif desc in {'foul', 'foul_tip', 'foul_bunt', 'foul_pitchout'}:
        return 'foul'
    elif desc in {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}:
        return 'contact'
    else:
        return 'other'

pa_seq_df['outcome'] = pa_seq_df['description'].apply(classify_outcome)
pa_seq_df = pa_seq_df[pa_seq_df['outcome'] != 'other']

# Outcome probabilities conditional on fastball vs. offspeed
for pitch_cat, label in [(True, 'Fastball'), (False, 'Offspeed')]:
    sub = pa_seq_df[pa_seq_df['is_fastball'] == (1 if pitch_cat else 0)]
    if len(sub) == 0:
        continue
    counts = sub['outcome'].value_counts(normalize=True)
    print(f"\n  {label} (n={len(sub):,}):")
    for outcome in ['ball', 'called_strike', 'whiff', 'foul', 'contact']:
        print(f"    {outcome:15s}: {counts.get(outcome, 0):.4f}")

# Now compute composite transition probabilities as a function of P(fastball)
fb_outcomes = pa_seq_df[pa_seq_df['is_fastball'] == 1]['outcome'].value_counts(normalize=True)
os_outcomes = pa_seq_df[pa_seq_df['is_fastball'] == 0]['outcome'].value_counts(normalize=True)

def get_composite_probs(p_fb):
    """Get composite take/swing outcome probabilities given P(fastball)."""
    p_os = 1 - p_fb

    # Take outcomes: ball or called_strike
    p_ball_fb = fb_outcomes.get('ball', 0) / (fb_outcomes.get('ball', 0) + fb_outcomes.get('called_strike', 0) + 1e-10)
    p_ball_os = os_outcomes.get('ball', 0) / (os_outcomes.get('ball', 0) + os_outcomes.get('called_strike', 0) + 1e-10)

    # P(ball | take) depends on pitch mix
    take_rate = (fb_outcomes.get('ball', 0) + fb_outcomes.get('called_strike', 0)) * p_fb + \
                (os_outcomes.get('ball', 0) + os_outcomes.get('called_strike', 0)) * p_os

    p_ball_on_take = (fb_outcomes.get('ball', 0) * p_fb + os_outcomes.get('ball', 0) * p_os) / take_rate if take_rate > 0 else 0.5
    p_strike_on_take = 1 - p_ball_on_take

    # Swing outcomes: whiff, foul, contact
    swing_fb = fb_outcomes.get('whiff', 0) + fb_outcomes.get('foul', 0) + fb_outcomes.get('contact', 0)
    swing_os = os_outcomes.get('whiff', 0) + os_outcomes.get('foul', 0) + os_outcomes.get('contact', 0)

    sw_total = swing_fb * p_fb + swing_os * p_os
    if sw_total > 0:
        p_whiff = (fb_outcomes.get('whiff', 0) * p_fb + os_outcomes.get('whiff', 0) * p_os) / sw_total
        p_foul = (fb_outcomes.get('foul', 0) * p_fb + os_outcomes.get('foul', 0) * p_os) / sw_total
        p_contact = (fb_outcomes.get('contact', 0) * p_fb + os_outcomes.get('contact', 0) * p_os) / sw_total
    else:
        p_whiff, p_foul, p_contact = 0.22, 0.40, 0.38

    return {
        'p_ball_on_take': p_ball_on_take,
        'p_strike_on_take': p_strike_on_take,
        'p_whiff_on_swing': p_whiff,
        'p_foul_on_swing': p_foul,
        'p_contact_on_swing': p_contact,
    }


# Build lookup: P(fastball) for each (balls, strikes, fastballs_seen)
fb_prob_lookup = {}
for _, row in fb_prob.iterrows():
    key = (int(row['balls']), int(row['strikes']), int(row['fastballs_seen']))
    fb_prob_lookup[key] = row['p_fastball']

# Fallback: use count-level average
count_fb_avg = pa_seq_df.groupby(['balls', 'strikes'])['is_fastball'].mean().to_dict()
global_fb_avg = pa_seq_df['is_fastball'].mean()


def get_p_fastball(b, s, fb_seen):
    key = (b, s, fb_seen)
    if key in fb_prob_lookup:
        return fb_prob_lookup[key]
    # Fallback to count average
    if (b, s) in count_fb_avg:
        return count_fb_avg[(b, s)]
    return global_fb_avg


# ── Build augmented V3 transition dict and solve ─────────────────────────
print(f"\n  Building 768-state augmented MDP...")

def build_v3_transition_dict():
    """State: (balls, strikes, deficit, runs_scored, fastballs_seen)"""
    T = {}
    for b in range(4):
        for s in range(3):
            for d in DEFICITS:
                for r in range(4):
                    for fb in range(4):
                        state = (b, s, d, r, fb)
                        p_fb = get_p_fastball(b, s, fb)
                        probs = get_composite_probs(p_fb)

                        pb = probs['p_ball_on_take']
                        ps = probs['p_strike_on_take']
                        pw = probs['p_whiff_on_swing']
                        pf = probs['p_foul_on_swing']
                        pc = probs['p_contact_on_swing']

                        T[state] = {}

                        # How does fastballs_seen update?
                        # On a take: the pitch was not swung at.
                        # If it was a ball: could be FB or OS. Expected fb_new:
                        #   P(was FB | ball) * (fb+1) + P(was OS | ball) * fb
                        # Simplify: use expected fastball given count
                        fb_if_ball = min(fb + (1 if p_fb > 0.5 else 0), 3)  # heuristic
                        fb_if_strike = min(fb + (1 if p_fb > 0.5 else 0), 3)
                        # For swing outcomes: batter saw the pitch
                        fb_if_swing = min(fb + (1 if p_fb > 0.5 else 0), 3)

                        # TAKE
                        take = {}
                        if b == 3:
                            if d == 0:
                                take[('WIN',)] = take.get(('WIN',), 0) + pb
                            else:
                                next_state = (0, 0, d - 1, min(r + 1, 3), 0)  # reset fb on walk
                                take[next_state] = take.get(next_state, 0) + pb
                        else:
                            take[(b + 1, s, d, r, fb_if_ball)] = pb

                        if s == 2:
                            take[('K', d, r)] = take.get(('K', d, r), 0) + ps
                        else:
                            take[(b, s + 1, d, r, fb_if_strike)] = ps
                        T[state]['take'] = take

                        # SWING
                        swing = {}
                        if s == 2:
                            swing[('K', d, r)] = swing.get(('K', d, r), 0) + pw
                        else:
                            swing[(b, s + 1, d, r, fb_if_swing)] = swing.get((b, s + 1, d, r, fb_if_swing), 0) + pw

                        if s < 2:
                            swing[(b, s + 1, d, r, fb_if_swing)] = swing.get((b, s + 1, d, r, fb_if_swing), 0) + pf
                        else:
                            swing[(b, 2, d, r, fb_if_swing)] = swing.get((b, 2, d, r, fb_if_swing), 0) + pf

                        swing[('BIP', d, r)] = swing.get(('BIP', d, r), 0) + pc
                        T[state]['swing'] = swing

    return T


def value_iteration_v3(bip_dist, tol=1e-8, max_iter=1000):
    T = build_v3_transition_dict()
    V = {}
    policy = {}
    for b in range(4):
        for s in range(3):
            for d in DEFICITS:
                for r in range(4):
                    for fb in range(4):
                        V[(b, s, d, r, fb)] = 0.0

    for iteration in range(max_iter):
        V_new = {}
        for state in V:
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
                            rem = bip_d - runs
                            if rem < 0:
                                bip_ev += rp * 1.0
                            elif rem == 0:
                                bip_ev += rp * EXTRAS_WP
                            else:
                                if runs > 0 and rem in DEFICITS:
                                    # Next batter starts fresh (fb=0)
                                    bip_ev += rp * V.get((0, 0, rem, min(bip_r + runs, 3), 0), 0.0)
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
            return V, policy, iteration + 1
    return V, policy, max_iter


print(f"  Solving 768-state MDP...")
V3, policy3, n_iter3 = value_iteration_v3(baseline_bip)
print(f"  Converged in {n_iter3} iterations")

# ── Compare to Session 5 baseline (192-state) ───────────────────────────
print(f"\n  ── Comparison to 192-state baseline ──")

# Load baseline
baseline_pol = pd.read_csv(f"{MDP_DIR}/optimal_policies_v2.csv")
baseline_dict = {}
for _, row in baseline_pol[baseline_pol['era'] == 'statcast'].iterrows():
    state = (int(row['balls']), int(row['strikes']), int(row['deficit']))
    baseline_dict[state] = row['optimal_action']

# Compare: for each (b, s, d), check if the v3 policy differs across fb_seen values
flips_vs_baseline = []
policy_by_count = {}
for b in range(4):
    for s in range(3):
        for d in DEFICITS:
            base_action = baseline_dict.get((b, s, d), 'take')
            actions_by_fb = {}
            for fb in range(4):
                state = (b, s, d, 0, fb)  # r=0
                actions_by_fb[fb] = policy3[state]

            policy_by_count[(b, s, d)] = actions_by_fb

            # Check if any fb_seen value flips vs baseline
            for fb, action in actions_by_fb.items():
                if action != base_action:
                    flips_vs_baseline.append({
                        'count': f"{b}-{s}", 'deficit': d,
                        'fastballs_seen': fb,
                        'baseline_action': base_action,
                        'v3_action': action,
                        'v3_value': V3[(b, s, d, 0, fb)],
                    })

print(f"  Total state flips vs. baseline: {len(flips_vs_baseline)}")
if flips_vs_baseline:
    for flip in flips_vs_baseline:
        print(f"    {flip['count']} d={flip['deficit']} fb={flip['fastballs_seen']}: "
              f"{flip['baseline_action']} -> {flip['v3_action']} (V={flip['v3_value']:.4f})")

# Check if policy WITHIN v3 varies by fastballs_seen
print(f"\n  ── Policy variation by fastballs_seen ──")
varied_states = []
for b in range(4):
    for s in range(3):
        for d in DEFICITS:
            actions = policy_by_count[(b, s, d)]
            if len(set(actions.values())) > 1:
                varied_states.append({
                    'count': f"{b}-{s}", 'deficit': d,
                    'actions': dict(actions),
                })
                acts_str = ", ".join(f"fb={k}:{v}" for k, v in actions.items())
                print(f"    {b}-{s} d={d}: {acts_str}")

if not varied_states:
    print(f"    None — policy is invariant to fastballs_seen at all counts")

# Value function comparison
print(f"\n  V*(0-0, d=0, r=0) by fastballs_seen:")
for fb in range(4):
    v = V3[(0, 0, 0, 0, fb)]
    print(f"    fb={fb}: {v:.4f}")

# Save extension results
ext_rows = []
for state, action in policy3.items():
    b, s, d, r, fb = state
    ext_rows.append({
        'balls': b, 'strikes': s, 'deficit': d,
        'runs_scored': r, 'fastballs_seen': fb,
        'count': f"{b}-{s}", 'optimal_action': action,
        'value': V3[state],
    })
ext_df = pd.DataFrame(ext_rows)
ext_df.to_csv(f"{EXT_DIR}/history_augmented_mdp.csv", index=False)
print(f"\n  Saved {len(ext_df)} states to data/extensions/history_augmented_mdp.csv")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: STAKES ENRICHMENT WITH PROPER STANDINGS
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 3: Stakes Enrichment with Proper Standings")
print("=" * 70)

# ── Build game results from Retrosheet ───────────────────────────────────
# Parse GAME_ID to get home team, date; use final score to determine winner.
# Then build cumulative W-L records.

# For Statcast era: extract game results directly from the pitch data
print("\n  Building game results from Statcast pitch data...")

game_results = []
for year in range(2015, 2025):
    path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path, low_memory=False,
                     usecols=['game_pk', 'game_date', 'home_team', 'away_team',
                              'home_score', 'fld_score', 'bat_score', 'game_type',
                              'inning', 'inning_topbot'])

    # Actually, our primary data is ONLY situation pitches (bot 9, 2 out, bases loaded).
    # We cannot derive full standings from this.
    # We need ALL game results for the season.
    break

# The primary data is only the situation sample — we cannot build standings from it.
# Alternative approach: use pybaseball to pull season standings or game logs.
print("  Primary data is situation-only — cannot derive full standings.")
print("  Attempting to pull game results via pybaseball...")

try:
    from pybaseball import schedule_and_record
    # Test with one team-year
    test = schedule_and_record(2024, 'NYY')
    print(f"  pybaseball schedule_and_record works! Got {len(test)} games for NYY 2024")
    pybaseball_works = True
except Exception as e:
    print(f"  pybaseball schedule_and_record failed: {e}")
    pybaseball_works = False

if not pybaseball_works:
    # Try FanGraphs standings data
    try:
        from pybaseball import standings
        test_standings = standings(2024)
        print(f"  pybaseball standings works! Got {len(test_standings)} divisions")
        standings_works = True
    except Exception as e:
        print(f"  pybaseball standings failed: {e}")
        standings_works = False
else:
    standings_works = False

# If we can get schedule_and_record, build proper standings
if pybaseball_works:
    print("\n  Building proper standings from pybaseball schedule_and_record...")

    # All 30 MLB teams (current codes)
    MLB_TEAMS = ['ARI','ATL','BAL','BOS','CHC','CHW','CIN','CLE','COL','DET',
                 'HOU','KCR','LAA','LAD','MIA','MIL','MIN','NYM','NYY','OAK',
                 'PHI','PIT','SDP','SEA','SFG','STL','TBR','TEX','TOR','WSN']

    # Division membership (2015-2024, stable period)
    DIVISIONS = {
        'AL_East': ['BAL','BOS','NYY','TBR','TOR'],
        'AL_Central': ['CHW','CLE','DET','KCR','MIN'],
        'AL_West': ['HOU','LAA','OAK','SEA','TEX'],
        'NL_East': ['ATL','MIA','NYM','PHI','WSN'],
        'NL_Central': ['CHC','CIN','MIL','PIT','STL'],
        'NL_West': ['ARI','COL','LAD','SDP','SFG'],
    }

    # Invert: team -> division
    team_to_div = {}
    for div, teams in DIVISIONS.items():
        for t in teams:
            team_to_div[t] = div

    # Wild card spots by year
    WC_SPOTS = {y: 2 if y < 2022 else 3 for y in range(2015, 2025)}

    # Pull schedules for all teams in each year, build cumulative standings
    all_standings = []

    for year in range(2015, 2025):
        print(f"    {year}...", end=" ", flush=True)
        year_games = {}

        for team in MLB_TEAMS:
            try:
                sched = schedule_and_record(year, team)
                if sched is None or len(sched) == 0:
                    continue
                # Parse: Date, W/L, R, RA, Win, Loss, Save
                sched = sched.reset_index()
                for _, game in sched.iterrows():
                    date = pd.to_datetime(game.get('Date', game.get('date', '')), errors='coerce')
                    if pd.isna(date):
                        continue
                    wl = str(game.get('W/L', game.get('w/l', '')))
                    if wl.startswith('W'):
                        won = True
                    elif wl.startswith('L'):
                        won = False
                    else:
                        continue  # postponed, etc.

                    key = (team, date.strftime('%Y-%m-%d'))
                    if key not in year_games:
                        year_games[key] = won
            except Exception as e:
                continue

        # Build cumulative W-L by team-date
        if not year_games:
            print("no data")
            continue

        # Sort by date
        records = sorted([(t, d, w) for (t, d), w in year_games.items()], key=lambda x: x[1])

        team_wl = {t: {'W': 0, 'L': 0} for t in MLB_TEAMS}
        daily_records = {}

        for team, date, won in records:
            if won:
                team_wl[team]['W'] += 1
            else:
                team_wl[team]['L'] += 1
            daily_records[(team, date)] = dict(team_wl[team])

        # Season end: last game date
        season_end = max(d for _, d, _ in records)
        season_dates = sorted(set(d for _, d, _ in records))

        # For each date: compute games behind playoff for each team
        for date in season_dates:
            # Get current W-L for all teams
            current_wl = {}
            for team in MLB_TEAMS:
                # Find most recent record up to this date
                best = None
                for d in season_dates:
                    if d > date:
                        break
                    if (team, d) in daily_records:
                        best = daily_records[(team, d)]
                if best:
                    current_wl[team] = best
                else:
                    current_wl[team] = {'W': 0, 'L': 0}

            # Compute division leaders and wild card positions
            div_leaders = {}
            for div_name, div_teams in DIVISIONS.items():
                league = div_name[:2]  # 'AL' or 'NL'
                best_team = max(div_teams,
                                key=lambda t: current_wl.get(t, {}).get('W', 0) -
                                              current_wl.get(t, {}).get('L', 0))
                div_leaders[div_name] = best_team

            # Wild card: best non-division-leaders in each league
            for league in ['AL', 'NL']:
                league_divs = [d for d in DIVISIONS if d.startswith(league)]
                leaders = {div_leaders[d] for d in league_divs}
                non_leaders = []
                for d in league_divs:
                    for t in DIVISIONS[d]:
                        if t not in leaders:
                            wl = current_wl.get(t, {'W': 0, 'L': 0})
                            non_leaders.append((t, wl['W'] - wl['L']))
                non_leaders.sort(key=lambda x: -x[1])

                n_wc = WC_SPOTS.get(year, 2)
                wc_cutoff = non_leaders[n_wc - 1][1] if len(non_leaders) >= n_wc else -999

                for team in MLB_TEAMS:
                    if team_to_div.get(team, '')[:2] != league:
                        continue

                    wl = current_wl.get(team, {'W': 0, 'L': 0})
                    diff = wl['W'] - wl['L']

                    # Games behind division leader
                    div = team_to_div.get(team, '')
                    leader = div_leaders.get(div, team)
                    leader_wl = current_wl.get(leader, {'W': 0, 'L': 0})
                    gb_div = ((leader_wl['W'] - leader_wl['L']) - diff) / 2

                    # Games behind wild card
                    if team in leaders:
                        gb_wc = 0  # division leader = in playoffs
                    else:
                        gb_wc = (wc_cutoff - diff) / 2

                    gb_playoff = min(gb_div, gb_wc)

                    days_remaining = (pd.to_datetime(season_end) - pd.to_datetime(date)).days

                    all_standings.append({
                        'year': year, 'team': team, 'date': date,
                        'wins': wl['W'], 'losses': wl['L'],
                        'gb_division': max(gb_div, 0),
                        'gb_wildcard': max(gb_wc, 0) if team not in leaders else 0,
                        'gb_playoff': max(gb_playoff, 0),
                        'days_remaining': days_remaining,
                        'is_division_leader': team in leaders,
                    })

        print(f"{len(year_games)} games, {len(season_dates)} dates")

    standings_df = pd.DataFrame(all_standings)
    standings_df.to_csv(f"{STAKES_DIR}/cumulative_standings.csv", index=False)
    print(f"\n  Saved {len(standings_df):,} team-date records to stakes/cumulative_standings.csv")

    # ── Assign stakes tiers to situation PAs ─────────────────────────────
    print(f"\n  Assigning stakes tiers to situation PAs...")

    all_situation_pitches = []
    for year in range(2015, 2025):
        path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, low_memory=False)
        df['deficit'] = df['fld_score'] - df['bat_score']
        df['year'] = year
        df['game_date'] = pd.to_datetime(df['game_date'])
        all_situation_pitches.append(df)

    all_sc = pd.concat(all_situation_pitches, ignore_index=True)

    def assign_stakes_proper(row, standings_df):
        gt = str(row.get('game_type', 'R')).upper()
        if gt in ('D', 'L', 'W', 'F', 'P'):
            return 'postseason'

        team = row.get('home_team', '')
        date_str = row['game_date'].strftime('%Y-%m-%d')
        year = row['year']

        # Look up standings
        match = standings_df[
            (standings_df['team'] == team) &
            (standings_df['date'] == date_str) &
            (standings_df['year'] == year)
        ]

        if len(match) == 0:
            return 'medium'  # fallback

        rec = match.iloc[0]
        days_rem = rec['days_remaining']
        gb = rec['gb_playoff']

        if days_rem <= 14 and gb > 12:
            return 'low'
        elif days_rem <= 28 and gb <= 5:
            return 'high'
        else:
            return 'medium'

    all_sc['stakes_tier'] = all_sc.apply(lambda r: assign_stakes_proper(r, standings_df), axis=1)

    # Report
    tier_dist = all_sc.groupby('stakes_tier').agg(
        n_pitches=('game_pk', 'count'),
        n_games=('game_pk', 'nunique'),
    ).reset_index()
    print(f"\n  Proper standings-based stakes distribution:")
    for _, row in tier_dist.iterrows():
        print(f"    {row['stakes_tier']:12s}: {row['n_pitches']:>5} pitches, {row['n_games']:>4} games")

    # Compare to Session 5 heuristic
    all_sc['month'] = all_sc['game_date'].dt.month
    all_sc['day'] = all_sc['game_date'].dt.day
    all_sc['heuristic_tier'] = all_sc.apply(
        lambda r: 'postseason' if str(r.get('game_type', 'R')).upper() in ('D','L','W','F','P')
        else ('high' if (r['month'] == 9 or (r['month'] == 10 and r['day'] <= 7)) else 'medium'),
        axis=1
    )
    confusion = pd.crosstab(all_sc['heuristic_tier'], all_sc['stakes_tier'],
                            margins=True, margins_name='Total')
    print(f"\n  Heuristic vs. Proper classification confusion matrix:")
    print(confusion.to_string())

    # ── Recompute policy gaps by proper stakes tier ──────────────────────
    print(f"\n  Computing policy gaps by proper stakes tier...")

    # Load baseline V and T from Session 5
    sens_path = f"{MDP_DIR}/sensitivity_analysis_v2.csv"
    sens = pd.read_csv(sens_path)
    baseline_V_dict = {}
    baseline_pol_dict = {}
    for _, row in sens[sens['scenario'] == 'baseline'].iterrows():
        state = (int(row['balls']), int(row['strikes']), int(row['deficit']), 0)
        baseline_V_dict[state] = row['value']
        baseline_pol_dict[state] = row['optimal_action']

    # Rebuild transitions for computing v_observed
    pb = statcast_trans['p_ball_on_take']
    ps = statcast_trans['p_strike_on_take']
    pw = statcast_trans['p_whiff_on_swing']
    pf = statcast_trans['p_foul_on_swing']
    pc = statcast_trans['p_contact_on_swing']

    def build_simple_T():
        T = {}
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
                                take[(0, 0, d-1, min(r+1, 3))] = take.get((0, 0, d-1, min(r+1, 3)), 0) + pb
                        else:
                            take[(b+1, s, d, r)] = pb
                        if s == 2:
                            take[('K', d, r)] = take.get(('K', d, r), 0) + ps
                        else:
                            take[(b, s+1, d, r)] = ps
                        T[state]['take'] = take

                        swing = {}
                        if s == 2:
                            swing[('K', d, r)] = swing.get(('K', d, r), 0) + pw
                        else:
                            swing[(b, s+1, d, r)] = swing.get((b, s+1, d, r), 0) + pw
                        if s < 2:
                            swing[(b, s+1, d, r)] = swing.get((b, s+1, d, r), 0) + pf
                        else:
                            swing[(b, 2, d, r)] = swing.get((b, 2, d, r), 0) + pf
                        swing[('BIP', d, r)] = swing.get(('BIP', d, r), 0) + pc
                        T[state]['swing'] = swing
        return T

    # Need full V for all (b,s,d,r) states — load from Session 4
    v2_df = pd.read_csv(f"{MDP_DIR}/optimal_policies_v2.csv")
    V_full = {}
    for _, row in v2_df[v2_df['era'] == 'statcast'].iterrows():
        state = (int(row['balls']), int(row['strikes']), int(row['deficit']), int(row['runs_scored']))
        V_full[state] = row['value']

    T_simple = build_simple_T()

    def compute_v_obs_simple(state, swing_rate, V, T, bip_dist):
        v = 0.0
        for action, ap in [('swing', swing_rate), ('take', 1.0 - swing_rate)]:
            if ap == 0:
                continue
            for ns, tp in T[state][action].items():
                if ns == ('WIN',):
                    v += ap * tp * 1.0
                elif isinstance(ns, tuple) and ns[0] == 'K':
                    v += ap * tp * 0.0
                elif isinstance(ns, tuple) and ns[0] == 'BIP':
                    _, bd, br = ns
                    bev = 0.0
                    for runs, rp in bip_dist.items():
                        rem = bd - runs
                        if rem < 0: bev += rp * 1.0
                        elif rem == 0: bev += rp * EXTRAS_WP
                        else:
                            if runs > 0 and (0,0,rem,min(br+runs,3)) in V:
                                bev += rp * V[(0,0,rem,min(br+runs,3))]
                    v += ap * tp * bev
                elif ns in V:
                    v += ap * tp * V[ns]
        return v

    stakes_gap_v2 = []
    for tier in ['postseason', 'high', 'medium', 'low']:
        tier_data = all_sc[all_sc['stakes_tier'] == tier]
        if len(tier_data) == 0:
            continue

        gaps = []
        for _, pitch in tier_data.iterrows():
            b = min(int(pitch['balls']), 3)
            s = min(int(pitch['strikes']), 2)
            d = int(pitch['deficit'])
            if d not in DEFICITS:
                continue
            state = (b, s, d, 0)
            if state not in V_full:
                continue
            swung = 1.0 if pitch['description'] in SWING_DESCS else 0.0
            vobs = compute_v_obs_simple(state, swung, V_full, T_simple, baseline_bip)
            gap = vobs - V_full[state]
            gaps.append(gap)

        if gaps:
            mg = np.mean(gaps)
            se = np.std(gaps) / np.sqrt(len(gaps))
            n_pa = tier_data['events'].notna().sum()
            stakes_gap_v2.append({
                'tier': tier, 'mean_gap': mg, 'se': se,
                'ci_lo': mg - 1.96*se, 'ci_hi': mg + 1.96*se,
                'n_pitches': len(gaps), 'n_pa_approx': n_pa,
            })
            below = " (below floor)" if n_pa < 40 else ""
            print(f"  {tier:12s}: gap={mg:+.4f} [{mg-1.96*se:+.4f}, {mg+1.96*se:+.4f}], "
                  f"n_pitches={len(gaps)}, n_pa~{n_pa}{below}")

    stakes_v2_df = pd.DataFrame(stakes_gap_v2)
    stakes_v2_df.to_csv(f"{STAKES_DIR}/stakes_policy_gaps_v2.csv", index=False)

    # ── Test: does high > medium gap? ────────────────────────────────────
    if len(stakes_gap_v2) >= 2:
        h = next((s for s in stakes_gap_v2 if s['tier'] == 'high'), None)
        m = next((s for s in stakes_gap_v2 if s['tier'] == 'medium'), None)
        if h and m:
            diff = h['mean_gap'] - m['mean_gap']
            print(f"\n  High - Medium gap difference: {diff:+.4f}")
            print(f"  Direction: {'Pressure amplifies convention' if diff < 0 else 'Pressure sharpens decisions'}")

else:
    print("\n  Cannot build proper standings — pybaseball functions unavailable.")
    print("  Falling back to improved date-based heuristic from Session 5.")
    print("  (Session 5 stakes results remain as primary.)")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: 2-1 COUNT CASE STUDY
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" FIGURE 7: 2-1 Count Case Study")
print("=" * 70)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

# ── Left panel: 2-1 swing rate by year (Statcast + Retrosheet) ──────────
print("  Building Figure 7...")

# Statcast years
yearly_21 = []
for year in range(2015, 2025):
    path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path, low_memory=False)
    df['deficit'] = df['fld_score'] - df['bat_score']
    sub = df[(df['balls'] == 2) & (df['strikes'] == 1) & (df['deficit'] == 1)]
    if len(sub) >= 3:
        sr = sub['description'].isin(SWING_DESCS).mean()
        yearly_21.append({'year': year, 'swing_rate': sr, 'n': len(sub), 'source': 'statcast'})

# Retrosheet years (from pitch sequences)
BALL_CHARS_R = set('BIPV')
CALLED_STRIKE_CHARS_R = set('CK')
SWINGING_STRIKE_CHARS_R = set('SMQT')
FOUL_CHARS_R = set('FHLOR')
CONTACT_CHARS_R = set('XY')
IGNORE_CHARS_R = set('>+*123.N')

for year_file in sorted(os.listdir(f"{BASE_DIR}/data/retrosheet/")):
    if not year_file.startswith('events_') or not year_file.endswith('.csv'):
        continue
    year = int(year_file.replace('events_', '').replace('.csv', ''))
    df = pd.read_csv(f"{BASE_DIR}/data/retrosheet/{year_file}", low_memory=False)

    situation = df[
        (df['INN_CT'] == 9) & (df['BAT_HOME_ID'] == 1) & (df['OUTS_CT'] == 2) &
        (df['BASE1_RUN_ID'].notna()) & (df['BASE1_RUN_ID'] != '') &
        (df['BASE2_RUN_ID'].notna()) & (df['BASE2_RUN_ID'] != '') &
        (df['BASE3_RUN_ID'].notna()) & (df['BASE3_RUN_ID'] != '')
    ]
    if 'BAT_EVENT_FL' in situation.columns:
        situation = situation[situation['BAT_EVENT_FL'] == 'T']

    situation = situation.copy()
    situation['deficit'] = situation['AWAY_SCORE_CT'] - situation['HOME_SCORE_CT']

    swings_21 = 0
    total_21 = 0
    for _, pa in situation[situation['deficit'] == 1].iterrows():
        seq = str(pa.get('PITCH_SEQ_TX', ''))
        if seq == 'nan' or seq == '':
            continue
        balls, strikes = 0, 0
        for c in seq:
            if c in IGNORE_CHARS_R:
                continue
            if balls == 2 and strikes == 1:
                total_21 += 1
                if c in SWINGING_STRIKE_CHARS_R or c in FOUL_CHARS_R or c in CONTACT_CHARS_R:
                    swings_21 += 1
                break
            if c in BALL_CHARS_R:
                balls += 1
            elif c in CALLED_STRIKE_CHARS_R:
                strikes += 1
            elif c in SWINGING_STRIKE_CHARS_R:
                strikes += 1
                if strikes >= 3: break
            elif c in FOUL_CHARS_R:
                if strikes < 2: strikes += 1
            elif c in CONTACT_CHARS_R:
                break
            if balls >= 4: break

    if total_21 >= 2:
        sr = swings_21 / total_21
        yearly_21.append({'year': year, 'swing_rate': sr, 'n': total_21, 'source': 'retrosheet'})

yr21_df = pd.DataFrame(yearly_21).sort_values('year')

# ── Right panel: EV breakdown at 2-1, d=1 ───────────────────────────────
# V*(take) vs V*(swing) vs V(observed)
state_21_d1 = (2, 1, 1, 0)
V_take_21 = 0.0
V_swing_21 = 0.0
for ns, tp in T_simple[state_21_d1]['take'].items():
    if ns == ('WIN',):
        V_take_21 += tp * 1.0
    elif isinstance(ns, tuple) and ns[0] == 'K':
        pass
    elif ns in V_full:
        V_take_21 += tp * V_full[ns]

for ns, tp in T_simple[state_21_d1]['swing'].items():
    if ns == ('WIN',):
        V_swing_21 += tp * 1.0
    elif isinstance(ns, tuple) and ns[0] == 'K':
        pass
    elif isinstance(ns, tuple) and ns[0] == 'BIP':
        _, bd, br = ns
        bev = 0.0
        for runs, rp in baseline_bip.items():
            rem = bd - runs
            if rem < 0: bev += rp * 1.0
            elif rem == 0: bev += rp * EXTRAS_WP
            else:
                if runs > 0 and (0,0,rem,min(br+runs,3)) in V_full:
                    bev += rp * V_full[(0,0,rem,min(br+runs,3))]
        V_swing_21 += tp * bev
    elif ns in V_full:
        V_swing_21 += tp * V_full[ns]

# Observed: 73.7% swing
obs_sr = 0.737
V_obs_21 = obs_sr * V_swing_21 + (1 - obs_sr) * V_take_21
V_opt_21 = max(V_take_21, V_swing_21)

print(f"  2-1, d=1: V(take)={V_take_21:.4f}, V(swing)={V_swing_21:.4f}, "
      f"V(obs@73.7%)={V_obs_21:.4f}, V*={V_opt_21:.4f}")
print(f"  Gap: {V_obs_21 - V_opt_21:+.4f}")

# ── Draw figure ──────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={'width_ratios': [1.5, 1]})

# Left: swing rate by year
retro = yr21_df[yr21_df['source'] == 'retrosheet']
sc = yr21_df[yr21_df['source'] == 'statcast']

if len(retro) > 0:
    ax1.bar(retro['year'], retro['swing_rate'], width=3.5, color=CORAL, alpha=0.5,
            label='Retrosheet', edgecolor=CORAL)
if len(sc) > 0:
    ax1.bar(sc['year'], sc['swing_rate'], width=0.7, color=CORAL, alpha=0.85,
            label='Statcast', edgecolor=CORAL)

# Optimal line
ax1.axhline(0, color=TEAL, linewidth=2, linestyle='-', label='Optimal swing rate')
ax1.fill_between([1988, 2026], -0.02, 0.02, color=TEAL, alpha=0.1)

ax1.set_xlabel('Year')
ax1.set_ylabel('Swing Rate at 2-1 Count')
ax1.set_title('Swing Rate at 2-1 (Down 1 Run)\nBot 9, 2 Out, Bases Loaded')
ax1.set_xlim(1988, 2026)
ax1.set_ylim(-0.05, 1.0)
ax1.legend(loc='upper left', fontsize=9)

# Annotate
ax1.text(2020, 0.85, 'Observed: ~74%\nswing rate',
         fontsize=10, color=CORAL, ha='center', fontweight='bold')
ax1.text(2020, 0.08, 'Optimal: TAKE\n(0% swing)',
         fontsize=10, color=TEAL, ha='center', fontweight='bold')

# Right: EV comparison
labels = ['Take\n(Optimal)', 'Observed\n(73.7% swing)', 'Swing\n(100%)']
values = [V_take_21, V_obs_21, V_swing_21]
colors = [TEAL, AMBER, CORAL]

bars = ax2.barh(range(len(labels)), values, color=colors, alpha=0.85, height=0.6)
ax2.set_yticks(range(len(labels)))
ax2.set_yticklabels(labels)
ax2.set_xlabel('Expected Win Probability')
ax2.set_title('Expected Value at 2-1, Down 1\nWith Bases Loaded')

# Annotate bars
for i, (v, label) in enumerate(zip(values, labels)):
    ax2.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=11, fontweight='bold')

# Annotate the gap
gap = V_obs_21 - V_opt_21
ax2.annotate(f'Gap: {gap:+.3f} WP',
             xy=(V_obs_21, 1), xytext=(V_obs_21 + 0.06, 1.6),
             fontsize=11, color=SLATE, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=SLATE, lw=1.5))

fig.suptitle('The 2-1 Count: Where Convention Costs the Most',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig7_case_study.pdf", bbox_inches='tight')
fig.savefig(f"{FIG_DIR}/fig7_case_study.png", bbox_inches='tight')
plt.close(fig)
print(f"  Saved fig7_case_study.pdf/png")


# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" SESSION 6 ANALYSIS COMPLETE")
print("=" * 70)

for d in [EXT_DIR, STAKES_DIR, FIG_DIR]:
    if os.path.exists(d):
        print(f"\n  {os.path.relpath(d, BASE_DIR)}/")
        for f in sorted(os.listdir(d)):
            fp = os.path.join(d, f)
            sz = os.path.getsize(fp)
            unit = 'KB' if sz > 1024 else 'B'
            val = sz / 1024 if sz > 1024 else sz
            print(f"    {f} ({val:.0f} {unit})")
