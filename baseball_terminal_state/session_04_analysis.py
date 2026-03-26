"""
Session 4: Steps 4-7 — Historical Analysis with Corrected Retrosheet Data
Uses augmented MDP from session_04_longitudinal.py Step 1.
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
os.makedirs(LONG_DIR, exist_ok=True)

ERAS = {
    'post_war': (1950, 1968),
    'expansion': (1969, 1992),
    'offense_explosion': (1993, 2005),
    'post_steroid': (2006, 2014),
    'statcast': (2015, 2024),
}

def assign_era(year):
    for name, (s, e) in ERAS.items():
        if s <= year <= e:
            return name
    return 'unknown'

DEFICITS = [0, 1, 2, 3]
EXTRAS_WP = 0.52
MIN_PA_FLOOR = 40

BALL_CHARS = set('BIPV')
CALLED_STRIKE_CHARS = set('CK')
SWINGING_STRIKE_CHARS = set('SMQT')
FOUL_CHARS = set('FHLOR')
CONTACT_CHARS = set('XY')
IGNORE_CHARS = set('>+*123.N')

SWING_DESCS = {'swinging_strike', 'foul', 'foul_tip', 'swinging_strike_blocked',
               'foul_bunt', 'missed_bunt', 'hit_into_play', 'hit_into_play_no_out',
               'hit_into_play_score', 'foul_pitchout', 'swinging_pitchout'}

# ── Load augmented MDP results ──────────────────────────────────────────
print("Loading augmented MDP policies...")
policy_v2 = pd.read_csv(f"{MDP_DIR}/optimal_policies_v2.csv")

# Reconstruct solutions dict
augmented_V = {}
augmented_pol = {}
for _, row in policy_v2.iterrows():
    era = row['era']
    state = (int(row['balls']), int(row['strikes']), int(row['deficit']), int(row['runs_scored']))
    if era not in augmented_V:
        augmented_V[era] = {}
        augmented_pol[era] = {}
    augmented_V[era][state] = row['value']
    augmented_pol[era][state] = row['optimal_action']

# Load transition parameters
trans_df = pd.read_csv(f"{MDP_DIR}/transition_parameters.csv", index_col=0)
era_transitions = trans_df.to_dict('index')

# Load BIP run distribution from primary data
bip_runs_dist = {0: 0.669, 1: 0.122, 2: 0.143, 3: 0.021, 4: 0.044}  # from Step 1

# Build transition dicts
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


def compute_v_observed(state, swing_rate, V, T, bip_dist):
    """Compute expected value under observed behavior."""
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
# STEP 4: Historical Policy Gaps
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 4: Historical Policy Gaps (Retrosheet + Statcast)")
print("=" * 70)

def decode_actions(seq):
    """Decode pitch sequence to list of (balls, strikes, action) tuples."""
    if not isinstance(seq, str) or seq == 'nan':
        return []
    pitches = []
    balls, strikes = 0, 0
    for c in seq:
        if c in IGNORE_CHARS: continue
        if c in BALL_CHARS:
            pitches.append((balls, strikes, 'take'))
            balls += 1
            if balls >= 4: break
        elif c in CALLED_STRIKE_CHARS:
            pitches.append((balls, strikes, 'take'))
            strikes += 1
            if strikes >= 3: break
        elif c in SWINGING_STRIKE_CHARS:
            pitches.append((balls, strikes, 'swing'))
            strikes += 1
            if strikes >= 3: break
        elif c in FOUL_CHARS:
            pitches.append((balls, strikes, 'swing'))
            if strikes < 2: strikes += 1
        elif c in CONTACT_CHARS:
            pitches.append((balls, strikes, 'swing'))
            break
    return pitches


def compute_inning_run_context(df):
    """Compute runs_scored_this_inning for each event."""
    df = df.sort_values(['GAME_ID', 'INN_CT', 'BAT_HOME_ID']).copy()
    runs_list = []
    for _, group in df.groupby(['GAME_ID', 'INN_CT', 'BAT_HOME_ID']):
        group = group.sort_index()
        is_home = group['BAT_HOME_ID'].iloc[0] == 1
        scores = group['HOME_SCORE_CT'].values if is_home else group['AWAY_SCORE_CT'].values
        start_score = scores[0]
        runs_list.extend((scores - start_score).tolist())
    df['runs_scored_this_inning'] = runs_list
    return df


# Process all available Retrosheet years
retro_years = []
for f in sorted(os.listdir(f"{BASE_DIR}/data/retrosheet/")):
    if f.startswith('events_') and f.endswith('.csv'):
        year = int(f.replace('events_', '').replace('.csv', ''))
        retro_years.append(year)

print(f"  Available Retrosheet years: {retro_years}")

longitudinal_gaps = []

for year in retro_years:
    era = assign_era(year)
    if era == 'statcast':
        continue  # Handle Statcast separately

    # Get nearest era for MDP solution
    era_for_mdp = era
    if era_for_mdp not in augmented_V:
        # Use nearest available
        available = list(augmented_V.keys())
        era_order = ['post_war', 'expansion', 'offense_explosion', 'post_steroid', 'statcast']
        if era_for_mdp in era_order:
            idx = era_order.index(era_for_mdp)
            for offset in range(1, len(era_order)):
                for candidate in [idx - offset, idx + offset]:
                    if 0 <= candidate < len(era_order) and era_order[candidate] in available:
                        era_for_mdp = era_order[candidate]
                        break
                if era_for_mdp in available:
                    break

    V = augmented_V.get(era_for_mdp, {})
    if not V:
        continue

    # Build transitions for this era
    tparams = era_transitions.get(era_for_mdp, era_transitions.get('statcast', {}))
    T = build_transition_dict(tparams)

    # Load and process events
    df = pd.read_csv(f"{BASE_DIR}/data/retrosheet/events_{year}.csv", low_memory=False)
    df = compute_inning_run_context(df)

    # Filter to primary situation
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

    n_pa = len(situation)
    below_floor = n_pa < MIN_PA_FLOOR

    # Compute policy gaps from pitch sequences
    all_gaps = []
    for _, pa in situation.iterrows():
        seq = str(pa.get('PITCH_SEQ_TX', ''))
        if seq == 'nan' or seq == '':
            continue

        deficit = int(pa['deficit'])
        if deficit not in DEFICITS:
            continue

        r = int(pa.get('runs_scored_this_inning', 0))
        r = min(r, 3)

        pitches = decode_actions(seq)
        for balls, strikes, action in pitches:
            state = (balls, strikes, deficit, r)
            if state not in V:
                continue

            sr = 1.0 if action == 'swing' else 0.0
            v_obs = compute_v_observed(state, sr, V, T, bip_runs_dist)
            gap = v_obs - V[state]
            all_gaps.append(gap)

    if all_gaps:
        mean_gap = np.mean(all_gaps)
        longitudinal_gaps.append({
            'year': year, 'era': era,
            'weighted_gap': mean_gap, 'n_pitches': len(all_gaps),
            'n_pa': n_pa, 'below_floor': below_floor,
            'source': 'retrosheet',
        })
        flag = " ⚠<floor" if below_floor else ""
        print(f"  {year} ({era}): gap={mean_gap:+.4f}, n_pitches={len(all_gaps)}, n_pa={n_pa}{flag}")

# Add Statcast era
yearly_v2_path = f"{MDP_DIR}/yearly_policy_gaps_v2.csv"
if os.path.exists(yearly_v2_path):
    yearly_v2 = pd.read_csv(yearly_v2_path)
    for _, row in yearly_v2.iterrows():
        longitudinal_gaps.append({
            'year': int(row['year']), 'era': 'statcast',
            'weighted_gap': row['weighted_gap'], 'n_pitches': int(row['n']),
            'n_pa': int(row['n']) // 4,  # approximate
            'below_floor': False,
            'source': 'statcast',
        })

long_df = pd.DataFrame(longitudinal_gaps).sort_values('year')
long_df.to_csv(f"{LONG_DIR}/policy_gaps_historical.csv", index=False)

print(f"\n  ── Full Longitudinal Series ──")
for _, row in long_df.iterrows():
    flag = " ⚠" if row['below_floor'] else ""
    print(f"  {int(row['year'])} ({row['era']:20s}): gap={row['weighted_gap']:+.4f}, n={int(row['n_pitches']):>4}{flag}")

# Era-level aggregates
print(f"\n  ── Era-Level Summary ──")
for era_name in ['post_war', 'expansion', 'offense_explosion', 'post_steroid', 'statcast']:
    era_data = long_df[(long_df['era'] == era_name) & (~long_df['below_floor'])]
    if len(era_data) == 0:
        era_data = long_df[long_df['era'] == era_name]  # include below-floor if no other data
    if len(era_data) > 0:
        mean_gap = era_data['weighted_gap'].mean()
        n_years = len(era_data)
        total_pitches = era_data['n_pitches'].sum()
        print(f"  {era_name:20s}: mean gap={mean_gap:+.4f}, {n_years} years, {total_pitches:,} pitches")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: 3-0 Paradox Time Series
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 5: 3-0 Paradox Time Series")
print("=" * 70)

all_three_zero = []

# Retrosheet: decode 3-0 counts from pitch sequences
for year in retro_years:
    era = assign_era(year)
    if era == 'statcast':
        continue

    df = pd.read_csv(f"{BASE_DIR}/data/retrosheet/events_{year}.csv", low_memory=False)
    df = compute_inning_run_context(df)

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

    for _, pa in situation.iterrows():
        seq = str(pa.get('PITCH_SEQ_TX', ''))
        if seq == 'nan' or seq == '':
            continue

        balls, strikes = 0, 0
        for c in seq:
            if c in IGNORE_CHARS: continue

            if balls == 3 and strikes == 0:
                # We're at 3-0
                swung = c in SWINGING_STRIKE_CHARS or c in FOUL_CHARS or c in CONTACT_CHARS
                deficit = int(pa['deficit'])
                all_three_zero.append({
                    'year': year, 'deficit': deficit,
                    'swung': 1 if swung else 0, 'era': era,
                })
                break

            if c in BALL_CHARS: balls += 1
            elif c in CALLED_STRIKE_CHARS: strikes += 1
            elif c in SWINGING_STRIKE_CHARS:
                strikes += 1
                if strikes >= 3: break
            elif c in FOUL_CHARS:
                if strikes < 2: strikes += 1
            elif c in CONTACT_CHARS: break

            if balls >= 4: break

# Statcast: 3-0 counts from pitch-level data
for year in range(2015, 2025):
    path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path, low_memory=False)
    df['deficit'] = df['fld_score'] - df['bat_score']
    three_zero = df[(df['balls'] == 3) & (df['strikes'] == 0)]
    if len(three_zero) > 0:
        for _, row in three_zero.iterrows():
            swung = row['description'] in SWING_DESCS
            all_three_zero.append({
                'year': year, 'deficit': int(row['deficit']),
                'swung': 1 if swung else 0, 'era': 'statcast',
            })

three_zero_df = pd.DataFrame(all_three_zero)
if len(three_zero_df) > 0:
    # Aggregate by year × deficit
    ts_records = []
    for year in sorted(three_zero_df['year'].unique()):
        yr = three_zero_df[three_zero_df['year'] == year]

        # Overall
        ts_records.append({
            'year': year, 'deficit': 'all',
            'swing_rate': yr['swung'].mean(), 'n': len(yr),
            'era': assign_era(year),
        })
        # By deficit
        for d in DEFICITS:
            sub = yr[yr['deficit'] == d]
            if len(sub) >= 2:
                ts_records.append({
                    'year': year, 'deficit': d,
                    'swing_rate': sub['swung'].mean(), 'n': len(sub),
                    'era': assign_era(year),
                })

    ts_df = pd.DataFrame(ts_records)
    ts_df.to_csv(f"{LONG_DIR}/three_zero_time_series.csv", index=False)

    print(f"\n  3-0 swing rate in terminal situation by year:")
    overall_ts = ts_df[ts_df['deficit'] == 'all'].sort_values('year')
    for _, row in overall_ts.iterrows():
        print(f"  {int(row['year'])} ({row['era']:20s}): swing_rate={row['swing_rate']:.3f} (n={int(row['n'])})")

    # Trend test on 3-0 swing rate (overall)
    if len(overall_ts) >= 5:
        slope, intercept, r, p, se = stats.linregress(overall_ts['year'], overall_ts['swing_rate'])
        print(f"\n  Linear trend: slope={slope:+.5f}/year, r²={r**2:.3f}, p={p:.4f}")

    # 3-0 swing rate by deficit (pooled across years)
    print(f"\n  3-0 swing rate by deficit (pooled):")
    for d in DEFICITS:
        sub = three_zero_df[three_zero_df['deficit'] == d]
        if len(sub) >= 5:
            print(f"    deficit={d}: swing_rate={sub['swung'].mean():.3f} (n={len(sub)})")
else:
    print("  No 3-0 count data found")
    ts_df = pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: CUSUM Changepoint Detection
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 6: CUSUM Changepoint Detection")
print("=" * 70)


def cusum_detection(values, years, target_mean=None, k=0.005, h=0.03):
    if target_mean is None:
        pre = [v for y, v in zip(years, values) if y < 2015]
        target_mean = np.mean(pre) if pre else np.mean(values[:3])

    C_plus, C_minus = [0.0], [0.0]
    sig_up, sig_down = [], []

    for i, (y, x) in enumerate(zip(years, values)):
        cp = max(0, C_plus[-1] + (x - target_mean) - k)
        cm = max(0, C_minus[-1] - (x - target_mean) - k)
        C_plus.append(cp)
        C_minus.append(cm)
        if cp > h: sig_up.append((y, i))
        if cm > h: sig_down.append((y, i))

    return C_plus[1:], C_minus[1:], sig_up, sig_down, target_mean


def ewma_chart(values, years, lambda_=0.2, L=2.5):
    if len(values) < 3:
        return [], [], [], []
    mu_0 = np.mean(values[:max(3, len(values)//3)])
    sigma = np.std(values[:max(3, len(values)//3)])
    if sigma < 1e-10: sigma = 0.01

    z = [values[0]]
    ucl, lcl, signals = [], [], []
    for i in range(1, len(values)):
        zn = lambda_ * values[i] + (1 - lambda_) * z[-1]
        z.append(zn)
        cl = L * sigma * np.sqrt(lambda_ / (2 - lambda_) * (1 - (1 - lambda_)**(2*(i+1))))
        ucl.append(mu_0 + cl)
        lcl.append(mu_0 - cl)
        if zn > mu_0 + cl or zn < mu_0 - cl:
            signals.append((years[i], i))

    return z, ucl, lcl, signals


if len(long_df) >= 5:
    # Use only above-floor years for changepoint analysis
    valid = long_df[~long_df['below_floor']].sort_values('year')
    years = valid['year'].values
    gaps = valid['weighted_gap'].values

    print(f"\n  Analyzing {len(years)} data points ({years[0]}-{years[-1]})")

    # CUSUM
    C_plus, C_minus, sig_up, sig_down, target = cusum_detection(gaps, years)
    print(f"\n  CUSUM (target={target:.4f}, k=0.005, h=0.03):")
    if sig_up: print(f"    Upward shifts: {[y for y, _ in sig_up]}")
    if sig_down: print(f"    Downward shifts: {[y for y, _ in sig_down]}")
    if not sig_up and not sig_down: print(f"    No changepoints detected")

    # EWMA
    z, ucl, lcl, ewma_sig = ewma_chart(gaps, years)
    print(f"\n  EWMA (λ=0.2, L=2.5):")
    if ewma_sig: print(f"    Signals: {[y for y, _ in ewma_sig]}")
    else: print(f"    No signals detected")

    # Agreement
    cusum_set = set(y for y, _ in sig_up + sig_down)
    ewma_set = set(y for y, _ in ewma_sig)
    if cusum_set & ewma_set:
        print(f"\n  ✓ Methods agree on changepoints: {sorted(cusum_set & ewma_set)}")
    elif not cusum_set and not ewma_set:
        print(f"\n  ✓ Both methods agree: no structural breaks")
    else:
        print(f"\n  Methods disagree — CUSUM: {sorted(cusum_set) or 'none'}, EWMA: {sorted(ewma_set) or 'none'}")

    # Overall trend
    slope, intercept, r, p, se = stats.linregress(years, gaps)
    print(f"\n  Linear trend: slope={slope:+.6f}/year, r²={r**2:.3f}, p={p:.4f}")
    if p < 0.05:
        direction = "closing" if slope > 0 else "widening"
        print(f"  → Significant: policy gap is {direction}")
    else:
        print(f"  → No significant trend")

    # Save
    cp_df = pd.DataFrame({'year': years, 'gap': gaps, 'cusum_plus': C_plus, 'cusum_minus': C_minus})
    cp_df.to_csv(f"{LONG_DIR}/changepoint_analysis.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Pitcher Accountability (Statcast era)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 7: Pitcher Accountability Analysis")
print("=" * 70)

pit_dev_path = f"{BASE_DIR}/data/deviations/statcast_pitcher_deviations.csv"
if os.path.exists(pit_dev_path):
    pit_devs = pd.read_csv(pit_dev_path)
    print(f"  {len(pit_devs)} pitcher deviation records")

    for metric in ['dev_zone_rate', 'dev_hard_pct', 'dev_offspeed_pct', 'dev_fps_rate',
                   'dev_chase_rate_induced']:
        if metric not in pit_devs.columns:
            continue
        vals = pit_devs[metric].dropna()
        if len(vals) < 5:
            continue
        t, p = stats.ttest_1samp(vals, 0)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {metric:30s}: mean={vals.mean():+.4f}, t={t:>6.2f}, p={p:.4f} {sig}")
else:
    print("  No pitcher deviation data")


# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" SESSION 4 ANALYSIS COMPLETE")
print("=" * 70)

for d in [MDP_DIR, LONG_DIR]:
    for f in sorted(os.listdir(d)):
        fp = os.path.join(d, f)
        print(f"  {os.path.relpath(fp, BASE_DIR)} ({os.path.getsize(fp):,} bytes)")
