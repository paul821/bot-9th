# bot-9th

## Terminal Game State Strategy in Baseball

This repository contains the analysis code, data, and figures for a research project on pitcher and batter decision-making in terminal game states. The primary situation of interest is the bottom of the 9th inning, two outs, bases loaded, where every pitch carries elimination-level consequences for both sides.

The core question is whether pitchers and batters deviate from optimal strategy under terminal pressure, and if so, how those deviations have changed across baseball eras. To answer this, the project constructs a Markov Decision Process (MDP) that computes count-by-count optimal pitch policies, then measures how real-world behavior departs from those policies using both Statcast pitch-level data (2015-2024) and Retrosheet event files going back to 1955.

## What the Code Does

The analysis pipeline is organized across six sessions, each building on the outputs of the previous one.

**Session 01** handles data orientation. It pulls 2023 Statcast data, validates sample sizes for the primary situation filter, and confirms that the pitch-level granularity is sufficient for count-state analysis.

**Session 02** constructs pitcher and batter baselines from the full Statcast era (2015-2024). For each player, it computes zone rates, chase rates, whiff rates, and swing tendencies in non-terminal situations. These baselines serve as the counterfactual against which terminal-state behavior is compared. Deviation metrics (terminal minus baseline) isolate the behavioral shift attributable to game pressure rather than player skill.

**Session 03** builds the MDP infrastructure. The state space is defined by the ball-strike count and the batting team's run deficit (0 through 3). Transition probabilities are estimated from observed pitch outcomes, and the terminal reward structure reflects walk-off win expectancy. The model solves for the optimal pitch policy at every count-deficit combination, then computes the "policy gap" between what the MDP recommends and what pitchers actually do. A corrected deficit direction (fielding score minus batting score) ensures the model properly captures the asymmetry between tied games and trailing situations.

**Session 04** extends the analysis historically. It pulls Retrosheet event data from five-year intervals spanning 1955 to 2010, decodes pitch sequences into count states, and computes era-level policy gaps. This session also runs a CUSUM changepoint detection on the 3-0 count swing rate time series, identifying structural breaks in batter aggressiveness, and adds a pitcher accountability analysis that traces zone-rate deviations at the individual level.

**Session 05** produces the sensitivity analysis, stakes enrichment, and the full visualization suite. It stress-tests the MDP under varying assumptions about ball-in-play outcomes, stratifies policy gaps by game stakes (division race proximity, playoff implications), and generates the seven figures that summarize the project's findings.

**Session 06** implements model extensions: a 768-state history-augmented MDP that conditions optimal policy on the prior pitch sequence, and a corrected stakes enrichment using actual standings data.

## Data

The `data/` directory contains all intermediate and final outputs organized by analysis stage:

- `statcast/` — Yearly pitch-level data for the primary situation (2015-2024)
- `retrosheet/` — Processed event files with count-state information (1955-2010)
- `baselines/` — Player-level baseline metrics for pitchers and batters
- `deviations/` — Terminal-minus-baseline deviation metrics
- `mdp/` — Transition parameters, optimal policies, value functions, and policy gap measurements
- `longitudinal/` — Time series of policy gaps, 3-0 swing rates, and changepoint results
- `stakes/` — Policy gaps stratified by game importance
- `extensions/` — History-augmented MDP outputs

The `retrosheet_data/` directory contains the raw Retrosheet event files (.EVN, .EVA) and roster files (.ROS) used in Session 04.

## Figures

Seven figures are produced in `figures/`, covering the longitudinal policy gap trend, the optimal policy grid across count-deficit states, per-count decomposition of deviations, stakes stratification, pitcher zone-rate analysis, sensitivity heatmaps, and a case study visualization.

## Requirements

The pipeline depends on `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, and `pybaseball`. Retrosheet parsing requires `chadwick` command-line tools for the historical event file processing in Session 04.
