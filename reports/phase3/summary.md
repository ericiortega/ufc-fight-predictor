# Phase 3: walk-forward evaluation (2013-2026)

Predictions and research findings only; not betting advice.

Expanding window: each test year is predicted by models trained only on earlier years. Optuna tuning and calibration use the last two years of each training window. Probabilities are symmetric per fight.

## Pooled results

| model               | calibration   |    n | accuracy   | log_loss   | brier   |
|:--------------------|:--------------|-----:|:-----------|:-----------|:--------|
| Elo                 | none          | 6684 | 55.9%      | 0.6818     | 0.2444  |
| Higher win rate     | none          | 6684 | 57.2%      | -          | -       |
| LightGBM (5-seed)   | isotonic      | 6684 | 60.8%      | 0.6709     | 0.2328  |
| LightGBM (5-seed)   | none          | 6684 | 61.1%      | 0.6550     | 0.2315  |
| LightGBM (5-seed)   | platt         | 6684 | 61.1%      | 0.6552     | 0.2316  |
| LightGBM (5-seed)   | temperature   | 6684 | 61.1%      | 0.6552     | 0.2316  |
| Logistic regression | isotonic      | 6684 | 61.4%      | 0.6707     | 0.2319  |
| Logistic regression | none          | 6684 | 61.9%      | 0.6527     | 0.2305  |
| Logistic regression | platt         | 6684 | 61.9%      | 0.6532     | 0.2307  |
| Logistic regression | temperature   | 6684 | 61.9%      | 0.6532     | 0.2307  |
| More UFC fights     | none          | 6684 | 49.4%      | -          | -       |
| Younger fighter     | none          | 6684 | 57.8%      | -          | -       |

Pick-only baselines have no probabilities, so only accuracy is reported. Platt scaling without an intercept (required for symmetry) is mathematically the same as temperature scaling.

## By test year (uncalibrated)

|   test_year |   Elo accuracy |   LightGBM (5-seed) accuracy |   Logistic regression accuracy |   Elo log loss |   LightGBM (5-seed) log loss |   Logistic regression log loss |
|------------:|---------------:|-----------------------------:|-------------------------------:|---------------:|-----------------------------:|-------------------------------:|
|        2013 |         0.5479 |                       0.6383 |                         0.5957 |         0.6876 |                       0.6522 |                         0.6662 |
|        2014 |         0.582  |                       0.5709 |                         0.5709 |         0.6696 |                       0.6728 |                         0.6758 |
|        2015 |         0.5603 |                       0.6013 |                         0.6078 |         0.6814 |                       0.6641 |                         0.6622 |
|        2016 |         0.5787 |                       0.6149 |                         0.6294 |         0.6726 |                       0.6607 |                         0.6504 |
|        2017 |         0.5695 |                       0.6143 |                         0.6368 |         0.6837 |                       0.6523 |                         0.6503 |
|        2018 |         0.5416 |                       0.6141 |                         0.6375 |         0.6831 |                       0.66   |                         0.6458 |
|        2019 |         0.503  |                       0.5632 |                         0.5711 |         0.7052 |                       0.6692 |                         0.6755 |
|        2020 |         0.5788 |                       0.6351 |                         0.6532 |         0.6766 |                       0.6502 |                         0.6493 |
|        2021 |         0.5684 |                       0.5875 |                         0.5976 |         0.6828 |                       0.6604 |                         0.6551 |
|        2022 |         0.5613 |                       0.6047 |                         0.6126 |         0.6768 |                       0.6546 |                         0.6506 |
|        2023 |         0.5536 |                       0.631  |                         0.6052 |         0.6843 |                       0.6495 |                         0.653  |
|        2024 |         0.5635 |                       0.6094 |                         0.6484 |         0.6795 |                       0.6471 |                         0.632  |
|        2025 |         0.5643 |                       0.6491 |                         0.6725 |         0.678  |                       0.6376 |                         0.627  |
|        2026 |         0.5553 |                       0.6234 |                         0.6277 |         0.6848 |                       0.6389 |                         0.6484 |

## Subsets

| subset                  | model               |    n | accuracy   |   log_loss |   brier |
|:------------------------|:--------------------|-----:|:-----------|-----------:|--------:|
| all                     | Elo                 | 6684 | 55.9%      |     0.6818 |  0.2444 |
| all                     | LightGBM (5-seed)   | 6684 | 61.1%      |     0.655  |  0.2315 |
| all                     | Logistic regression | 6684 | 61.9%      |     0.6527 |  0.2305 |
| both_3plus_prior_fights | Elo                 | 3306 | 55.7%      |     0.682  |  0.2445 |
| both_3plus_prior_fights | LightGBM (5-seed)   | 3306 | 62.1%      |     0.6493 |  0.2288 |
| both_3plus_prior_fights | Logistic regression | 3306 | 63.0%      |     0.648  |  0.2283 |
| any_0to2_prior_fights   | Elo                 | 3378 | 56.1%      |     0.6816 |  0.2442 |
| any_0to2_prior_fights   | LightGBM (5-seed)   | 3378 | 60.1%      |     0.6606 |  0.2342 |
| any_0to2_prior_fights   | Logistic regression | 3378 | 60.9%      |     0.6573 |  0.2327 |
| profiles_complete       | Elo                 | 6294 | 55.7%      |     0.682  |  0.2445 |
| profiles_complete       | LightGBM (5-seed)   | 6294 | 61.1%      |     0.6548 |  0.2314 |
| profiles_complete       | Logistic regression | 6294 | 62.0%      |     0.6518 |  0.2301 |
| any_profile_imputed     | Elo                 |  390 | 59.1%      |     0.6793 |  0.2427 |
| any_profile_imputed     | LightGBM (5-seed)   |  390 | 60.3%      |     0.6578 |  0.2328 |
| any_profile_imputed     | Logistic regression |  390 | 60.5%      |     0.6673 |  0.2367 |
| title_or_five_round     | Elo                 |  687 | 57.1%      |     0.6766 |  0.2422 |
| title_or_five_round     | LightGBM (5-seed)   |  687 | 62.3%      |     0.6474 |  0.228  |
| title_or_five_round     | Logistic regression |  687 | 63.2%      |     0.6524 |  0.2302 |

## Profile-feature check (LightGBM, same tuned parameters per year)

| model                             |    n | accuracy   |   log_loss |   brier |
|:----------------------------------|-----:|:-----------|-----------:|--------:|
| LightGBM (5-seed)                 | 6684 | 61.1%      |      0.655 |  0.2315 |
| LightGBM without height/reach/age | 6684 | 59.4%      |      0.666 |  0.2368 |

## Training-start experiment (2001 vs 2005)

| model                                |    n | accuracy   |   log_loss |   brier |
|:-------------------------------------|-----:|:-----------|-----------:|--------:|
| LightGBM (5-seed)                    | 6684 | 61.1%      |     0.655  |  0.2315 |
| LightGBM, train from 2001            | 6684 | 60.8%      |     0.6556 |  0.2319 |
| Logistic regression                  | 6684 | 61.9%      |     0.6527 |  0.2305 |
| Logistic regression, train from 2001 | 6684 | 61.8%      |     0.6528 |  0.2306 |

## Global SHAP importance (top 15)

| feature            |   mean_abs_shap |
|:-------------------|----------------:|
| age_years          |          0.2791 |
| elo                |          0.1639 |
| sapm               |          0.0964 |
| td_per15           |          0.0879 |
| recent_win_rate    |          0.0638 |
| been_finished_rate |          0.0621 |
| win_rate           |          0.0549 |
| sig_def            |          0.0548 |
| recent_slpm        |          0.054  |
| recent_sapm        |          0.0529 |
| slpm               |          0.0525 |
| recent_sig_def     |          0.05   |
| kd_absorbed_per15  |          0.0478 |
| sig_acc            |          0.0438 |
| td_def             |          0.0412 |

## UFC 334 predicted win probabilities

| fighter_a        | fighter_b         | weight_class         |   p_a_win |   elo_p_a | top_drivers                                                                                    |
|:-----------------|:------------------|:---------------------|----------:|----------:|:-----------------------------------------------------------------------------------------------|
| Ciryl Gane       | Josh Hokit        | Heavyweight          |    0.5234 |    0.765  | elo +0.440; age_years -0.406; sapm +0.149; sig_def +0.130; slpm -0.093                         |
| Kayla Harrison   | Amanda Nunes      | Women's Bantamweight |    0.5215 |    0.2506 | elo -0.433; age_years +0.184; layoff_days +0.117; n_fights +0.095; sapm +0.092                 |
| Caio Borralho    | Yousri Belgaroui  | Middleweight         |    0.5779 |    0.6836 | elo +0.430; slpm -0.113; age_years +0.112; finish_win_share -0.100; recent_win_rate +0.097     |
| Uroš Medić       | Kevin Holland     | Welterweight         |    0.5574 |    0.5867 | elo +0.193; n_fights +0.133; kd_absorbed_per15 -0.111; recent_win_rate -0.107; reach_in -0.105 |
| Bilal Hasan      | Luis Gurule       | Flyweight            |    0.7221 |    0.6791 | elo +0.400; age_years +0.350; sapm +0.133; finish_win_share +0.093; been_finished_rate +0.091  |
| Stephen Thompson | Charles Radtke    | Welterweight         |    0.3837 |    0.5353 | age_years -0.378; td_per15 -0.131; slpm +0.112; n_fights -0.102; recent_slpm +0.100            |
| Drew Dober       | Chris Duncan      | Lightweight          |    0.2852 |    0.4467 | age_years -0.299; td_per15 -0.178; elo -0.144; n_fights -0.119; recent_sapm +0.069             |
| Jim Miller       | Terrance McKinney | Lightweight          |    0.3905 |    0.6121 | age_years -0.638; elo +0.203; sig_def +0.125; recent_sig_def +0.115; been_finished_rate +0.101 |
| Donte Johnson    | César Almeida     | Middleweight         |    0.781  |    0.6612 | age_years +0.590; elo +0.371; recent_sig_def +0.130; recent_win_rate -0.113; sig_def +0.111    |
| Jared Gordon     | Melquizael Costa  | Lightweight          |    0.3712 |    0.3342 | elo -0.367; age_years -0.358; recent_win_rate +0.087; slpm +0.081; been_finished_rate -0.077   |
| Macy Chiasson    | Beatriz Mesquita  | Women's Bantamweight |    0.3847 |    0.3922 | elo -0.197; recent_win_rate +0.118; recent_sig_def -0.111; sig_def -0.100; n_fights -0.099     |
| Evan Elder       | Samuel Sanches    | Lightweight          |    0.5044 |    0.5557 | age_years -0.363; is_debut +0.190; elo +0.124; layoff_days -0.118; slpm +0.109                 |
| Juan Díaz        | Alateng Heili     | Bantamweight         |    0.6948 |    0.4335 | age_years +0.398; sapm +0.197; td_per15 +0.164; elo -0.150; finish_win_share +0.102            |

## Figures

- `figures/accuracy_by_year.png`
- `figures/logloss_by_year.png`
- `figures/reliability_lgbm.png`
- `figures/reliability_lr.png`
- `figures/reliability_models.png`
- `figures/shap_global.png`
- `figures/ufc_334_drivers.png`
