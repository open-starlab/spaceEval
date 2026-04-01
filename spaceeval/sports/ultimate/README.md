# Ultimate Frisbee - CRSV Framework

This module implements spatial value evaluation for Ultimate Frisbee match analysis using **Counterfactual-Ready Space Value (CRSV)**.

---

## Overview

The CRSV framework integrates the following processes:

- **Initiation Detection**: Detect movement initiations from real plays
- **Pitch Control** (`wUPPCF`): Compute control regions considering player and disc speeds
- **Counterfactual Simulation**: Generate counterfactual scenarios
- **Space Value** (`Vtiming`): Quantify the spatial value gap between actual and optimal plays

---

## Processing Flow

An example analysis pipeline is shown below.

### 1. Data Preprocessing

```python
from PreProcessing.preprocessing import Space_data

provider = "UFATrack"  # UltimateTrack or UFATrack

# Preprocess input tracking data
Space_data(
    data_provider=provider,
    tracking_data_path="./data",
    out_path="./output",
).preprocessing()
```

**Output:**
- `./output/event/` - Event data
- `./output/home_tracking/` - Home team coordinates
- `./output/away_tracking/` - Away team coordinates

---

### 2. Initiation Detection

```python
detected_plays_dict = Space_data(
    data_provider=provider,
    tracking_data_path="./data",
    out_path="./output",
    testing_mode=True,
).detect_initiations()

# Select the target play via user input
files_list = list(detected_plays_dict.values())
selected_idx = int(input("Select file index: "))
selected_file = files_list[selected_idx]
```

---

### 3. Counterfactual Generation

```python
from spaceEval.spaceeval.sports.ultimate.crsv.ultimate_generate_counterfactual_main_class import (
    ultimate_generate_counterfactual,
)

selected_file_path = f"./output/initiation/plays/{selected_file}.csv"
counterfactual_out_path = f"./output/initiation/counterfactuals/{selected_file}"

generator = ultimate_generate_counterfactual(
    input_data=selected_file_path,
    provider=provider,
    out_path=counterfactual_out_path,
    testing_mode=False,
)

results = generator.generate_counterfactuals()
```

**Output:**
- Multiple counterfactual scenario CSV files
- Filename examples: `1_1_2-id1-play1_0.csv`, `1_1_2-id1-play1_-12.csv`

---

### 4. Counterfactual Preprocessing

```python
counterfactual_output_path = (
    f"./output/initiation/counterfactuals_processed/{selected_file}"
)

Space_data(
    data_provider=provider,
    tracking_data_path=counterfactual_out_path,
    out_path=counterfactual_output_path,
).preprocessing()
```

**Output:**
- `event/`, `home_tracking/`, `away_tracking/`

---

### 5. wUPPCF Calculation

```python
from spaceEval.spaceeval.sports.ultimate.wuppcf.ultimate_wuppcf_main_class import (
    ultimate_wuppcf,
)

counterfactual_wuppcf_out_path = (
    f"./output/initiation/counterfactuals_wuppcf/{selected_file}"
)

wuppcf_model = ultimate_wuppcf(
    event_data=f"{counterfactual_output_path}/event",
    tracking_home=f"{counterfactual_output_path}/home_tracking",
    tracking_away=f"{counterfactual_output_path}/away_tracking",
    provider=provider,
    out_path=counterfactual_wuppcf_out_path,
)

wuppcf_results = wuppcf_model.get_wuppcf()
```

**Output:**
- `wUPPCF/` - 3D array (frame × grid Y × grid X)
- `player_wUPPCF/` - 4D array (player × frame × grid Y × grid X)

---

### 6. Vtiming Calculation

```python
from spaceEval.spaceeval.sports.ultimate.crsv.ultimate_crsv_main_class import (
    ultimate_crsv,
)

counterfactual_crsv_out_path = (
    f"./output/initiation/counterfactuals_crsv/{selected_file}"
)

crsv_model = ultimate_crsv(
    wuppcf_path=f"{counterfactual_wuppcf_out_path}/player_wUPPCF",
    scenario_path=counterfactual_out_path,
    out_path=counterfactual_crsv_out_path,
    provider=provider,
)

v_frame_results, v_scenario_results, v_timing_results = crsv_model.calc_all()
```

**Output:**
- `v_frame/` - Frame-level spatial value
- `v_scenario/` - Maximum spatial value per scenario
- `v_timing.csv` - Difference between actual and optimal play (final result)

---

## Related Package

- [PreProcessing](https://github.com/open-starlab/PreProcessing)
- [spaceEval](https://github.com/open-starlab/spaceEval)

