# Doorway_Detection-Robot

**Project Website:** [**https://ggmeiner22.github.io/Doorway-Detection-Robot/**](https://ggmeiner22.github.io/Doorway-Detection-Robot/)

Wall-following robot with **sensor fusion** (Naive Bayes) and **PID** control.  
You learn **CPTs** from measurements, then fuse live/batched readings to estimate:

- probability you **just passed a door** (and a temporal “~10 cm ago” helper)
- **distance from wall** as a probability distribution → expected distance for PID

The repo also includes a small demo that produces plots and a CSV you can open in Excel.

---

## Contents

- [Why](#why)
- [Features](#features)
- [Repo layout](#repo-layout)
- [Requirements](#requirements)
- [Install](#install)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [How it works (high-level)](#how-it-works-high-level)
- [Troubleshooting & tips](#troubleshooting--tips)
- [License](#license)

---

## Why

Door transitions create distinctive short-lived signatures in near-wall sensing. By **collecting real measurements**, learning **conditional probability tables (CPTs)**, and then running a lightweight **Bayes + PID** loop, you can turn those signatures into a robust, explainable “door-passed?” signal while keeping stable wall-following.

---

## Features

- **Data → CPTs → Online fusion** pipeline (collect → train → run).  
- **Naive Bayes** for interpretable inference; **PID** for smooth control.  
- **Reproducible demo**: generates **plots** and an **Excel-ready CSV**.  
- Modular folders for **data**, **learned CPTs**, **code**, and **docs**.

---

## Repo layout

```text
Doorway-Detection-Robot/
├─ cpts/ # Learned CPTs (CSV)
├─ data/ # Raw logs & derived datasets
├─ docs/ # Notes, figures
├─ src/ # Python source (collection, training, runtime)
├─ requirements.txt
└─ README.md
```
> The code is organized so you can: **collect** measurements, **train** CPTs, then **run** the controller with live or replayed data.

---

## Requirements

- **Python 3.10+** (3.11 OK).  
- Recommended: VS Code + Python extension.  
- **Note for WSL** users: **WSL does not support BLE**; use native Linux/Windows for Bluetooth-based runs.

---

## Install

```bash
git clone https://github.com/ggmeiner22/Doorway-Detection-Robot
cd Doorway-Detection-Robot

python -m venv .venv
# Windows:
.\.venv\Scripts\Activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

---

## Quickstart

**1) Collect data** (drive near a wall; mark/label door events as your script supports):

```bash
python -m src.scripts.collect
```
- This creates/updates raw logs in *data/* (e.g., measurements with labels).

**2) Train CPTS** (learn Naive Bayes tables from your logs):

```bash
python -m src.scripts.train
```
- This writes learned tables to *cpts/*.csv*.

**3) Run** (online fusion + PID):

```bash
python -m src.scripts.run
```
- You’ll see plots and a timeseries CSV suitable for spreadsheets.
> On subsequent runs, you can **skip collection/training** and reuse existing CPTs in *cpts/*.

---

## Configuration

Key knobs you’ll typically adjust:
- **Setpoint** (desired wall distance).
- **PID gains** (Kp/Ki/Kd).
- **Windowing / temporal features** for the “10 cm ago” door signature.
- **Signal preprocessing** (e.g., median/EMA smoothing).
- **Labeling schema** for training (consistent, precise timestamps).
Store your defaults in a small config file or env vars and keep learned tables in *cpts/*.

---

## Outputs

After a run you should expect:
- *data/measurements_live.csv* – raw or lightly processed measurements with labels (example file name).
- *cpts/*.csv* – learned CPTs.
- *data/belief_timeseries.csv* – fused probabilities, expected distance, control outputs.
- **Plots** – quick visual checks (distance vs. setpoint, door probability spikes).

---

## How it works (high-level)

1. **Collect**: sample proximity/IR/other signals while wall-following; label door transitions.  
2. **Train**: fit **Naive Bayes** CPTs from labeled windows (e.g., deltas/ratios, short temporal context).  
3. **Fuse**: at runtime, compute **P(door_passed | signals)** and a **distance distribution**; use the **expected distance** as the measurement for **PID**.  
4. **Control**: PID regulates wheel commands to track the setpoint while door probability is monitored for events.

This provides a small, explainable loop that’s easy to tune and iterate with field data.

---

## Troubleshooting & tips

- **No motion / no control effect**: confirm your velocity publisher/SDK path, and that command topic or device connection is correct for your platform (e.g., Create3: verify message types and namespaces).  
- **BLE on WSL**: not supported—run on native Windows or Linux for Bluetooth connectivity.  
- **Noisy signals**: add smoothing (EMA/median), clip outliers, or expand training data with more varied wall materials and door frames.  
- **Door probability not peaking**: re-check labels and your temporal window (e.g., align the distance-change feature to “~10 cm” of forward travel).  
- **Reproducibility**: save your raw logs; keep each learning run’s CPTs versioned in `cpts/`.

---

## License

MIT — see [LICENSE](./LICENSE).

---

### Citation / Acknowledgments

- iRobot® Create® platform interfaces & docs are helpful for message definitions and setup.

