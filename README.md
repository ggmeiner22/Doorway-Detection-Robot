# Doorway_Detection-Robot

**Project Website:** [**https://ggmeiner22.github.io/Doorway-Detection-Robot/**](https://ggmeiner22.github.io/Doorway-Detection-Robot/)

Wall-following robot with **sensor fusion** (Naive Bayes) and **PID** control.  
You learn **CPTs** from measurements, then fuse live/batched readings to estimate:

- probability you **just passed a door** (and a temporal “~10 cm ago” helper)
- **distance from wall** as a probability distribution → expected distance for PID

The repo also includes a small demo that produces plots and a CSV you can open in Excel.

---

## Repo layout

```text
TBD
```

---

## 0) Prereqs

- Python 3.10+ (3.11 OK)
- (Recommended) VS Code with **Python** and **Pylance**.
  - Note: WSL does not support BLE

---

## 1) Setup & Execution (venv + deps)

```bash
cd Doorway-Detection-Robot
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# To execute
python -m src.scripts.collect
python -m src.scripts.train
python -m src.scripts.run
```

---

## 2) What the robot does, step by step

---

## 3) Files you will see
`data/measurements_live.csv` — auto-labeled training data from warm-up.

`cpts/*.csv` — learned BN tables used by control.

---

## 4) Stop / rerun
**Ctrl+C** to stop.

On later runs, you can skip collect and train and reuse CPTs.

---

## 5) License

This project is released under the **MIT License**.  

You are free to use, modify, and distribute this code, but attribution is appreciated.  
Replace the sample data & CPTs with your own robot measurements for your own use.
