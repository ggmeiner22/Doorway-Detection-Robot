# Doorway_Detection-Robot

Wall-following robot with **sensor fusion** (Naive Bayes) and **PID** control.  
You learn **CPTs** from measurements, then fuse live/batched readings to estimate:

- probability you **just passed a door** (and a temporal “~10 cm ago” helper)
- **distance from wall** as a probability distribution → expected distance for PID

The repo also includes a small demo that produces plots and a CSV you can open in Excel.

---

## Repo layout

```text
Doorway_Detection-Robot/
├── cpts
│   ├── cpt_BI1.csv
│   ├── cpt_BI2.csv
│   ├── cpt_BI3.csv
│   ├── cpt_BI4.csv
│   ├── cpt_BI5.csv
│   ├── cpt_BI6.csv
│   ├── cpt_BI7.csv
│   ├── cpt_BI8.csv
│   ├── cpt_BI9.csv
│   ├── cpt_IR1.csv
│   ├── cpt_IR2.csv
│   ├── cpt_IR3.csv
│   ├── cpt_IR4.csv
│   ├── cpt_IR5.csv
│   ├── cpt_IR6.csv
│   ├── cpt_IR7.csv
│   ├── cpt_IR8.csv
│   ├── cpt_IR9.csv
│   ├── cpt_PIDD1.csv
│   ├── cpt_PIDD2.csv
│   ├── cpt_PIDD3.csv
│   ├── cpt_PIDD4.csv
│   ├── cpt_PIDD5.csv
│   ├── cpt_PIDD6.csv
│   ├── cpt_PIDD7.csv
│   ├── cpt_PIDD8.csv
│   ├── cpt_PIDD9.csv
│   ├── cpt_PIDI1.csv
│   ├── cpt_PIDI2.csv
│   ├── cpt_PIDI3.csv
│   ├── cpt_PIDI4.csv
│   ├── cpt_PIDI5.csv
│   ├── cpt_PIDI6.csv
│   ├── cpt_PIDI7.csv
│   ├── cpt_PIDI8.csv
│   ├── cpt_PIDI9.csv
│   ├── cpt_PIDP1.csv
│   ├── cpt_PIDP2.csv
│   ├── cpt_PIDP3.csv
│   ├── cpt_PIDP4.csv
│   ├── cpt_PIDP5.csv
│   ├── cpt_PIDP6.csv
│   ├── cpt_PIDP7.csv
│   ├── cpt_PIDP8.csv
│   ├── cpt_PIDP9.csv
│   ├── meta.json
│   └── prior_location.csv
├── data
│   └── measurements_v2.csv
├── docs
│   ├── collectData.py   # Old groups work EVENTUALLY REMOVE
│   ├── images
│   │   ├── network.png
│   │   ├── posterior_rep.png
│   │   └── timeseries.png
│   └── index.html
├── LICENSE
├── README.md
├── requirements.txt
├── src
│   ├── belief_network.py
│   ├── collect.py
│   ├── config.py
│   ├── control.py
│   ├── cpt_learn.py
│   ├── data
│   │   └── measurements_live.csv
│   ├── __init__.py
│   ├── pid.py
│   ├── robot_io.py
│   └── run_all_bt.py
├── requirements.txt
└── README.md
```

---

## 0) Prereqs

- Python 3.10+ (3.11 OK)
- (Recommended) VS Code with **Python**, **Pylance**, and **Remote – WSL** extensions if you use WSL.

---

## 1) Setup & Execution(venv + deps)

```bash
cd Doorway-Detection-Robot
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# To execute
python -m src.run_all_bt

```

---

## 2) What the robot does, step by step

### A) Connect
- The console prints “searching…” until the Create 3 connects over Bluetooth.
- As soon as it’s connected you’ll see “Connected to …” and Phase 1 begins.

 
## 3) License

This project is released under the **MIT License**.  

You are free to use, modify, and distribute this code, but attribution is appreciated.  
Replace the sample data & CPTs with your own robot measurements for your own use.
