# SOilSense - Autonomous Process Intelligence & Decision Platform

SoilSense is an end-to-end, full-stack, machine learning, and IoT platform that applies chemical engineering transport principles and machine learning to precision agriculture.

It reads real-time telemetry from an ESP32 or simulated sensor array, executes deterministic pre-inference sensor health checks, computes chemical engineering mass balances (water and nitrogen conservation), runs a Random Forest decision classifier, formulates transparent attributions ("WHY"), enforces non-negotiable hardware safety interlocks, mandates human-in-the-loop approval, actuates an active-LOW relay pump, and evaluates post-irrigation feedback into an Experiment Lab to retrain the AI model continuously.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Folder Structure](#3-folder-structure)
4. [Installation](#4-installation)
5. [Environment Variables](#5-environment-variables)
6. [Database Setup](#6-database-setup)
7. [ML Training](#7-ml-training)
8. [Running Backend](#8-running-backend)
9. [Running Frontend](#9-running-frontend)
10. [Running Simulator](#10-running-simulator)
11. [Flashing ESP32](#11-flashing-esp32)
12. [Connecting Hardware](#12-connecting-hardware)
13. [Running Tests](#13-running-tests)
14. [Demo Instructions](#14-demo-instructions)
15. [API Documentation](#15-api-documentation)
16. [Safety Notes](#16-safety-notes)
17. [Deployment Instructions](#17-deployment-instructions)
18. [Troubleshooting](#18-troubleshooting)

---

## 1. Project Overview

Conventional agricultural automation relies on crude static thresholds (e.g. `if moisture < 30%: pump ON`). In reality, soil matric potential is tightly coupled with atmospheric evaporative demand (Vapor Pressure Deficit, temperature, solar radiation) and chemical equilibria (solute concentration, osmotic salt burn, nitrate leaching).

SoilSence AI solves this by introducing:

- **Sensor Health Engine**: Evaluates telemetry validity _before_ AI inference. Catches missing data, out-of-bounds spikes, flatlining, drift, and analog vs digital sensor disagreement.
- **Mass Balances**: Calculates $\Delta M_w = W_{in} - W_{loss}$ (Water Balance) and $\Delta N = N_{input} - N_{utilization} - N_{loss}$ (Nitrogen Balance).
- **Random Forest AI**: Classifies optimal operational states into 5 deterministic outcomes: `IRRIGATE`, `FERTIGATE`, `DO NOTHING`, `WAIT / MONITOR`, and `VERIFY`.
- **Interpretable Reasoning**: Every recommendation explains **WHY** by combining model feature importances, current process states, and applicable agronomic rules.
- **Human-in-the-Loop & Safety**: Zero pump commands execute without human operator approval (or explicit Automate mode). Hard-coded hardware watchdogs enforce max runtimes and minimum cool-down gaps.
- **Closed-Loop Feedback**: Logs predicted vs. actual moisture responses to calculate prediction error %, creating a self-improving continuous learning loop.

---

## 2. Architecture

```
ESP32 Physical Hardware / Simulator
                ↓
1. Data Validation / Sensor Health Engine (bounds, flatline, jump, cross-sensor)
                ↓
2. Chemical Engineering Process State & Mass Balances (ΔMw, ΔN)
                ↓
3. Random Forest ML Classifier (100 Trees, predict_proba)
                ↓
4. Decision & Action Optimization (What, Volume, Duration, Target, Expected state)
                ↓
5. Safety Interlocks (Max runtime, cool-down gap, upper moisture cutoff)
                ↓
6. Human-in-the-Loop Approval (APPROVE | MODIFY | REJECT | AUTOMATE)
                ↓
7. ESP32 Actuation & Relay Driver (Active-LOW Optocoupler -> DC Pump)
                ↓
8. Post-Monitoring Telemetry Acquisition (5-10 min response)
                ↓
9. Experiment Lab & Feedback Logging (Predicted vs Actual, Error %)
                ↓
10. Model Retraining Trigger
```

---

## 3. Folder Structure

```
agrichem-ai/
├── backend/
│   ├── app/
│   │   ├── api/             # Modular REST API route handlers
│   │   ├── engines/         # Sensor health, mass balance, decision, safety, feedback engines
│   │   ├── ml/              # Inference engine and retraining pipeline
│   │   ├── models/          # SQLAlchemy database models
│   │   ├── schemas/         # Pydantic validation schemas
│   │   ├── simulator/       # ESP32 telemetry hardware simulator
│   │   ├── config.py        # Centralized settings
│   │   ├── database.py      # SQLAlchemy engine and session dependency
│   │   └── main.py          # FastAPI application entrypoint with demo seeder
│   ├── tests/               # 16-suite pytest automated tests
│   ├── Dockerfile           # Backend container
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── components/layout/   # TopNav, Sidebar, DashboardLayout
│   ├── lib/                 # API client and auth token storage
│   ├── pages/               # All 12 Next.js application modules + login
│   ├── styles/              # Tailwind CSS stylesheet
│   ├── types/               # TypeScript definitions
│   ├── Dockerfile           # Frontend container
│   └── package.json         # Node.js dependencies
├── ml/
│   ├── data/                # Initial synthetic & experiment dataset
│   ├── train.py             # Reproducible training script
│   └── random_forest.joblib # Serialized model artifact
├── esp32/
│   └── esp32_firmware.ino   # Arduino C++ ESP32 firmware
├── docs/
│   ├── architecture.md      # Detailed system architecture
│   └── api_spec.md          # REST API documentation
├── docker-compose.yml       # Production multi-container startup
├── .env.example             # Environment template
└── README.md                # System build manual
```

---

## 4. Installation

### Prerequisites

- Python 3.11+
- Node.js 18+ (tested on Node v20/v24)
- npm or yarn

### 1. Clone & Enter Directory

```bash
cd /home/jevith/.gemini/antigravity/scratch/agrichem-ai
```

### 2. Set Up Backend Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt email-validator
```

### 3. Set Up Frontend

```bash
cd frontend
npm install
cd ..
```

---

## 5. Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Key environment configuration variables:

```ini
PORT=8000
HOST=0.0.0.0
ENVIRONMENT=development
DATABASE_URL=sqlite:///./agrichem.db
SECRET_KEY=agrichem-ai-super-secret-key-change-in-production-2026
ESP32_DEVICE_ID=ESP32-AGRI-01
ESP32_API_KEY=esp32-secure-token-agrichem-2026
PUMP_HARDWARE_ENABLED=false
MAX_PUMP_RUNTIME_MINUTES=15
MIN_GAP_BETWEEN_PUMPS_MINUTES=10
MOISTURE_UPPER_LIMIT=85.0
```

---

## 6. Database Setup

The backend utilizes SQLAlchemy with automatic schema generation and demo benchmark data seeding.
When the backend starts up for the first time, it automatically:

1. Creates all tables (`users`, `readings`, `decisions`, `actions`, `experiments`).
2. Seeds default admin credentials (`admin@agrichem.ai` / `admin123`).
3. Seeds the exact benchmark demo telemetry:
   - Soil Moisture: 27%
   - Soil Temp: 29°C, pH: 6.4, N: 64, P: 51, K: 73
   - Air Temp: 33°C, Humidity: 65%, Solar: 910 W/m²
   - CH4: 18 ppm, CO2: 620 ppm
   - Water Balance: Supplied 12.5L, Est. Loss 9.8L, Net +2.7L
   - Nitrogen Balance: Input 100g, Uptake 64g, Loss 21g, Remaining 15g

---

## 7. ML Training

To retrain the Random Forest model independently:

```bash
python ml/train.py
```

This script:

1. Synthesizes 2,000 samples based on physical transport rules.
2. Trains a `RandomForestClassifier(n_estimators=100, random_state=42)`.
3. Validates out-of-sample accuracy (~98.75%).
4. Serializes the artifact to `ml/random_forest.joblib`.

---

## 8. Running Backend

Start the FastAPI application:

```bash
source venv/bin/activate
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI documentation will be available at: `http://localhost:8000/docs`

---

## 9. Running Frontend

Start the Next.js development server:

```bash
cd frontend
npm run dev
```

Open your browser at: `http://localhost:3000`

---

## 10. Running Simulator

The application contains a built-in telemetry simulator. You can trigger scenarios directly:

- **Via UI**: Use the **DEMO** dropdown selector in the top navigation bar.
- **Via API**:
  ```bash
  curl -X POST http://localhost:8000/api/simulation -H "Content-Type: application/json" -d '{"scenario": "DRY_SOIL"}'
  ```
  Available scenarios:
- `DRY_SOIL`: Moisture 27%, High Temp & Solar -> Recommendation: `IRRIGATE`
- `WET_SOIL`: Moisture 74% -> Recommendation: `DO NOTHING`
- `LOW_N`: Nitrogen 28 mg/kg, Moisture 48% -> Recommendation: `FERTIGATE`
- `SENSOR_FAILURE`: Moisture 142% (or analog/digital conflict) -> Recommendation: `VERIFY`
- `POST_IRRIGATION`: Moisture 51.2% -> Closes feedback loop in Experiment Lab
- `NORMAL`: Nominal operational balance

---

## 11. Flashing ESP32

1. Open Arduino IDE.
2. Install ESP32 Board Support:
   - File -> Preferences -> Additional Boards Manager URLs -> `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
3. Open `esp32/esp32_firmware.ino`.
4. Update your Wi-Fi credentials and local backend IP address:
   ```cpp
   const char* WIFI_SSID     = "YOUR_WIFI_SSID";
   const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
   const char* BACKEND_URL   = "http://192.168.1.100:8000/api";
   ```
5. Select Board: `DOIT ESP32 DEVKIT V1` and your USB serial port.
6. Click **Upload**.

---

## 12. Connecting Hardware

### Pinout Mapping:

- **Analog Soil Moisture Sensor (A0)** -> **ESP32 GPIO 34** (ADC1)
- **Digital Soil Moisture Sensor (D0)** -> **ESP32 GPIO 2**
- **Relay Module Signal (IN)** -> **ESP32 GPIO 23**
- **Status LED** -> **ESP32 GPIO 22**
- **Relay COM / NO** -> In-line with DC 12V Water Pump positive lead
- **Power**: Common Ground (GND) across ESP32, Relay board, and Sensor module.

---

## 13. Running Tests

Run the complete 16-test acceptance suite:

```bash
source venv/bin/activate
PYTHONPATH=backend pytest backend/tests/test_all_subsystems.py -v
```

All 16 acceptance test criteria pass:

1. Dry soil -> `IRRIGATE`
2. Wet soil -> `DO NOTHING`
3. Low N + suitable moisture + suitable environment -> `FERTIGATE`
4. Sensor failure -> `VERIFY`
5. Rejected action -> pump remains OFF
6. Moisture above upper limit -> no irrigation
7. Safety runtime limit enforcement
8. Minimum gap between pump runs
9. ESP32 command generation & polling
10. Experiment feedback calculation
11. What-If simulation
12. Model status & retraining endpoint
13. Cross-sensor disagreement detection
14. Water balance calculation
15. Nitrogen balance calculation
16. Complete closed-loop simulated workflow

---

## 14. Demo Instructions

Follow this step-by-step presentation sequence:

1. **Open Home / Command Center (`/`)**:
   - Inspect the 5 answers answering What is happening, Is there a problem, AI recommendation, Why, and What should I do.
   - Note live benchmark readings: Moisture 27%, Soil Temp 29°C, Air Temp 33°C, Solar 910 W/m².
2. **Click "WHY? (Explain Model)"**:
   - Inspect the numbered reasons and model feature importance breakdown.
3. **Open Live Farm (`/live`)**:
   - View live time-series charts for Soil, Canopy Atmosphere, and Emissions.
4. **Open Process Intelligence (`/process`)**:
   - Inspect the coupled Process Tree, Water Balance ($\Delta M_w = 12.5 - 9.8 = +2.7\text{ L}$), and Nitrogen Balance ($\Delta N = 100 - 64 - 21 = 15\text{ g}$).
5. **Open AI Decision Center (`/decision`)**:
   - View the 4-step pipeline: State Diagnosis -> AI Recommendation -> WHY -> Prescription Action (6 min, 12.5 L).
6. **Click "APPROVE PRESCRIPTION"**:
   - System checks safety interlocks and approves the action.
7. **Open Smart Control (`/control`)**:
   - Observe live pump state transition to `ON`.
   - Inspect the 6-stage actuation pipeline and audit trail.
8. **Simulate Post-Irrigation**:
   - Select **5. Post-Irrigation** from the TopNav DEMO dropdown.
   - Soil moisture rises to 51.2%.
9. **Open Experiment Lab (`/experiments`)**:
   - View the newly recorded experiment trial.
   - Inspect predicted target (50.0%) vs actual (51.2%) with percentage error (2.4%).
10. **Click "RETRAIN MODEL ON EXPERIMENTS"**:
    - Observe model retrain with feedback included and accuracy reported.
11. **Open Research & Reports (`/reports`)**:
    - Review sustainability metrics (water saved, NUE %, carbon offset) and click **EXPORT REPORT (CSV)**.

---

## 15. API Documentation

Comprehensive endpoint schemas are provided in [docs/api_spec.md](docs/api_spec.md).
Interactive Swagger docs: `http://localhost:8000/docs`.

---

## 16. Safety Notes

- **Default Dry Run Mode**: By default, `PUMP_HARDWARE_ENABLED=false`. Physical relays will never be energized unless explicitly set to `true`.
- **Active-LOW Relay Logic**: The firmware ensures GPIO23 boots `HIGH` so the relay remains de-energized during ESP32 power-on and reboot cycles.
- **Hardware Watchdog**: If network communication disconnects during irrigation, a firmware hardware timer cuts off pump power after 10 minutes maximum.

---

## 17. Deployment Instructions

### Docker Compose Production Startup

```bash
docker-compose up --build -d
```

- Backend runs on `http://localhost:8000`
- Frontend runs on `http://localhost:3000`

---

## 18. Troubleshooting

| Issue                               | Cause                                                                                           | Solution                                                                               |
| :---------------------------------- | :---------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------- |
| Sensor status is stuck in `FAULT`   | Telemetry value is out of physical bounds (e.g. moisture > 100) or analog/digital contradiction | Check sensor jumper cables or select `NORMAL` in DEMO dropdown.                        |
| Pump fails to turn on upon approval | `PUMP_HARDWARE_ENABLED` is set to `false` (default safe mode)                                   | Check Smart Control audit log: simulated activations are expected in development mode. |
| ESP32 cannot connect to Wi-Fi       | Incorrect SSID or password                                                                      | Edit `WIFI_SSID` and `WIFI_PASSWORD` in `esp32_firmware.ino` and reflash.              |
| Retrain button shows error          | Insufficient training rows                                                                      | Run `python ml/train.py` to regenerate base `data/training.csv`.                       |
