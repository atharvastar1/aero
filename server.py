"""
AeroShield Live Simulation Server
----------------------------------
FastAPI + WebSocket server that:
  1. Trains the Random Forest stall-detection model once on startup.
  2. Accepts WebSocket connections from the dashboard.
  3. Runs the full physics simulation (with Dryden wind gusts) step-by-step
     and streams every Nth frame as JSON in real time.

Usage:
    pip install fastapi uvicorn
    python server.py
"""

import asyncio
import json
import math
import pathlib

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ── Import the simulation engine ───────────────────────────────────────────────
from stall_prevention_optimized import (
    AircraftFlightModel,
    DatasetGenerator,
    StallDetectionModel,
    StallPreventionController,
)

# ── Constants ──────────────────────────────────────────────────────────────────
STREAM_EVERY_N_STEPS = 2   # send one frame per 2 physics steps = 50 Hz
DT = 0.01
DURATION = 20.0
FRAME_INTERVAL = STREAM_EVERY_N_STEPS * DT  # 0.02 s
BASE_DIR = pathlib.Path(__file__).parent

# ── One-time model training at startup ─────────────────────────────────────────
print("[SERVER] Training stall-detection model…")
_generator = DatasetGenerator(samples=6000, random_state=42)
_data = _generator.generate()
_balanced = _generator.balance_dataset(_data)

X_all = _balanced[:, :4]
y_all = _balanced[:, 4].astype(int)

from sklearn.model_selection import train_test_split
X_train, _, y_train, _ = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

STALL_MODEL = StallDetectionModel(model_type="random_forest")
STALL_MODEL.train_random_forest(X_train, y_train)
CONTROLLER = StallPreventionController(STALL_MODEL, threshold=0.7)
print("[SERVER] Model ready.")

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="AeroShield Live API")

# Serve the dashboard and outputs/ folder as static files
app.mount("/outputs", StaticFiles(directory=str(BASE_DIR / "outputs")), name="outputs")


@app.get("/")
async def root():
    return FileResponse(str(BASE_DIR / "dashboard_live.html"))


# ── WebSocket endpoint ─────────────────────────────────────────────────────────
@app.websocket("/ws/simulate")
async def simulate(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected.")

    try:
        # ── 1. Receive configuration from the dashboard ──────────────────────
        raw = await websocket.receive_text()
        cfg = json.loads(raw)

        scenario     = cfg.get("scenario", "normal")         # normal | climb | stall
        ai_enabled   = cfg.get("ai_enabled", True)
        wind_mode    = cfg.get("wind", "moderate")           # light | moderate | severe | off

        print(f"[WS] Scenario={scenario}  AI={ai_enabled}  Wind={wind_mode}")

        # ── 2. Build aircraft with / without wind ───────────────────────────
        wind_on = wind_mode != "off"
        aircraft = AircraftFlightModel(
            dt=DT,
            wind_intensity=wind_mode if wind_on else "light",
            wind_enabled=wind_on,
        )

        # ── 3. Send initial "ready" handshake ────────────────────────────────
        await websocket.send_json({
            "type": "ready",
            "scenarios": scenario,
            "ai": ai_enabled,
            "wind": wind_mode,
        })

        # ── 4. Simulation loop ───────────────────────────────────────────────
        time = 0.0
        step = 0
        total_steps = int(DURATION / DT)
        stall_count = 0
        loop_start = asyncio.get_event_loop().time()  # wall-clock reference

        while step < total_steps:
            # Pilot profile
            if scenario == "climb":
                if time < 5:
                    elevator_base, throttle_base = 0.20, 0.70
                elif time < 10:
                    elevator_base, throttle_base = 0.35, 0.50
                else:
                    elevator_base, throttle_base = 0.10, 0.40
            elif scenario == "stall":
                if time < 3:
                    elevator_base, throttle_base = 0.15, 0.65
                elif time < 7:
                    elevator_base, throttle_base = 0.75, 0.15  # hard nose-up, throttle cut
                elif time < 12:
                    elevator_base, throttle_base = 0.55, 0.10  # sustained stall
                else:
                    elevator_base, throttle_base = 0.10, 0.65  # recovery attempt
            else:   # normal — visible sinusoidal pilot input, AI damps but doesn't fully cancel
                # Use a larger amplitude so pitch visibly swings ±8-10° even when AI is on
                elevator_base = 0.30 * math.sin(time * 0.7) + 0.08 * math.sin(time * 1.8)
                throttle_base = 0.45  # slightly below mid to prevent velocity pinning at max

            # AI override
            if ai_enabled and STALL_MODEL.is_trained:
                state_vec = np.array([[
                    aircraft.pitch,
                    aircraft.pitch_rate,
                    aircraft.velocity,
                    throttle_base,
                ]])
                stall_prob = float(STALL_MODEL.predict_probability(state_vec)[0])
                elevator_cmd, throttle_cmd = CONTROLLER.compute_control_action(
                    {"pitch": aircraft.pitch,
                     "pitch_rate": aircraft.pitch_rate,
                     "velocity": aircraft.velocity},
                    stall_prob,
                )
            else:
                elevator_cmd = elevator_base
                throttle_cmd = throttle_base
                stall_prob = 0.0

            state = aircraft.step(elevator_cmd, throttle_cmd)

            if state["stalled"]:
                stall_count += 1

            # Stream every Nth step to avoid overwhelming the WebSocket
            if step % STREAM_EVERY_N_STEPS == 0:
                frame = {
                    "type":       "frame",
                    "t":          round(time, 3),
                    "pitch":      round(math.degrees(state["pitch"]), 3),
                    "aoa":        round(math.degrees(state["aoa"]), 3),
                    "velocity":   round(state["velocity"], 3),
                    "altitude":   round(state["altitude"], 3),
                    "stall_prob": round(stall_prob, 4),
                    "stalled":    int(state["stalled"]),
                    "elevator":   round(elevator_cmd, 3),
                    "throttle":   round(throttle_cmd, 3),
                    "wind_u":     round(state.get("wind_speed", 0.0), 3),
                }
                await websocket.send_json(frame)

                # ── Real-time pacing ──────────────────────────────────────────
                # Calculate when this frame *should* have arrived in wall time,
                # then sleep the remaining gap so playback matches 1:1 physics time.
                target_wall = loop_start + time
                now = asyncio.get_event_loop().time()
                sleep_for = target_wall - now
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                else:
                    await asyncio.sleep(0)  # always yield to event loop

            time += DT
            step += 1

        # ── 5. Send summary ───────────────────────────────────────────────
        await websocket.send_json({
            "type":         "done",
            "stall_count":  stall_count,
            "duration":     DURATION,
            "scenario":     scenario,
            "ai_enabled":   ai_enabled,
        })
        print(f"[WS] Simulation done — stalls={stall_count}")

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as exc:
        print(f"[WS] Error: {exc}")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model_ready": STALL_MODEL.is_trained}


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8765, reload=False)
