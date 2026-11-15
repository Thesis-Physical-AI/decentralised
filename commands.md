# Physical AI startup commands
# Copy the section you need into a terminal. Do not run this whole file as a script.
# Prerequisites: existing central/edge .venv environments, local model weights,
# built Isaac Sim, and GEMINI_API_KEY configured in centralized/.env.

# ============================================================================
# 1. DASHBOARD / UI SERVER — recommended entry point
# ============================================================================
cd "$HOME/Desktop/Projects/centralized"
.venv/bin/python mission_dashboard.py

# In another terminal, open the UI:
xdg-open http://127.0.0.1:8787

# Submit a natural-language mission in the UI. The dashboard automatically runs
# central planning, starts/reuses the VLM server, opens the decentralized Isaac
# Sim executor, ingests feedback, and requests a replan after a halted mission.
# The UI and its HTTP server are the same process; there is no separate UI build.
# Results: centralized/outputs/dashboard/<run-id>/
# Use Stop run in the UI to stop a mission. Ctrl+C stops the dashboard server.
# Do not run a manual robot mission at the same time as a dashboard mission.

# ============================================================================
# 2. VLM SERVER — optional manual startup, in its own terminal
# ============================================================================
# Skip this if the server is already running; the dashboard can start it for you.
cd "$HOME/Desktop/Projects/decentralised"
.venv/bin/python vlm_server/smolvlm_server.py \
  --model models/SmolVLM-Instruct \
  --profile workstation_gpu_live

# In another terminal, view the VLM camera and answers:
xdg-open http://127.0.0.1:8765/view

# Check VLM server readiness:
curl --fail http://127.0.0.1:8765/health

# ============================================================================
# 3. MANUAL CENTRALIZED -> DECENTRALIZED -> FEEDBACK -> REPLANNING SEQUENCE
# ============================================================================
# Keep these commands in ONE terminal so the variables remain available.
# For VLM perception, leave the VLM server from section 2 running separately.
# Central components are CLI processes, not a separate persistent central server.

PHYSICAL_AI_ROOT="$HOME/Desktop/Projects"
PHYSICAL_AI_RUN="$(date +%Y%m%d-%H%M%S)"
PHYSICAL_AI_OUT="$PHYSICAL_AI_ROOT/centralized/outputs/manual-$PHYSICAL_AI_RUN"
mkdir -p "$PHYSICAL_AI_OUT"

# A. Centralized: natural-language request -> reasoning/interface/orchestrator.
# Edit natural_language in this request file to choose a different mission.
cd "$PHYSICAL_AI_ROOT/centralized"
.venv/bin/python reasoning-layer/run_mission.py \
  --input reasoning-layer/examples/replanning_one_room_request.json \
  --handoff-output "$PHYSICAL_AI_OUT/candidate.json" \
  --output "$PHYSICAL_AI_OUT/dispatch.json"

# Continue only if dispatch.json was produced successfully.
# B. Decentralized: execute the central dispatch with the visible Isaac Sim app.
cd "$PHYSICAL_AI_ROOT/decentralised"
"$PHYSICAL_AI_ROOT/isaacsim/_build/linux-x86_64/release/python.sh" run_prototype.py \
  --dispatch "$PHYSICAL_AI_OUT/dispatch.json" \
  --plan-revision 0 \
  --perceiver vlm --approach live \
  --out "$PHYSICAL_AI_OUT/simulation"

# Alternative baseline: replace --perceiver vlm --approach live above with
# --perceiver ground_truth (no VLM required). Add --headless to hide the window.
# A blocked mission can exit nonzero and still produce valid results.json.
# Continue only if simulation/results.json exists.

# C. Centralized: validate results and ingest the feedback/world snapshot.
cd "$PHYSICAL_AI_ROOT/centralized"
.venv/bin/python -m structured_mission_interface.feedback \
  --dispatch "$PHYSICAL_AI_OUT/dispatch.json" \
  --report "$PHYSICAL_AI_OUT/simulation/results.json" \
  --plan-revision 0 \
  --output "$PHYSICAL_AI_OUT/feedback.json"

# D. Centralized reasoning: request a replan ONLY if execution status is HALTED.
.venv/bin/python reasoning-layer/replan_mission.py \
  --input reasoning-layer/examples/replanning_one_room_request.json \
  --dispatch "$PHYSICAL_AI_OUT/dispatch.json" \
  --report "$PHYSICAL_AI_OUT/simulation/results.json" \
  --plan-revision 0 \
  --output "$PHYSICAL_AI_OUT/replan.json"

# Add --dry-run to the replanning command to inspect the prompt without an API call.
# Replanning returns a proposal for review; it never activates a replacement plan.
# Output files must be new. Start again with a new PHYSICAL_AI_RUN for another run.
