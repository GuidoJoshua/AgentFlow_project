#!/bin/bash

# ===========================================================================
# Script: serve_vllm_qwen.sh
# Description:
#   Launch the AgentFlow planner, Qwen2.5-7B-Instruct, and
#   Qwen2.5-Coder-7B-Instruct with vLLM in separate tmux sessions.
#
# Usage:
#   PLANNER_GPU=2 INSTRUCT_GPU=0 CODER_GPU=1 bash scripts/serve_vllm_qwen.sh
#   PLANNER_GPU=0 INSTRUCT_GPU=0 CODER_GPU=0 bash scripts/serve_vllm_qwen.sh
# ===========================================================================

set -euo pipefail

#PLANNER_MODEL="${PLANNER_MODEL:-AgentFlow/agentflow-planner-7b}"
#PLANNER_GPU="${PLANNER_GPU:-2}"
#PLANNER_PORT="${PLANNER_PORT:-8000}"
#PLANNER_SESSION="${PLANNER_SESSION:-vllm_agentflow_planner}"

INSTRUCT_MODEL="${INSTRUCT_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
INSTRUCT_GPU="${INSTRUCT_GPU:-2}"
INSTRUCT_PORT="${INSTRUCT_PORT:-8001}"
INSTRUCT_SESSION="${INSTRUCT_SESSION:-vllm_qwen_instruct}"

CODER_MODEL="${CODER_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
CODER_GPU="${CODER_GPU:-3}"
CODER_PORT="${CODER_PORT:-8002}"
CODER_SESSION="${CODER_SESSION:-vllm_qwen_coder}"

TP="${TP:-1}"
VENV_ACTIVATE="${VENV_ACTIVATE:-source .venv/bin/activate}"

start_server() {
    local model="$1"
    local gpu="$2"
    local port="$3"
    local session="$4"

    if tmux has-session -t "$session" 2>/dev/null; then
        echo "tmux session '$session' already exists. Kill it first or choose another session name."
        return 1
    fi

    echo "Launching model: $model"
    echo "  Session: $session"
    echo "  Port:    $port"
    echo "  GPU:     $gpu"
    echo "  TP:      $TP"

    tmux new-session -d -s "$session"

    local cmd_start="
        cd /home/minsung.bae/AgentFlow;
        $VENV_ACTIVATE;
        export CUDA_VISIBLE_DEVICES=$gpu;
        echo '--- Starting $model on port $port with TP=$TP ---';
        echo 'CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES';
        echo 'Current virtual env: \$(python -c \"import sys; print(sys.prefix)\")';
        vllm serve \"$model\" \
            --host 0.0.0.0 \
            --port $port \
            --tensor-parallel-size $TP
    "

    tmux send-keys -t "${session}:0" "$cmd_start" C-m
}

#start_server "$PLANNER_MODEL" "$PLANNER_GPU" "$PLANNER_PORT" "$PLANNER_SESSION"
start_server "$INSTRUCT_MODEL" "$INSTRUCT_GPU" "$INSTRUCT_PORT" "$INSTRUCT_SESSION"
start_server "$CODER_MODEL" "$CODER_GPU" "$CODER_PORT" "$CODER_SESSION"

echo ""
echo "=== Planner and Qwen servers launched"
#echo "Planner:  tmux attach-session -t $PLANNER_SESSION"
echo "Instruct: tmux attach-session -t $INSTRUCT_SESSION"
echo "Coder:    tmux attach-session -t $CODER_SESSION"
