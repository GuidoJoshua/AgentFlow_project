#!/bin/bash

# Configuration
TASK="bamboogle"
THREADS=10
DATA_FILE_NAME="data.json"
FIXED_PORT=8001
FIXED_BASE_URL="http://localhost:${FIXED_PORT}/v1"
CODER_PORT=8002
CODER_BASE_URL="http://localhost:${CODER_PORT}/v1"

MODELS=(
    "${FIXED_PORT}:vllm-Qwen/Qwen2.5-7B-Instruct,Qwen2.5-7B-Instruct-emb,\
    Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,\
    gpt-4o-mini|vllm-Qwen/Qwen2.5-Coder-7B-Instruct@${CODER_BASE_URL}|Default|Default,\
    trainable|vllm-Qwen/Qwen2.5-7B-Instruct|fixed|fixed"
    #"8000:vllm-AgentFlow/agentflow-planner-7b,AgentFlow-7B-embedding,\
    #Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,\
    #gpt-4o-mini|vllm-Qwen/Qwen2.5-Coder-7B-Instruct@${CODER_BASE_URL}|Default|Default,\
    #trainable|vllm-Qwen/Qwen2.5-7B-Instruct|fixed|fixed"
#     ":dashscope-qwen2.5-7b-instruct,Qwen2.5-7b-naive,\
# Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,\
# dashscope-qwen2.5-7b-instruct|dashscope-qwen2.5-7b-instruct|Default|Default,\
# trainable|dashscope|dashscope|dashscope"
)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Set project directory to parent (test/ folder)
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd $PROJECT_DIR

# Loop through all models
for MODEL_SPEC in "${MODELS[@]}"; do
    # Parse model specification
    PORT=$(echo "$MODEL_SPEC" | cut -d":" -f1)
    REST=$(echo "$MODEL_SPEC" | cut -d":" -f2-)
    IFS="," read -r LLM LABEL ENABLED_TOOLS_RAW TOOL_ENGINE_RAW MODEL_ENGINE_RAW FIXED_BASE_URL_OVERRIDE_RAW <<< "$REST"
    ENABLED_TOOLS=$(echo "$ENABLED_TOOLS_RAW" | tr "|" ",")
    TOOL_ENGINE=$(echo "$TOOL_ENGINE_RAW" | tr "|" ",")
    MODEL_ENGINE=$(echo "$MODEL_ENGINE_RAW" | tr "|" ",")
    MODEL_FIXED_BASE_URL="${FIXED_BASE_URL_OVERRIDE_RAW:-$FIXED_BASE_URL}"
    [ -z "$MODEL_ENGINE" ] && MODEL_ENGINE="trainable,dashscope,dashscope,dashscope"
    
    if [ -n "$PORT" ]; then
        BASE_URL="http://localhost:${PORT}/v1"
        USE_BASE_URL=true
    else
        BASE_URL=""
        USE_BASE_URL=false
    fi

    echo "========================================"
    echo "MODEL: $LLM"
    echo "LABEL: $LABEL"
    echo "TASK: $TASK"
    echo "========================================"

    DATA_FILE="$TASK/data/$DATA_FILE_NAME"
    LOG_DIR="$TASK/logs/$LABEL"
    OUT_DIR="$TASK/results/$LABEL"
    CACHE_DIR="$TASK/cache"

    mkdir -p "$LOG_DIR"
    mkdir -p "$OUT_DIR"

    INDICES=($(seq 0 49))

    # Skip already completed indices
    new_indices=()
    for i in "${INDICES[@]}"; do
        if [ ! -f "$OUT_DIR/output_$i.json" ]; then
            new_indices+=($i)
        fi
    done
    indices=("${new_indices[@]}")

    if [ ${#indices[@]} -eq 0 ]; then
        echo "All subtasks completed for $TASK with $LABEL."
    else
        run_task() {
            local i=$1
            if [ "$USE_BASE_URL" = true ]; then
                uv run python solve.py --index $i --task "$TASK" --data_file "$DATA_FILE" --llm_engine_name "$LLM" --root_cache_dir "$CACHE_DIR" --output_json_dir "$OUT_DIR" --output_types direct --enabled_tools "$ENABLED_TOOLS" --tool_engine "$TOOL_ENGINE" --model_engine "$MODEL_ENGINE" --max_time 300 --max_steps 10 --temperature 0.0 --base_url "$BASE_URL" --fixed_base_url "$MODEL_FIXED_BASE_URL" 2>&1 | tee "$LOG_DIR/$i.log"
            else
                uv run python solve.py --index $i --task "$TASK" --data_file "$DATA_FILE" --llm_engine_name "$LLM" --root_cache_dir "$CACHE_DIR" --output_json_dir "$OUT_DIR" --output_types direct --enabled_tools "$ENABLED_TOOLS" --tool_engine "$TOOL_ENGINE" --model_engine "$MODEL_ENGINE" --max_time 300 --max_steps 10 --temperature 0.0 --fixed_base_url "$MODEL_FIXED_BASE_URL" 2>&1 | tee "$LOG_DIR/$i.log"
            fi
        }
        export -f run_task
        export TASK DATA_FILE LOG_DIR OUT_DIR CACHE_DIR LLM ENABLED_TOOLS TOOL_ENGINE MODEL_ENGINE BASE_URL MODEL_FIXED_BASE_URL USE_BASE_URL
        parallel -j $THREADS run_task ::: "${indices[@]}"
    fi

    # Calculate Scores
    uv run python calculate_score_unified.py --task_name "$TASK" --data_file "$DATA_FILE" --result_dir "$OUT_DIR" --response_type "direct_output" --output_file "finalresults_direct_output.json" | tee "$OUT_DIR/finalscore_direct_output.log"
done
