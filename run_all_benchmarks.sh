#!/bin/bash -e

# Master launcher for all benchmarking tools using SLURM array jobs
# Submits separate array job for each tool, with array indexes corresponding to tasks
#
# Usage: bash run_all_benchmarks.sh [OPTIONS]
#
# This script:
# 1. Parses tool/task selection arguments
# 2. For each selected tool, submits a SLURM array job
# 3. Each array element runs one task (determined by SLURM_ARRAY_TASK_ID)
# 4. Reports job IDs for monitoring
#
# See benchmark_args_parser.sh or run with --help for options

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the shared argument parser
source "$SCRIPT_DIR/benchmark_args_parser.sh"

# Parse arguments manually first to handle help and validation
TASKS_ARG=""
TOOLS_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark-tasks)
            TASKS_ARG="$2"
            shift 2
            ;;
        --tools)
            TOOLS_ARG="$2"
            shift 2
            ;;
        -h|--help)
            cat << EOF
Usage: $(basename "$0") [OPTIONS]

Master launcher for all benchmarking tools. Submits separate SLURM array jobs
for each tool, with array indexes corresponding to tasks. For n tools and m tasks,
this creates n array jobs with m parallel elements each.

Options:
    --benchmark-tasks TASKS    Comma-separated list of tasks to run
                              Available: ${ALL_TASKS[*]}
                              Default: all tasks
    
    --tools TOOLS              Comma-separated list of tools to run
                              Available: ${CPU_TOOLS[*]} ${GPU_TOOLS[*]}
                              Default: all tools
    
    -h, --help                Show this help message

Examples:
    # Run all tools on all tasks (creates 9 array jobs, each with 6 tasks)
    bash $(basename "$0")

    # Run only on CRC and IBD tasks (creates 9 array jobs, each with 2 tasks)
    bash $(basename "$0") --benchmark-tasks "crc,ibd"

    # Run only Random Forest (creates 1 array job with 6 tasks)
    bash $(basename "$0") --tools "random_forest"

    # Run all GPU tools (creates 3 array jobs, each with 6 tasks)
    bash $(basename "$0") --tools "deep_micro,fully_connected,tabpfn"


EOF
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            echo "Run with --help for usage information" >&2
            exit 1
            ;;
    esac
done

# Determine which tools and tasks to run
SELECTED_TOOLS_LIST=()
SELECTED_TASKS_LIST=()

if [[ -n "$TOOLS_ARG" ]]; then
    string_to_array "$TOOLS_ARG" SELECTED_TOOLS_LIST
    # Validate tools
    for tool in "${SELECTED_TOOLS_LIST[@]}"; do
        if ! array_contains "$tool" "${CPU_TOOLS[@]}" && ! array_contains "$tool" "${GPU_TOOLS[@]}"; then
            echo "Error: Unknown tool: $tool" >&2
            echo "Available CPU tools: ${CPU_TOOLS[*]}" >&2
            echo "Available GPU tools: ${GPU_TOOLS[*]}" >&2
            exit 1
        fi
    done
else
    # Use all tools
    SELECTED_TOOLS_LIST=("${CPU_TOOLS[@]}" "${GPU_TOOLS[@]}")
fi

# Determine tasks
if [[ -n "$TASKS_ARG" ]]; then
    string_to_array "$TASKS_ARG" SELECTED_TASKS_LIST
    # Validate tasks
    for task in "${SELECTED_TASKS_LIST[@]}"; do
        if ! array_contains "$task" "${ALL_TASKS[@]}"; then
            echo "Error: Unknown task: $task" >&2
            echo "Available tasks: ${ALL_TASKS[*]}" >&2
            exit 1
        fi
    done
else
    # Use all tasks
    SELECTED_TASKS_LIST=("${ALL_TASKS[@]}")
fi

# Validate we have tools and tasks
if [[ ${#SELECTED_TOOLS_LIST[@]} -eq 0 ]]; then
    echo "Error: No tools selected" >&2
    exit 1
fi

if [[ ${#SELECTED_TASKS_LIST[@]} -eq 0 ]]; then
    echo "Error: No tasks selected" >&2
    exit 1
fi

# Build task list argument
TASKS_LIST=$(IFS=,; echo "${SELECTED_TASKS_LIST[*]}")

# Calculate array range
NUM_TASKS=${#SELECTED_TASKS_LIST[@]}
ARRAY_RANGE="0-$((NUM_TASKS - 1))"

# Track submitted jobs
SUBMITTED_JOBS=()

echo "════════════════════════════════════════════════════════════"
echo "Benchmark Array Job Launcher"
echo "════════════════════════════════════════════════════════════"
echo "Tasks: ${SELECTED_TASKS_LIST[*]}"
echo "Tools: ${SELECTED_TOOLS_LIST[*]}"
echo "Array size: $NUM_TASKS tasks per tool"
echo "Total job arrays: ${#SELECTED_TOOLS_LIST[@]}"
echo "Total parallel tasks: $((${#SELECTED_TOOLS_LIST[@]} * NUM_TASKS))"
echo "════════════════════════════════════════════════════════════"
echo ""

# Submit array job for each tool
for tool in "${SELECTED_TOOLS_LIST[@]}"; do
    # Determine if CPU or GPU tool
    if array_contains "$tool" "${CPU_TOOLS[@]}"; then
        if [[ "$tool" == "siamcat" || "$tool" == "debias_m" ]]; then
            SCRIPT="$SCRIPT_DIR/run_long_cpu_tools.sh"
            TOOL_TYPE="CPU (LONG)"
        else
            SCRIPT="$SCRIPT_DIR/run_medium_cpu_tools.sh"
            TOOL_TYPE="CPU (MEDIUM)"
        fi
    else
        SCRIPT="$SCRIPT_DIR/run_gpu_tools.sh"
        TOOL_TYPE="GPU"
    fi
    
    # Submit array job
    echo "→ Submitting array job for: $tool ($TOOL_TYPE)"
    echo "  Array range: $ARRAY_RANGE"
    echo "  Script: $(basename "$SCRIPT")"
    
    JOB_ID=$(sbatch --parsable --array="$ARRAY_RANGE" "$SCRIPT" \
        --tool "$tool" \
        --benchmark-tasks "$TASKS_LIST")
    
    SUBMITTED_JOBS+=("$tool:$JOB_ID")
    
    echo "  ✓ Job ID: $JOB_ID"
    echo ""
done

# Print summary
echo "════════════════════════════════════════════════════════════"
echo "Submitted ${#SUBMITTED_JOBS[@]} array job(s)"
echo "════════════════════════════════════════════════════════════"

for job in "${SUBMITTED_JOBS[@]}"; do
    tool="${job%%:*}"
    id="${job##*:}"
    printf "  %-25s %s\n" "$tool:" "$id"
done

# Extract just job IDs for monitoring commands
JOB_IDS=()
for job in "${SUBMITTED_JOBS[@]}"; do
    JOB_IDS+=("${job##*:}")
done

echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
echo ""
echo "Check output (each array task creates separate log):"
echo "  ls -ltr slurm-*.out | tail"
echo ""
echo "Cancel all jobs if needed:"
echo "  scancel $(IFS=' '; echo "${JOB_IDS[*]}")"
echo ""

