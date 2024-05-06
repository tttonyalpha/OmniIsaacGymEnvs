NUM_ENVS_ARRAY=(1024 2048 4096 8192 16384 32768)
SEED_ARRAY=(10 42 1337)
TASKS_ARRAY=("AnymalTerrain" "FrankaCabinet" "ShadowHand")

PYTHON_SCRIPT="scripts/rlgames_train.py"
ISAAC_SIM_SCRIPT="/isaac-sim/python.sh -m torch.distributed.run --nnodes=1 --nproc_per_node=2"
PYTHON_COMMAND="task=TASK wandb_activate=True wandb_project=isaac-gym-benchmarks wandb_group=2xA40-4xVCPU-isaac-gym-benchmark headless=True experiment=2xA40-4xVCPU-isaac-gym-TASK-seed-SEED-envs-NUM_ENVS seed=SEED num_envs=NUM_ENVS wandb_entity=tttonyalpha multi_gpu=True max_iterations=50"

for TASK in "${TASKS_ARRAY[@]}"; do
    for NUM_ENVS in "${NUM_ENVS_ARRAY[@]}"; do
        for SEED in "${SEED_ARRAY[@]}"; do
            COMMAND="${ISAAC_SIM_SCRIPT} ${PYTHON_SCRIPT} $PYTHON_COMMAND"
            COMMAND="${COMMAND//TASK/$TASK}"
            COMMAND="${COMMAND//NUM_ENVS/$NUM_ENVS}"
            COMMAND="${COMMAND//SEED/$SEED}"
            eval "$COMMAND"
        done
    done
done
