# Set your key in the environment (do not commit real keys): export OPENAI_API_KEY=...
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for gpt-4o}"

flag="--exp_name cont-cwp-opennav-gibson
      --exp-config run_OpenNav_gibson.yaml
      --llm gpt-4o-2024-08-06
      --api_key ${OPENAI_API_KEY}
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_ID 0
      TORCH_GPU_IDS [0]
      EVAL.SPLIT all
      "
CUDA_VISIBLE_DEVICES=0 python run.py $flag