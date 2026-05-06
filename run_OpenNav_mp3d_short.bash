# Set your key in the environment (do not commit real keys): export OPENAI_API_KEY=...
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for gpt-4o}"

cd "$(dirname "$0")"

# Do not put # comments inside flag= — they are forwarded to merge_from_list as bogus keys.
# Do not pass EVAL.SPLIT in flag= unless you intend to override run_OpenNav_mp3d_short.yaml.

flag="--exp_name cont-cwp-opennav-mp3d_short
      --exp-config run_OpenNav_mp3d_short.yaml
      --llm gpt-4o-2024-08-06
      --api_key ${OPENAI_API_KEY}
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_ID 0
      TORCH_GPU_IDS [0]
      "
CUDA_VISIBLE_DEVICES=2 python run.py $flag