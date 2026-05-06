import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

prompt_gen_output_folder = Path("/ssd1/seoyoung/ns_vln_bench/prompt_gen/outputs/gibson-20260416")
eval_episodes_file = Path(
    "/ssd1/seoyoung/vln-zero.github.io/VLN_CE/data/datasets/ANSR_V1/gibson_short/gibson_short.json.gz"
)
# Single OpenNav *_snapshot.json (episode_id → metrics dict), **or** a directory of
# VLN-zero-style stats (one JSON per episode with fields "id", "success", BSA/CSR).
eval_stats_path = Path(
    "/ssd1/seoyoung/Open-Nav/logs/eval_results/cont-cwp-opennav-gibson_short/"
    "stats_ep_ckpt_gibson_short_r0_w1_snapshot.json"
)

# If True, drop any episode whose eval row has NaN/Inf or non-numeric BSA, CSR,
# or success (same spirit as analyze_results_opennav default).
EXCLUDE_NONFINITE_EVAL = True

# Ablation: hold one dimension fixed while averaging BSA/CSR over the other.
DEPTH_HELD_FIXED = 1
CHAIN_LENGTH_HELD_FIXED = 1


def _safe_metric(value, default=0.0):
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _to_finite_float(value):
    """Finite scalar as float, or None if missing / non-numeric / NaN / Inf."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def load_episodes_data(path):
    """Load ANSR/VLN-CE episodes JSON or .json.gz."""
    path = Path(path)
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_eval_bsa_csr(path: Path, *, exclude_nonfinite: bool = True):
    """
    Build id → (bsa, csr) and successful episode ids (as strings).

    - If ``path`` is a **file**, expect OpenNav snapshot: top-level mapping
      ``episode_id → { branch_selection_accuracy, conditional_success_rate, success, … }``.
    - If ``path`` is a **directory**, expect one ``*.json`` per episode with
      keys ``id``, ``branch_selection_accuracy``, ``conditional_success_rate``, ``success``
      (original vln-zero layout).

    If ``exclude_nonfinite`` is True, episodes where any of BSA / CSR / success is
    missing or non-finite are omitted entirely (not counted toward success_ids either).

    Returns ``(id_metrics_dict, success_ids, n_skipped_nonfinite)``.
    """
    id_metrics_dict = {}
    success_ids = []
    n_skipped_nonfinite = 0

    if path.is_file():
        with path.open(encoding="utf-8") as f:
            blob = json.load(f)
        if not isinstance(blob, dict):
            raise SystemExit(f"Expected JSON object in snapshot {path}")

        for eid, row in blob.items():
            if not isinstance(row, dict):
                continue
            key = str(eid)
            if exclude_nonfinite:
                bsa = _to_finite_float(row.get("branch_selection_accuracy"))
                csr = _to_finite_float(row.get("conditional_success_rate"))
                succ = _to_finite_float(row.get("success"))
                if bsa is None or csr is None or succ is None:
                    n_skipped_nonfinite += 1
                    continue
                id_metrics_dict[key] = (bsa, csr)
                if succ > 0.0:
                    success_ids.append(key)
            else:
                bsa = _safe_metric(row.get("branch_selection_accuracy"), 0.0)
                csr = _safe_metric(row.get("conditional_success_rate"), 0.0)
                id_metrics_dict[key] = (bsa, csr)
                if _safe_metric(row.get("success"), 0.0) > 0.0:
                    success_ids.append(key)
        return id_metrics_dict, success_ids, n_skipped_nonfinite

    if path.is_dir():
        for json_file in sorted(path.glob("*.json")):
            with json_file.open(encoding="utf-8") as f:
                data = json.load(f)
            raw_id = data.get("id")
            if raw_id is None:
                continue
            key = str(raw_id)
            if exclude_nonfinite:
                bsa = _to_finite_float(data.get("branch_selection_accuracy"))
                csr = _to_finite_float(data.get("conditional_success_rate"))
                succ = _to_finite_float(data.get("success"))
                if bsa is None or csr is None or succ is None:
                    n_skipped_nonfinite += 1
                    continue
                id_metrics_dict[key] = (bsa, csr)
                if succ > 0.0:
                    success_ids.append(key)
            else:
                id_metrics_dict[key] = (
                    _safe_metric(data.get("branch_selection_accuracy"), 0.0),
                    _safe_metric(data.get("conditional_success_rate"), 0.0),
                )
                if _safe_metric(data.get("success"), 0.0) > 0.0:
                    success_ids.append(key)
        return id_metrics_dict, success_ids, n_skipped_nonfinite

    raise SystemExit(f"Not a file or directory: {path}")


def scene_name_from_prompt_source_graph(source_scene_graph: str) -> str:
    """
    Align prompt_gen ``source_scene_graph`` stem with ``Path(ep['scene_id']).stem``.

    Paths look like ``…/Adairsville_relations_semantic_3dbb.json``; the old
    ``.replace('_relations_semantic', '')`` left ``Adairsville_3dbb``, which never
    matched Habitat's ``Adairsville`` from ``gibson/Adairsville/Adairsville.glb``.
    """
    stem = Path(source_scene_graph).stem
    if "_relations_semantic" in stem:
        return stem.split("_relations_semantic", 1)[0]
    return stem.replace("_relations_semantic", "")


WARN_PROMPT_KEY_MISSING_CAP = 12
WARN_EVAL_ID_MISSING_CAP = 12


###################################################################################
instruct_complexity_dict = {}
for json_file in prompt_gen_output_folder.glob("*.json"):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        source_scene_graph = data["metadata"]["source_scene_graph"]
        scene_name = scene_name_from_prompt_source_graph(source_scene_graph)

        for _category_name, instructions in data["categories"].items():
            for item in instructions:
                instruction = item["instruction"]
                depth = item["depth"]
                chain_length = item["chain_length"]

                key_k = (scene_name, instruction)
                value_k = (depth, chain_length)
                instruct_complexity_dict[key_k] = value_k

    except json.JSONDecodeError:
        print(f"Warning: invalid JSON file skipped: {json_file}")
    except Exception as e:
        print(f"Warning: error reading {json_file}: {e}")

##################################################################################
id_metrics_dict, success_ids, n_skipped_nonfinite = load_eval_bsa_csr(
    eval_stats_path, exclude_nonfinite=EXCLUDE_NONFINITE_EVAL
)

print(f"Loaded eval metrics from: {eval_stats_path.resolve()}")
if EXCLUDE_NONFINITE_EVAL and n_skipped_nonfinite:
    print(
        "Skipped episodes (missing or non-finite BSA / CSR / success):",
        n_skipped_nonfinite,
    )
print("success episodes:\t", len(success_ids))

#################################################################################
no_subgoals_ids = []

episodes_root = load_episodes_data(eval_episodes_file)

for ep in episodes_root["episodes"]:
    episode_id = ep["episode_id"]
    if ep["subgoals"] == []:
        no_subgoals_ids.append(str(episode_id))

print("no subgoal episodes:\t", len(no_subgoals_ids))

##################################################################################
count = 0
for key in id_metrics_dict:
    if (key in no_subgoals_ids) and (key in success_ids):
        count += 1
        id_metrics_dict[key] = (1.0, 1.0)
print("succ & no subgoal epi:\t", count)

##################################################################################

complexity_metrics = []
match_count = 0
missing_complexity = 0
missing_eval_id = 0
warned_complexity = 0
warned_eval = 0

data = episodes_root

for ep in data["episodes"]:
    scene_id = ep["scene_id"]
    episode_id = ep["episode_id"]
    scene_name = Path(scene_id).stem
    instruction_text = ep["instruction"]["instruction_text"]

    key = (scene_name, instruction_text)
    if key not in instruct_complexity_dict:
        missing_complexity += 1
        if warned_complexity < WARN_PROMPT_KEY_MISSING_CAP:
            snippet = (
                instruction_text[:120] + "…"
                if len(instruction_text) > 120
                else instruction_text
            )
            print(
                f"Warning: no prompt_gen depth/len for scene={scene_name!r} "
                f"instruction={snippet!r}"
            )
            warned_complexity += 1
        continue

    complexity = instruct_complexity_dict[key]
    ekey = str(episode_id)
    if ekey not in id_metrics_dict:
        missing_eval_id += 1
        if warned_eval < WARN_EVAL_ID_MISSING_CAP:
            print(f"Warning: episode_id {ekey!r} missing from eval snapshot/log; skip")
            warned_eval += 1
        continue
    metrics = id_metrics_dict[ekey]

    item = (complexity, metrics)
    complexity_metrics.append(item)
    match_count += 1

print("Total episodes:\t", len(data["episodes"]))
print("Matching episodes:\t", match_count)
if missing_complexity:
    extra = missing_complexity - warned_complexity
    print(
        "Episodes with no (scene, instruction) in prompt_gen outputs: "
        f"{missing_complexity}"
        + (
            f" (only printed first {warned_complexity} warnings)"
            if extra > 0
            else ""
        )
    )
if missing_eval_id:
    extra_ev = missing_eval_id - warned_eval
    print(
        f"Episodes skipped (id not in eval): {missing_eval_id}"
        + (f" (only printed first {warned_eval} warnings)" if extra_ev > 0 else "")
    )


####################################################################################

def _print_averages(sums, label_name):
    for k in sorted(sums.keys()):
        sum1, sum2, n = sums[k]
        print(
            f"  {label_name}={k}\t(bsa, csr)=({sum1 / n:.6f}, {sum2 / n:.6f})\t n={n}"
        )


# Vary chain_length only: keep depth fixed.
print()
print(f"=== Ablation: by chain_length (depth held at {DEPTH_HELD_FIXED}) ===")
sums_by_len = defaultdict(lambda: [0.0, 0.0, 0])
n_depth_fixed = 0
for (depth, chain_length), metrics in complexity_metrics:
    if depth != DEPTH_HELD_FIXED:
        continue
    n_depth_fixed += 1
    sums_by_len[chain_length][0] += metrics[0]
    sums_by_len[chain_length][1] += metrics[1]
    sums_by_len[chain_length][2] += 1
print(f"Episodes with depth=={DEPTH_HELD_FIXED}: {n_depth_fixed}")
_print_averages(sums_by_len, "chain_length")

# Vary depth only: keep chain_length fixed.
print()
print(f"=== Ablation: by depth (chain_length held at {CHAIN_LENGTH_HELD_FIXED}) ===")
sums_by_depth = defaultdict(lambda: [0.0, 0.0, 0])
n_len_fixed = 0
for (depth, chain_length), metrics in complexity_metrics:
    if chain_length != CHAIN_LENGTH_HELD_FIXED:
        continue
    n_len_fixed += 1
    sums_by_depth[depth][0] += metrics[0]
    sums_by_depth[depth][1] += metrics[1]
    sums_by_depth[depth][2] += 1
print(f"Episodes with chain_length=={CHAIN_LENGTH_HELD_FIXED}: {n_len_fixed}")
_print_averages(sums_by_depth, "depth")

# Full (depth, chain_length) joint buckets (same as non-ablation script).
print()
print("=== Joint (depth, chain_length) buckets ===")
sums_joint = defaultdict(lambda: [0.0, 0.0, 0])
for (depth, chain_length), metrics in complexity_metrics:
    jkey = (depth, chain_length)
    sums_joint[jkey][0] += metrics[0]
    sums_joint[jkey][1] += metrics[1]
    sums_joint[jkey][2] += 1
for jkey in sorted(sums_joint.keys()):
    sum1, sum2, n = sums_joint[jkey]
    print(f"  (depth, len)={jkey}\t(bsa, csr)=({sum1 / n:.6f}, {sum2 / n:.6f})\t n={n}")
