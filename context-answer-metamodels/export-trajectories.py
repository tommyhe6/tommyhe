"""Project saved answer-token activations for the website's selected rollouts."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from tokenizers import Tokenizer


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_record(path, index):
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["index"] == index:
            return record
    raise ValueError(f"Missing record {index} in {path.name}")


def text_offsets(tokenizer, token_ids, text, length):
    assert tokenizer.decode(token_ids, skip_special_tokens=True) == text
    offsets = []
    for count in range(1, len(token_ids) + 1):
        prefix = tokenizer.decode(token_ids[:count], skip_special_tokens=True)
        # A token can end partway through a UTF-8 character.
        while prefix.endswith("\ufffd") and not text.startswith(prefix):
            prefix = prefix[:-1]
        assert text.startswith(prefix)
        offsets.append(len(prefix.encode(encoding="utf-16-le")) // 2)
    offsets.extend([offsets[-1]] * (length - len(offsets)))
    assert offsets == sorted(offsets)
    return offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["dataset-root", "demo", "provenance", "projection", "tokenizer", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    demo = read_json(args.demo)
    provenance = read_json(args.provenance)
    source_indices = {item["id"]: item["sourceIndex"] for item in provenance["examples"]}
    projection = load_file(str(args.projection))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    output = {
        "metadata": {
            "schemaVersion": 1,
            "definition": "Cumulative mean of saved decoder-block-18 answer-token activations, rounded to bfloat16 and projected with the original training-fitted PCA.",
            "answerSuffix": "The final frame includes the saved non-overlapping canonical assistant suffix, matching the study target.",
            "textOffsets": "UTF-16 character offsets from decoding the exact generated token IDs; canonical suffix tokens do not add displayed text.",
            "projectionSha256": hashlib.sha256(args.projection.read_bytes()).hexdigest(),
            "tokenizerSha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
            "modelRevision": demo["metadata"]["modelRevision"],
            "maximumEndpointDifference": 0,
            "endpointPolicy": "Completed rollouts use the original archived endpoint; independently reconstructed endpoints must agree within 0.002 projected units.",
        },
        "examples": [],
    }
    for example in demo["examples"]:
        index = source_indices[example["id"]]
        exported = {"id": example["id"], "rollouts": []}
        for observation in example["observations"]:
            seed = observation["seed"]
            if seed < 47:
                source = args.dataset_root / f"Qwen3.5-9B_seed{seed}"
            else:
                source = args.dataset_root / "Qwen3.5-9B_layer18_100k_16seeds" / "parts" / f"seed{seed}" / "rows_0000000_0024999"
            config = read_json(source / "run_config.json")
            assert config["model_revision"] == demo["metadata"]["modelRevision"]
            assert config["layers"] == [18]
            shard_size = config["activation_shard_size"]
            start = index // shard_size * shard_size
            stem = f"full_{start:06d}_{start + shard_size - 1:06d}"
            span = read_record(source / "data" / "activation_spans" / f"{stem}.jsonl", index)
            response_start = index // 100 * 100
            record = read_record(source / "data" / "rollouts" / f"rollouts_{response_start:06d}_{response_start + 99:06d}.jsonl", index)
            assert record["response"] == observation["text"]
            assert span["prompt_sha256"] == record["prompt_sha256"] == example["promptHash"]
            assert span["response_sha256"] == record["response_sha256"] == hashlib.sha256(observation["text"].encode()).hexdigest()
            assert record["generated_token_count"] == observation["tokenCount"]
            with safe_open(source / "activations" / "full_tokens" / f"{stem}.safetensors", framework="pt", device="cpu") as handle:
                tokens = handle.get_tensor(span["token_ids_key"])
                generated_ids = tokens[span["generated_span_start"]:span["generated_span_end_exclusive"]].tolist()
                assert generated_ids == record["generated_token_ids"]
                hidden = handle.get_slice(span["activation_key"])[0, span["answer_span_start"]:span["answer_span_end_exclusive"], :]
            assert hidden.shape == (observation["tokenCount"] + span["appended_suffix_length"], 4096)
            assert torch.isfinite(hidden).all()
            raw_hash = hashlib.sha256(hidden.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()
            counts = torch.arange(1, hidden.shape[0] + 1, dtype=torch.float32).unsqueeze(1)
            means = (hidden.float().cumsum(dim=0) / counts).to(dtype=torch.bfloat16).float()
            points = (means - projection["center"]) @ projection["basis"]
            error = (points[-1] - torch.tensor(observation["point"])).abs().max().item()
            assert error < 0.002, (example["id"], seed, error)
            output["metadata"]["maximumEndpointDifference"] = max(error, output["metadata"]["maximumEndpointDifference"])
            exported["rollouts"].append({
                "seed": seed,
                "points": [[round(x, 5), round(y, 5)] for x, y in points.tolist()],
                "textEnds": text_offsets(tokenizer, generated_ids, observation["text"], hidden.shape[0]),
                "activationSha256": raw_hash,
                "generatedTokenIdsSha256": hashlib.sha256(json.dumps(generated_ids, separators=(",", ":")).encode()).hexdigest(),
            })
        output["examples"].append(exported)
        print(f"Exported {example['id']}: {len(exported['rollouts'])} token trajectories", flush=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(output["metadata"], indent=2), flush=True)


if __name__ == "__main__":
    main()
