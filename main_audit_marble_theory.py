"""Label-free applicability audit; never computes position MAE or consistency R2."""

import argparse
import logging
from pathlib import Path

import torch

from models.marble_math_common import PhaseDictionary
from models.marble_rrr_theory import choose_ridge, rrr, spectral_certificate
from models.network_marble_experiment import inverse_sqrt
from utils.utils_marble import read_json, save_json, sha256

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
RATS = ("achilles", "buddy", "cicero", "gatsby")


def moments(values):
    assert len(values) > 2
    left, right = values[:-1], values[1:]
    left, right = left - left.mean(0), right - right.mean(0)
    return (
        left.T @ left / len(left),
        right.T @ right / len(left),
        left.T @ right / len(left),
    )


def matrix_state(c0, c1, cross, rank, ridge, method):
    if method == "rrr":
        result = rrr(c0, cross, rank, ridge)
        return result["h"], result["singular"], [float(result["condition"])]
    assert method == "vamp"
    b = inverse_sqrt(c0, ridge) @ cross @ inverse_sqrt(c1, ridge)
    singular = torch.linalg.svdvals(b)
    conditions = []
    for c in (c0, c1):
        eigen = torch.linalg.eigvalsh(c)
        conditions.append(float((eigen[-1] + ridge) / (eigen[0] + ridge)))
    return b @ b.T, singular, conditions


def audit_graph(graph, saved_operator, rank, device):
    # Reuse the exact original dictionary: this audit conditions on it being fixed.
    state = torch.load(saved_operator, weights_only=True, map_location=device)
    assert "dictionary" in state
    dictionary = PhaseDictionary()
    for name in ("mean", "scale", "ids", "anchors", "bandwidth"):
        assert name in state["dictionary"]
        setattr(dictionary, name, state["dictionary"][name])
    phase = torch.cat([graph["pos"], graph["x"]], dim=1).to(device, torch.float64)
    values = dictionary.kernel(phase)
    values = values / values.sum(1, keepdim=True)
    full = moments(values)
    split = len(values) // 2
    # Neither half contains a time pair crossing the split boundary.
    halves = [moments(values[:split]), moments(values[split:])]
    c0, c1, _ = full
    ridges = {
        "vamp": (c0.trace() + c1.trace()) / (2 * len(c0)) * 1e-6,
        "rrr": choose_ridge(c0, condition_budget=100),
    }
    records = {}
    for method, ridge in ridges.items():
        h, singular, conditions = matrix_state(*full, rank, ridge, method)
        comparisons = []
        for half in halves:
            other_h, _, _ = matrix_state(*half, rank, ridge, method)
            comparisons.append(spectral_certificate(h, other_h, rank))
        ratio = float((singular[rank] / singular[rank - 1]).square())
        records[method] = {
            "ridge": float(ridge),
            "condition_numbers": conditions,
            "singular_values": singular.cpu().tolist(),
            "subspace_iteration_factor": ratio,
            "comparisons_full_to_halves": comparisons,
        }
    return records


@torch.no_grad()
def audit(input_path, rat_input, old_run, output, device):
    assert torch.cuda.is_available() and torch.version.cuda == "13.0"
    assert not output.exists(), (
        "Preserve previous audits; choose a new output directory"
    )
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    old_protocol = read_json(old_run / "protocol.json")
    assert sha256(input_path / "training_input.pt") == old_protocol["input_sha256"]
    assert sha256(rat_input / "rat_consistency.pt") == old_protocol["rat_input_sha256"]
    pack = torch.load(input_path / "training_input.pt", weights_only=True)
    rat_pack = torch.load(rat_input / "rat_consistency.pt", weights_only=True)
    output.mkdir(parents=True)
    paths = [
        Path(__file__),
        ROOT / "models/marble_rrr_theory.py",
        ROOT / "models/marble_math_common.py",
        ROOT / "models/network_marble_experiment.py",
        ROOT / "docs/MARBLE_MATH_FOUNDATIONS.md",
        ROOT / "uv.lock",
    ]
    save_json(
        output / "protocol.json",
        {
            "behavioral_labels_used": False,
            "benchmarks_run": False,
            "condition_budget": 100,
            "decision_rule": (
                "All 30 RRR full-to-half comparisons require 2*epsilon < gap"
            ),
            "interpretation": (
                "Finite observed perturbations, not population confidence bounds"
            ),
            "dictionary": "Frozen original VAMP dictionary; not refit on halves",
            "ridge": "Selected on complete neural training data, then frozen on halves",
            "sources_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
            "input_sha256": old_protocol["input_sha256"],
            "rat_input_sha256": old_protocol["rat_input_sha256"],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
    )
    results, operator_hashes = {}, {}
    for seed in (0, 1, 2):
        for name in ("decoding", *RATS):
            graph = (
                pack["train_graph"]
                if name == "decoding"
                else rat_pack["animals"][name]["graph"]
            )
            rank = 32 if name == "decoding" else 3
            operator = old_run / f"seed-{seed}/{name}/fitted_operator.pt"
            key = f"{seed}/{name}"
            operator_hashes[key] = sha256(operator)
            results[key] = {"rank": rank, **audit_graph(graph, operator, rank, device)}
            logger.info(
                "Audited seed %d %s, rank %d; no behavioral labels", seed, name, rank
            )
    certificates = [
        c["nontrivial_certificate"]
        for v in results.values()
        for c in v["rrr"]["comparisons_full_to_halves"]
    ]
    assert len(certificates) == 30
    summary = {
        "benchmarks_run": False,
        "rrr_nontrivial_certificates": sum(certificates),
        "comparisons": len(certificates),
        "may_proceed_to_behavior_benchmark": all(certificates),
        "operator_sha256": operator_hashes,
        "results": results,
    }
    save_json(output / "audit.json", summary)
    logger.info(
        "RRR nontrivial certificates: %d/%d; behavior benchmark gate: %s",
        sum(certificates),
        len(certificates),
        all(certificates),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rat-input", type=Path, required=True)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    audit(args.input, args.rat_input, args.old_run, args.output, args.device)
