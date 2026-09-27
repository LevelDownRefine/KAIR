"""Generate the tiny PyG 2.1 reference fixture with pinned MARBLE CPU Python.

Run from KAIR with PYTHONPATH containing KAIR and the original MARBLE repository.
This generator never imports KAIR's network or loss implementation.
"""

import copy
import importlib.metadata
from pathlib import Path

import cebra
import numpy as np
import torch
from cebra.integrations.sklearn.helpers import align_embeddings
from MARBLE.main import loss_fun
from torch.nn import functional as F
from torch_geometric.nn import MLP

from scripts.marble.rat_reference import capture_dropout
from utils.utils_marble import save_json, sha256


def serialized(value):
    return {"dtype": str(value.dtype).removeprefix("torch."), "values": value.tolist()}


def state(model):
    values = {
        f"enc.{name}": serialized(value) for name, value in model.state_dict().items()
    }
    values["diffusion.diffusion_time"] = serialized(torch.tensor(0.0))
    return values


def generate(path):
    assert importlib.metadata.version("torch-geometric") == "2.1.0.post1"
    torch.set_num_threads(1)
    torch.manual_seed(19)
    encoder = MLP([8, 8, 3], dropout=0.5, norm="batch_norm").train()
    initial = state(encoder)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=1.0, momentum=0.9)
    steps = []
    for index in range(3):
        features = torch.randn(12, 8)
        native = copy.deepcopy(encoder)
        torch.manual_seed(index)
        with capture_dropout() as masks:
            embedding = F.normalize(encoder(features), dim=-1)
        torch.manual_seed(index)
        torch.testing.assert_close(
            embedding, F.normalize(native(features), dim=-1), rtol=0, atol=0
        )
        loss = loss_fun()(embedding, torch.zeros(len(embedding), dtype=torch.bool))
        optimizer.zero_grad()
        loss.backward()
        gradients = {
            f"enc.{name}": serialized(value.grad)
            for name, value in encoder.named_parameters()
        }
        optimizer.step()
        steps.append(
            {
                "features": serialized(features),
                "dropout_mask": serialized(masks[0]),
                "embedding": serialized(embedding),
                "loss": serialized(loss),
                "gradients": gradients,
                "state_after": state(encoder),
            }
        )
    encoder.eval()
    fixture = {
        "generator_sha256": sha256(Path(__file__)),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torch-geometric")
        },
        "initial_state": initial,
        "steps": steps,
        "eval_embedding": serialized(F.normalize(encoder(features), dim=-1)),
        "native_cpu_dropout_matched": True,
    }
    rng = np.random.RandomState(29)
    embeddings = [rng.normal(size=(36, 3)).astype(np.float32) for _ in range(4)]
    labels = [np.linspace(0, 1, 36) + i * 0.01 for i in range(4)]
    animals = ["achilles", "buddy", "cicero", "gatsby"]
    scores, pairs, _ = cebra.sklearn.metrics.consistency_score(
        embeddings=embeddings,
        labels=labels,
        dataset_ids=animals,
        between="datasets",
        num_discretization_bins=9,
    )
    fixture["consistency"] = {
        "embeddings": [e.tolist() for e in embeddings],
        "labels": [y.tolist() for y in labels],
        "aligned": [
            np.asarray(e).tolist()
            for e in align_embeddings(embeddings, labels, n_bins=9)
        ],
        "animals": animals,
        "n_bins": 9,
        "scores": scores.tolist(),
        "pairs": pairs.tolist(),
        "cebra_version": importlib.metadata.version("cebra"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, fixture)


if __name__ == "__main__":
    generate(
        Path(__file__).resolve().parents[2]
        / "tests/fixtures/marble_dropout_reference.json"
    )
