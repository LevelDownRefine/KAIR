"""MARBLE contrastive training, validation selection and compatible checkpoints.

Adapted from LevelDownRefine/MARBLE's validated CUDA backend; see
licenses/MARBLE.txt. Graph tasks use this wrapper and main_train_marble.py,
independently of the image restoration L/H, PSNR and DataParallel contracts.
"""

import copy
import logging
import time

import torch

from models.network_marble import MARBLEEncoder, contrastive_loss
from utils.utils_marble import save_json, tensor_digest

logger = logging.getLogger(__name__)


def cpu_state(model):
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


class ModelMARBLE:
    def __init__(self, options):
        assert "params" in options and "device" in options
        self.params = copy.deepcopy(options["params"])
        self.device = torch.device(options["device"])
        self.netG = MARBLEEncoder(self.params).to(self.device)

    def epoch_loss(self, features, batches, optimizer=None):
        assert batches
        self.netG.train(optimizer is not None)
        losses = []
        for targets, ids in batches:
            with torch.set_grad_enabled(optimizer is not None):
                embedding = self.netG(features(ids, targets))
                loss = contrastive_loss(embedding)
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            losses.append(loss.detach())
        self.netG.eval()
        # The reference averages batch means, including the short last batch.
        return float(torch.stack(losses).double().mean())

    def validate_reference(self, features, validation):
        """Fail before training if three original SGD steps do not agree."""
        assert "initial_state" in validation and "batches" in validation
        assert len(validation["batches"]) == 3
        self.netG.load_state_dict(validation["initial_state"], strict=True)
        self.netG.train()
        optimizer = torch.optim.SGD(self.netG.parameters(), lr=1.0, momentum=0.9)
        errors = {}

        def compare(actual, expected, name):
            actual = actual.detach().cpu()
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
            errors[name] = float((actual - expected).abs().max())

        for step, batch in enumerate(validation["batches"]):
            assert all(
                key in batch
                for key in (
                    "ids",
                    "targets",
                    "features",
                    "embedding",
                    "loss",
                    "state_after",
                )
            )
            values = features(batch["ids"], batch["targets"])
            compare(values, batch["features"], f"{step}:features")
            embedding = self.netG(values)
            compare(embedding, batch["embedding"], f"{step}:embedding")
            loss = contrastive_loss(embedding)
            compare(loss, batch["loss"], f"{step}:loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for key, value in self.netG.state_dict().items():
                assert key in batch["state_after"]
                compare(value, batch["state_after"][key], f"{step}:{key}")
        return {"rtol": 2e-4, "atol": 2e-5, "max_absolute_errors": errors}

    def fit(self, features, plan, directory, seed):
        assert all(key in plan for key in ("initial_state", "epochs", "test"))
        assert all(key in self.params for key in ("epochs", "lr", "momentum"))
        assert len(plan["epochs"]) == self.params["epochs"] > 0
        self.params["seed"] = seed
        torch.manual_seed(seed)
        self.netG.load_state_dict(plan["initial_state"], strict=True)
        initial_digest = tensor_digest(self.netG.state_dict().items())
        optimizer = torch.optim.SGD(
            self.netG.parameters(),
            lr=self.params["lr"],
            momentum=self.params["momentum"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        history = {"train_loss": [], "val_loss": [], "test_loss": [], "lr": []}
        best_loss, best_epoch = float("inf"), -1
        started = time.perf_counter()

        def checkpoint(epoch):
            return {
                "epoch": epoch,
                "model_state_dict": cpu_state(self.netG),
                "params": copy.deepcopy(self.params),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "losses": copy.deepcopy(history),
            }

        for epoch, batches in enumerate(plan["epochs"]):
            assert "train" in batches and "val" in batches
            training = self.epoch_loss(features, batches["train"], optimizer)
            validation = self.epoch_loss(features, batches["val"])
            assert torch.isfinite(torch.tensor([training, validation])).all()
            scheduler.step(training)
            history["train_loss"].append(training)
            history["val_loss"].append(validation)
            assert "lr" in optimizer.param_groups[0]
            history["lr"].append(optimizer.param_groups[0]["lr"])
            if validation < best_loss:
                best_loss, best_epoch = validation, epoch
                torch.save(checkpoint(epoch), directory / "best_model.pth")
            if epoch == 0 or (epoch + 1) % 10 == 0:
                logger.info(
                    "Seed %d epoch %d/%d train %.6f val %.6f",
                    seed,
                    epoch + 1,
                    self.params["epochs"],
                    training,
                    validation,
                )
            save_json(
                directory / "progress.json",
                {
                    "seed": seed,
                    "epochs_completed": epoch + 1,
                    "train_loss": training,
                    "val_loss": validation,
                },
            )
        # This test sampler belongs to the training graph, not the held-out recording.
        history["test_loss"].append(self.epoch_loss(features, plan["test"]))
        torch.save(checkpoint(self.params["epochs"] - 1), directory / "last_model.pth")
        best = torch.load(
            directory / "best_model.pth", map_location="cpu", weights_only=True
        )
        assert "model_state_dict" in best
        self.netG.load_state_dict(best["model_state_dict"], strict=True)
        self.netG.eval()
        trained_digest = tensor_digest(self.netG.state_dict().items())
        assert trained_digest != initial_digest
        save_json(directory / "loss_history.json", history)
        return history, {
            "seed": seed,
            "training_epochs": self.params["epochs"],
            "representation_training_performed": True,
            "initial_state_sha256": initial_digest,
            "trained_state_sha256": trained_digest,
            "best_epoch_zero_based": best_epoch,
            "best_validation_loss": best_loss,
            "training_seconds": time.perf_counter() - started,
        }

    @torch.no_grad()
    def embeddings(self, train_features, test_features):
        best = cpu_state(self.netG)
        outputs = {}
        for mode in ("eval", "notebook"):
            self.netG.load_state_dict(best, strict=True)
            self.netG.train(mode == "notebook")
            outputs[mode] = {
                "train": self.netG(train_features).cpu(),
                "test": self.netG(test_features).cpu(),
            }
        # Test-batch BN statistics must never leak into the saved best model.
        self.netG.load_state_dict(best, strict=True)
        self.netG.eval()
        return outputs
