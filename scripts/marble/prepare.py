"""Export original MARBLE graphs and sampling plans using its pinned CPU env."""

import argparse
import logging
import os
import subprocess
from pathlib import Path

SOURCE_COMMIT = "872e46bd6dff2d092f8554a8c084701450e84904"
logger = logging.getLogger(__name__)


def prepare(repository, python, output, protocol="achilles", data=None):
    repository, python, output = map(Path.resolve, (repository, python, output))
    if output.exists():
        raise FileExistsError(f"Choose a new export directory: {output}")
    for path in (python, repository / "reproduction/src/prepare_gpu_training.py"):
        if not path.is_file():
            raise FileNotFoundError(path)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    if revision != SOURCE_COMMIT:
        raise ValueError(f"Expected MARBLE commit {SOURCE_COMMIT}, got {revision}")
    changes = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repository,
        text=True,
    ).strip()
    if changes:
        raise ValueError("MARBLE tracked sources must be clean for reference export.")
    assert protocol in ("achilles", "rat-consistency")
    root = Path(__file__).resolve().parents[2]
    command = [str(python), "-u"]
    if protocol == "rat-consistency":
        assert data is not None, "The rat-consistency protocol requires --data"
        command += [
            str(root / "scripts/marble/rat_reference.py"),
            "--marble-repo",
            str(repository),
            "--data",
            str(Path(data).resolve()),
        ]
    else:
        command.append(str(repository / "reproduction/src/prepare_gpu_training.py"))
    command += ["--output", str(output)]
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=os.pathsep.join((str(root), str(repository / "reproduction/src"))),
        PYTHONIOENCODING="utf-8",
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(output.parent / ".matplotlib"),
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        NUMBA_NUM_THREADS="4",
    )
    logger.info("Exporting original %s protocol to %s", protocol, output)
    subprocess.run(
        command,
        cwd=repository / "reproduction",
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marble-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol", choices=("achilles", "rat-consistency"), default="achilles"
    )
    parser.add_argument("--data", type=Path)
    arguments = parser.parse_args()
    prepare(
        arguments.marble_repo,
        arguments.python,
        arguments.output,
        arguments.protocol,
        arguments.data,
    )
