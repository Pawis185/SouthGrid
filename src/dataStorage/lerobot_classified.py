"""Category-routed LeRobot writers with deferred staging commit.

Recording streams into a single staging writer. After human classification the
parked episode is committed to exactly one category dataset, or discarded.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from dataStorage.lerobot_data_storage import LeRobotDatasetWriter

GOOD = "GOOD"
QUALIFIED = "QUALIFIED"
DOUBTFUL = "DOUBTFUL"
DELETE = "DELETE"

CATEGORY_DIR = {
    GOOD: "good",
    QUALIFIED: "qualified",
    DOUBTFUL: "Doubtful",
}

REPO_SUFFIX = {
    GOOD: "good",
    QUALIFIED: "qualified",
    DOUBTFUL: "doubtful",
}


def derive_repo_id(base_repo_id: str, category: str) -> str:
    return f"{base_repo_id}_{REPO_SUFFIX[category]}"


class ClassifiedLeRobotHub:
    """Own one staging writer and lazily resume each category writer."""

    def __init__(
        self,
        parent_root: str,
        repo_id: str,
        writer_kwargs: dict,
        staging_root: str,
        writer_factory: Callable | None = None,
        staging: "LeRobotDatasetWriter | None" = None,
    ) -> None:
        self.parent_root = os.path.abspath(os.path.expanduser(parent_root))
        self.repo_id = repo_id
        self._writer_kwargs = dict(writer_kwargs)
        if writer_factory is None:
            from dataStorage.lerobot_data_storage import LeRobotDatasetWriter
            writer_factory = LeRobotDatasetWriter.create_or_resume
        self._factory = writer_factory
        self._writers: dict[str, object] = {}
        self._closed = False
        self.staging_root = os.path.abspath(os.path.expanduser(staging_root))
        if staging is not None:
            self.staging = staging
        else:
            staging_path = Path(self.staging_root)
            if staging_path.exists():
                shutil.rmtree(staging_path)
            from dataStorage.lerobot_data_storage import LeRobotDatasetWriter
            self.staging = LeRobotDatasetWriter.create(
                repo_id=f"{repo_id}_staging",
                root=self.staging_root,
                resume=False,
                **self._writer_kwargs,
            )

    def category_root(self, category: str) -> str:
        return os.path.join(self.parent_root, CATEGORY_DIR[category])

    def writer_for(self, category: str):
        if category == DELETE:
            raise ValueError("DELETE 不创建数据集")
        if category not in CATEGORY_DIR:
            raise ValueError(f"未知分类: {category}")
        writer = self._writers.get(category)
        if writer is None:
            writer = self._factory(
                repo_id=derive_repo_id(self.repo_id, category),
                root=self.category_root(category),
                **self._writer_kwargs,
            )
            self._writers[category] = writer
        return writer

    def commit(self, category: str) -> int | None:
        if category == DELETE:
            self.staging.discard_episode()
            return None
        dest = self.writer_for(category)
        return self.staging.commit_uncommitted_to(dest)

    def discard(self) -> None:
        self.staging.discard_episode()

    def counts(self) -> dict[str, int]:
        out = {}
        for cat in (GOOD, QUALIFIED, DOUBTFUL):
            writer = self._writers.get(cat)
            out[cat] = int(writer.num_episodes) if writer is not None else 0
        return out

    @property
    def total_episodes(self) -> int:
        return sum(self.counts().values())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.staging.close()
        except Exception:
            pass
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception:
                pass

    def __enter__(self) -> "ClassifiedLeRobotHub":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
