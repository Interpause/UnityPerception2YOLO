from __future__ import annotations

import logging
import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from pickle import UnpicklingError
from typing import Dict, List

import cv2
import dask
import dask.dataframe as dd
import imagesize
import pandas as pd
import yaml
from dask.diagnostics import ProgressBar
from datasetinsights.datasets.unity_perception import AnnotationDefinitions, Captures

from unityperception2yolo.utils import get_formatted_time

# See DatasetInsights documentation: https://datasetinsights.readthedocs.io/en/latest/

YOLO_IMG_FOLDER_NAME = "images"
YOLO_LABEL_FOLDER_NAME = "labels"


class Converter:
    """Converts Unity Perception format to YOLO format.

    Uses Unity's provided Dataset Insights package to perform dataset conversion
    from Unity Perception to YOLO. Datasets must be located at
    `{data_dir}/{split}/{dataset}`. Each split is a folder named accordingly
    (e.g. "test" or "val") containing multiple datasets. The original name of each
    dataset should be preserved (e.g. "6194df51-93d9-48f8-90af-05c6ef55ee69").

    Attributes:
        data_dir: `pathlib.Path` object pointing to directory of dataset splits.
        out_dir: `pathlib.Path` object to the output directory.
        split_paths: `pathlib.Path` object for each split.
        cache_paths: `pathlib.Path` object for each metadata cache.
        annotations: `datasetinsights.datasets.unity_perception.AnnotationDefinitions` for each split.
        labels: List of labels derived from annotations for each split.
        captures: `datasetinsights.datasets.unity_perception.Captures` for each split.

    Args:
        data_dir (str): Directory containing splits of datasets.
        out_dir (str): Directory to output to.
        split_names (List[str]): Names of folders containing splits. Defaults to `["train", "val"]`.
        cache_name (str): Path of cache relative to `{data_dir}/{split}`. Defaults to `cache.pkl`.
        check_cache (bool): Whether to check the cache first. Defaults to `True`.
        verbose (bool): Whether to show debug messages. Defaults to `False`.
        quiet (bool): Suppress messages. Defaults to `False`.
        convert_img (Union[str|bool]): Whether to convert to JPEG, leave as original, or some other format (i.e. `.webp`). Defaults to `False`.
    """

    def __init__(
        self,
        data_dir: str,
        out_dir: str,
        split_names: List[str] = ["train", "val"],
        cache_name: str = "cache.pkl",
        check_cache: bool = True,
        verbose: bool = False,
        quiet: bool = False,
        convert_img: bool = False,
    ):
        """Initializes Converter."""

        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)
        self.split_paths = {name: self.data_dir / name for name in split_names}
        self.cache_paths = {
            name: split / cache_name for name, split in self.split_paths.items()
        }
        self.annotations: Dict[str, AnnotationDefinitions] = {}
        self.captures: Dict[str, Captures] = {}
        self._check_cache = check_cache
        self._img_format = convert_img

        # logging config stuff
        self._log = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._log.setLevel(
            logging.DEBUG if verbose else logging.WARN if quiet else logging.INFO
        )

    @property
    def labels(self):
        if self.annotations is None:
            return None
        return {
            name: [
                x["label_name"]
                for x in sorted(anno_defs["spec"][0], key=lambda x: x["label_id"])
            ]
            for name, anno_defs in self.annotations.items()
        }

    def convert(self):
        """Perform dataset conversion."""

        self.out_dir.mkdir(exist_ok=True, parents=True)
        out_meta_path = self.out_dir / "dataset.yml"

        out_meta = dict(path=".")
        for split in self.split_paths:
            self.convert_split(split)
            out_meta[split] = f"{YOLO_IMG_FOLDER_NAME}/{split}"

        labels = list(self.labels.values())
        assert all(x == labels[0] for x in labels), "Incompatible datasets found!"

        out_meta["nc"] = len(labels[0])
        out_meta["names"] = labels[0]

        self._log.info(f"Saving metadata at {out_meta_path}...")
        with open(out_meta_path, "w") as f:
            yaml.safe_dump(out_meta, f)

    def convert_split(self, split: str):
        """Performs dataset conversion for one split folder."""

        data_path = self.split_paths[split]
        cache_path = self.cache_paths[split]

        self._log.info(f"Converting datasets at `{data_path}`...")
        self._log.debug(f"Expected cache path: `{cache_path}`")

        self._log.debug(f"1. Load metadata...")
        anno_defs = self._load_annotations(split)
        self._log.info(f"Labels: {self.labels[split]}")
        self._log.debug(f"1. Metadata loaded!")

        df: pd.DataFrame = None
        if cache_path.exists():
            expected_ids = (
                self._load_captures(split).captures["id"] if self._check_cache else None
            )
            df = self._read_cache(split, expected_ids=expected_ids)
        else:
            self._log.debug(f"Cache not found.")

        # no cache available
        if df is None:
            self._log.info("Transforming data...")
            df = pd.DataFrame(columns=("image_id", "filename", "annotations"))
            self._log.debug(f"2. Load data...")
            data = self._load_captures(split)
            self._log.debug(f"2. Data loaded!")

            self._log.debug(f"3. Transform data...")
            # unity unfortunately doesn't keep track of database name for img paths
            # so we have to rediscover them ourselves
            im_dirs: List[str] = []
            im_dirs += list(
                data.captures["filename"]
                .str.split("/", expand=True)[0]
                .drop_duplicates()
            )

            # (optional) filename to semantic/instance segmentation masks
            if "filename" in data.annotations:
                im_dirs += list(
                    data.annotations["filename"]
                    .dropna()
                    .str.split("/", expand=True)[0]
                    .drop_duplicates()
                )

            im_dir_map = {
                im_dir: next(data_path.glob(f"**/{im_dir}")) for im_dir in im_dirs
            }

            df["image_id"] = data.captures["id"]
            # use posix form for storing relative path for compatibility
            df["filename"] = (
                data.captures["filename"]
                .str.split("/")
                .apply(lambda p: (im_dir_map[p[0]] / p[1]).as_posix())
            )

            with ExitStack() as stack:
                stack.enter_context(dask.config.set(scheduler="processes"))
                if self._log.getEffectiveLevel() <= logging.INFO:
                    stack.enter_context(ProgressBar())

                ds = dd.from_pandas(data.captures, npartitions=os.cpu_count())
                df["annotations"] = ds.apply(
                    self._transform_yolo,
                    args=(anno_defs, self.labels[split], im_dir_map),
                    meta=pd.Series(dtype="object", name="annotations"),
                    axis=1,
                )
            self._log.debug(f"3. Data transformed!")

            if not cache_path.exists():
                df.to_pickle(cache_path)
                self._log.info(f"Cached to {cache_path}.")

        self._log.info("Saving annotations...")
        self._log.debug(f"4. Save annotations...")
        im_path = self.out_dir / YOLO_IMG_FOLDER_NAME / split
        label_path = self.out_dir / YOLO_LABEL_FOLDER_NAME / split
        im_path.mkdir(exist_ok=True, parents=True)
        label_path.mkdir(exist_ok=True, parents=True)
        with ExitStack() as stack:
            stack.enter_context(dask.config.set(scheduler="processes"))
            if self._log.getEffectiveLevel() <= logging.INFO:
                stack.enter_context(ProgressBar())

            ds = dd.from_pandas(df, npartitions=os.cpu_count())
            assert all(
                ds.apply(
                    self._dump_yolo,
                    args=(im_path, label_path),
                    meta=pd.Series(dtype="bool", name="None"),
                    axis=1,
                )
            ), "Not all annotations were saved successfully."
        self._log.debug(f"4. Annotations saved!")

    def _load_annotations(self, split: str) -> AnnotationDefinitions:
        if split in self.annotations:
            return self.annotations[split]

        anno_defs = AnnotationDefinitions(self.split_paths[split]).table
        anno_defs = anno_defs.reset_index(drop=True)

        assert len(anno_defs) > 0, "Datasets not found!"
        # TODO: implement more thorough metadata check
        assert (
            len(anno_defs["spec"].drop_duplicates()) == 1
        ), f"Incompatible datasets found!\n{anno_defs['spec']}"
        self.annotations[split] = anno_defs
        return anno_defs

    def _load_captures(self, split: str) -> Captures:
        if split in self.captures:
            return self.captures[split]

        self._log.info("Loading datasets. This could take a while...")
        data = Captures(self.split_paths[split])
        # Unity concats the dataframes, so reset index to prevent sorting problems
        data.captures = data.captures.reset_index(drop=True)
        data.annotations = data.annotations.reset_index(drop=True)
        self.captures[split] = data
        return data

    def _read_cache(self, split: str, expected_ids: pd.Series = None) -> pd.DataFrame:
        self._log.info("Reading from cache...")
        try:
            df: pd.DataFrame = pd.read_pickle(self.cache_paths[split])
        except UnpicklingError:
            self._log.warning("Cache is invalid! Ignoring...")
            new_path = self.cache_paths[split].with_stem(
                f"invalid_{get_formatted_time()}"
            )
            self.cache_paths[split].rename(new_path)
            return None

        # Comparing using .equals() is glitchy for some reason
        if not expected_ids is None and not all(
            df["image_id"].sort_values().eq(expected_ids.sort_values())
        ):
            self._log.warning("Cache is stale! Ignoring...")
            new_path = self.cache_paths[split].with_stem(
                f"stale_{get_formatted_time()}"
            )
            self.cache_paths[split].rename(new_path)
            return None

        return df

    def _transform_yolo(self, capture, anno_defs, label_map, im_dir_map):
        # capture: https://datasetinsights.readthedocs.io/en/latest/Synthetic_Dataset_Schema.html#captures
        # anno_defs: https://datasetinsights.readthedocs.io/en/latest/Synthetic_Dataset_Schema.html#annotation-definitions-json

        bboxes = []

        p = capture["filename"].split("/")
        im_w, im_h = imagesize.get((im_dir_map[p[0]] / p[1]))

        for anno in capture["annotations"]:
            anno_id = anno["annotation_definition"]
            anno_type = anno_defs.loc[anno_defs.id == anno_id, "name"].item()

            # YOLO only has bbox
            if anno_type != "bounding box":
                continue

            # for each annotated object
            for o in anno["values"]:
                ind = label_map.index(o["label_name"])
                x = o["x"]
                y = o["y"]
                w = o["width"]
                h = o["height"]
                cx = min(1.0, max(0.0, (x + 0.5 * w) / im_w))
                cy = min(1.0, max(0.0, (y + 0.5 * h) / im_h))
                bboxes.append(dict(i=ind, x=cx, y=cy, w=w / im_w, h=h / im_h))

        return bboxes

    def _dump_yolo(self, row, im_path: Path, label_path: Path):
        cur_path = Path(row["filename"])
        if not cur_path.exists():
            return False

        if not self._img_format:
            shutil.copy2(
                str(cur_path), str(im_path / f"{row['image_id']}{cur_path.suffix}")
            )
        else:
            im = cv2.imread(str(cur_path))
            if type(self._img_format) == bool:
                cv2.imwrite(str(im_path / f"{row['image_id']}.jpg"), im)
            else:
                cv2.imwrite(str(im_path / f"{row['image_id']}{self._img_format}"), im)

        with open(label_path / f"{row['image_id']}.txt", "w") as f:
            f.write(
                "\n".join(
                    f"{b['i']} {b['x']} {b['y']} {b['w']} {b['h']}"
                    for b in row["annotations"]
                )
            )
        return True
