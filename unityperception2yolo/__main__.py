import logging
import multiprocessing

import fire

from unityperception2yolo.console import console
from unityperception2yolo.convert import Converter

log = logging.getLogger(__name__)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        console.rule("Unity Perception 2 YOLO Converter")
        fire.Fire(Converter)
        log.warning(
            "NOTE: `dataset.yml` has to be manually modified using a text editor to have the correct path (relative to YOLO's root directory) after moving the dataset."
        )
    except Exception as e:
        log.exception(e)
