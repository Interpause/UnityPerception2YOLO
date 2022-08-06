import logging

import dask
import fire
import pandas
from rich.console import Console
from rich.logging import RichHandler

console = Console()
logging.basicConfig(
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=console,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_suppress=[fire, dask, pandas],
        )
    ],
)
