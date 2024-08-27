import logging
import os
import shutil
import unicodedata
from pathlib import Path
from typing import List, Union

# import wget


logger = logging.getLogger(Path(__file__).name)
logger.setLevel(logging.INFO)

LEXSUB_DATASETS_URL = "https://github.com/stephenroller/naacl2016/archive/master.zip"


# def download_dataset(url: str, dataset_path: str):
#     """
#     Method for downloading dataset from a given URL link.
#     After download dataset will be saved in the dataset_path directory.

#     Args:
#         url: URL link to dataset.
#         dataset_path: Directory path to save the downloaded dataset.

#     Returns:

#     """
#     os.makedirs(dataset_path, exist_ok=True)
#     logger.info(f"Downloading file from '{url}'...")
#     filename = wget.download(url, out=str(dataset_path))
#     logger.info(f"File {filename} is downloaded to '{dataset_path}'.")
#     filename = Path(filename)

#     # Extract archive if needed
#     extract_archive(arch_path=filename, dest=dataset_path)

#     # Delete archive
#     if os.path.isfile(filename):
#         os.remove(filename)
#     elif os.path.isdir(filename):
#         shutil.rmtree(filename)


def strip_accents(s: str) -> str:
    """
    Remove accents from given string:
    Example: strip_accents("Málaga") -> Malaga

    Args:
        s: str - string to process
    Returns:
        string without accents
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def split_line(line: str, sep: str = " ") -> List[str]:
    """
    Method for splitting line by given separator 'sep'.

    Args:
        line: Input line to split.
        sep: Separator char.
    Returns:
        line: List of parts of the input line.
    """
    line = [part.strip() for part in line.split(sep=sep)]
    return line


def extract_archive(arch_path: Union[str, Path], dest: str):
    """
    Extracts archive into a given folder.

    Args:
        arch_path: path to archive file.
            Could be given as string or `pathlib.Path`.
        dest: path to destination folder.
    """
    arch_path = Path(arch_path)
    file_suffixes = arch_path.suffixes
    outer_suffix = file_suffixes[-1]
    inner_suffix = file_suffixes[-2] if len(file_suffixes) > 1 else ""
    if outer_suffix == ".zip":
        with zipfile.ZipFile(arch_path, "r") as fp:
            fp.extractall(path=dest)
    elif outer_suffix == ".tgz" or (outer_suffix == ".gz" and inner_suffix == ".tar"):
        with tarfile.open(arch_path, "r:gz") as tar:
            dirs = [member for member in tar.getmembers()]
            tar.extractall(path=dest, members=dirs)
    elif outer_suffix == ".gz":
        with gzip.open(arch_path, "rb") as gz:
            with open(
                arch_path.parent / (arch_path.stem + inner_suffix), "wb"
            ) as uncomp:
                shutil.copyfileobj(gz, uncomp)