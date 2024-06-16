import os as _os
import argparse as _argparse
import sys


if "/share/apps/python/3.8.6/intel/lib/python3.8" in sys.path and sys.path[1] != '/home/cj59/.local/lib/python3.8/site-packages':
    sys.path.insert(1, '/home/cj59/.local/lib/python3.8/site-packages')


class _FilePathAdjustableRoot:
    _path_root = "."

    @classmethod
    def set_path(cls, path):
        cls._path_root = _os.path.expanduser(path)

    def __init__(self, filename):
        self._filename = _os.path.expanduser(filename)

    def __str__(self):
        if _os.path.isabs(self._filename):
            return self._filename
        else:
            return _os.path.abspath(
                _os.path.join(self._path_root, self._filename)
            )


class DataFile(_FilePathAdjustableRoot):
    _path_root = "./Data"


class FigureFile(_FilePathAdjustableRoot):
    _path_root = "./Figures"


class ScratchFile(_FilePathAdjustableRoot):
    _path_root = "/scratch/cj59/RAPA/"


class SchematicFile(_FilePathAdjustableRoot):
    _path_root = "./Schematic"


class ModelFile(_FilePathAdjustableRoot):
    _path_root = "./Models"


def parse_file_path_command_line():
    ap = _argparse.ArgumentParser(description="JTB Figure Pipeline")

    ap.add_argument(
        "-f", "-F",
        dest="figure_dir",
        help="Figure Path",
        metavar="PATH",
        default=None
    )

    ap.add_argument(
        "-d", "-D",
        dest="data_dir",
        help="Data Path",
        metavar="PATH",
        default=None
    )

    ap.add_argument(
        "-scratch",
        "-SCRATCH",
        dest="scratch_dir",
        help="Scratch Path",
        metavar="PATH",
        default=None,
    )

    ap.add_argument(
        "-s",
        "-S",
        dest="schematic_dir",
        help="Schematic Path",
        metavar="PATH",
        default=None,
    )

    ap.add_argument(
        "-m", "-M",
        dest="model_dir",
        help="Model Path",
        metavar="PATH",
        default=None
    )

    args, _ = ap.parse_known_args()

    if args.figure_dir is not None:
        FigureFile.set_path(args.figure_dir)

    if args.data_dir is not None:
        DataFile.set_path(args.data_dir)

    if args.scratch_dir is not None:
        ScratchFile.set_path(args.scratch_dir)

    if args.schematic_dir is not None:
        SchematicFile.set_path(args.schematic_dir)

    if args.model_dir is not None:
        ModelFile.set_path(args.model_dir)

    return args
