import os as _os

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
            return _os.path.abspath(_os.path.join(self._path_root, self._filename))
        
class DataFile(_FilePathAdjustableRoot):
    
    _path_root = "./Data"
    
class FigureFile(_FilePathAdjustableRoot):
    
    _path_root = "./Figures"
    
class ScratchFile(_FilePathAdjustableRoot):
    
    _path_root = "/scratch/cj59/RAPA/"
    
class SchematicFile(_FilePathAdjustableRoot):
    
    _path_root = "./Schematic"