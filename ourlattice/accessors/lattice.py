
from enum import Enum

class SupportFieldType(str, Enum):
    B = "#b"
    W = "#w"
    R = "#r"
    ID = "#id"
        
class Lattice:
    
    def __init__(self, obj):
        self._validate(obj)
        self._obj = obj
        
    @staticmethod
    def _validate(obj):
        raise NotImplementedError()