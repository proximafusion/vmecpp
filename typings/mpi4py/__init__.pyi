from typing import TypeVar

_T = TypeVar("_T")

class Comm:
    def Get_rank(self) -> int: ...
    def Get_size(self) -> int: ...
    def bcast(self, obj: _T, root: int = 0) -> _T: ...
    def gather(self, obj: _T, root: int = 0) -> list[_T]: ...

class _MPI:
    COMM_WORLD: Comm

MPI: _MPI
