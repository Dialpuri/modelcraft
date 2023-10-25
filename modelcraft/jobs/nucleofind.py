import dataclasses
import gemmi
from ..job import Job
from ..reflections import DataItem, write_mtz
from ..maps import read_map


@dataclasses.dataclass
class NucleoFindResult:
    predicted_map: gemmi.Ccp4Map
    seconds: float


class NucleoFind(Job):
    def __init__(self,
                 best_map: DataItem,
                 ):
        super().__init__("nucleofind")
        self.best_map = best_map

    def _setup(self) -> None:
        write_mtz(self._path("hklin.mtz"), [self.best_map])
        self._args += ["-m", "phos"]
        self._args += ["-i", "hklin.mtz"]
        self._args += ["-o", "predicted_phosphates.map"]
        self._args += ["-r", "3.0"]
        self._args += ["-intensity", self.best_map.label(0)]
        self._args += ["-phase", self.best_map.label(1)]

    def _result(self) -> NucleoFindResult:
        self._check_files_exist("predicted_phosphates.map")
        return NucleoFindResult(
            predicted_map=read_map(self._path("predicted_phosphates.map")),
            seconds=self._seconds
        )
