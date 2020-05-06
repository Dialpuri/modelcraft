import gemmi
from .job import Job
from ..reflections import FsigF, FreeRFlag, write_mtz
from ..structure import write_mmcif


class Sheetbend(Job):
    def __init__(self, fsigf: FsigF, freer: FreeRFlag, structure: gemmi.Structure):
        super().__init__()
        args = []

        hklin = self.path("hklin.mtz")
        args += ["-mtzin", hklin]
        args += ["-colin-fo", fsigf.label()]
        args += ["-colin-free", freer.label()]
        write_mtz(hklin, [fsigf, freer])

        xyzin = self.path("xyzin.cif")
        args += ["-pdbin", xyzin]
        write_mmcif(xyzin, structure)

        args += ["-cycles", "12"]
        args += ["-resolution-by-cycle", "6,6,3"]

        xyzout = self.path("xyzout.cif")
        args += ["-pdbout", xyzout]

        self.run("csheetbend", args)

        self.structure = gemmi.read_structure(xyzout)

        self.finish()