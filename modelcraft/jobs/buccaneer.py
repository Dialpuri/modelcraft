from typing import List, Optional, Union
import os
import gemmi
from ..contents import AsuContents, PolymerType
from .job import Job
from ..reflections import FsigF, FreeRFlag, ABCD, PhiFom, FPhi, write_mtz
from ..structure import write_mmcif


class Buccaneer(Job):
    def __init__(
        self,
        contents: AsuContents,
        fsigf: FsigF,
        freer: FreeRFlag,
        phases: Union[ABCD, PhiFom],
        fphi: Optional[FPhi] = None,
        input_structure: Optional[gemmi.Structure] = None,
        known_structure: Optional[List[str]] = None,
        mr_structure: Optional[gemmi.Structure] = None,
        use_mr: bool = True,
        filter_mr: bool = True,
        seed_mr: bool = True,
        cycles: int = 2,
        semet: bool = False,
        program: str = "cbuccaneer",
    ):
        super().__init__()
        args = []

        seqin = self.path("seqin.seq")
        args += ["-seqin", seqin]
        contents.write_sequence_file(seqin, PolymerType.PROTEIN)

        hklin = self.path("hklin.mtz")
        data_items = [fsigf, freer, phases]
        args += ["-mtzin", hklin]
        args += ["-colin-fo", fsigf.label()]
        args += ["-colin-free", freer.label()]
        if isinstance(phases, ABCD):
            args += ["-colin-hl", phases.label()]
        else:
            args += ["-colin-phifom", phases.label()]
        if fphi is not None:
            args += ["-colin-fc", fphi.label()]
            data_items.append(fphi)
        write_mtz(hklin, data_items)

        if input_structure is not None:
            xyzin = self.path("xyzin.cif")
            args += ["-pdbin", xyzin]
            args += ["-model-filter"]
            args += ["-model-filter-sigma", "1.0"]
            if known_structure is not None:
                for keyword in known_structure:
                    args += ["-known-structure", keyword]
            write_mmcif(xyzin, input_structure)

        if mr_structure is not None:
            xyzmr = self.path("xyzmr.cif")
            args += ["-pdbin-mr", xyzmr]
            if use_mr:
                args += ["mr-model"]
                if filter_mr:
                    args += ["mr-model-filter"]
                    args += ["mr-model-filter-sigma 2.0"]
                if seed_mr:
                    args += ["mr-model-seed"]
            write_mmcif(xyzmr, mr_structure)

        args += ["-cycles", str(cycles)]
        if semet:
            args += ["-build-semet"]
        args += ["-fast"]
        args += ["-correlation-mode"]
        args += ["-anisotropy-correction"]
        args += ["-resolution", "2.0"]

        xyzout = self.path("xyzout.cif")
        args += ["-pdbout", xyzout]
        args += ["-cif"]

        self.run(program, args)
        if not os.path.exists(xyzout):
            raise RuntimeError("Buccaneer did not produce an output structure")
        self.structure = gemmi.read_structure(xyzout)
        self.finish()