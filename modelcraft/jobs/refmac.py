from typing import Optional, Union
import xml.etree.ElementTree as ET
import gemmi
from ..reflections import FsigF, FreeRFlag, ABCD, PhiFom, write_mtz
from ..structure import write_mmcif
from .job import Job


class Refmac(Job):
    def __init__(
        self,
        structure: gemmi.Structure,
        fsigf: FsigF,
        freer: FreeRFlag,
        cycles: int,
        phases: Optional[Union[ABCD, PhiFom]] = None,
        twinned: bool = False,
    ):
        super().__init__()

        hklin = self.path("hklin.mtz")
        xyzin = self.path("xyzin.cif")
        hklout = self.path("hklout.mtz")
        xyzout = self.path("xyzout.cif")
        xmlout = self.path("xmlout.xml")

        write_mmcif(xyzin, structure)
        write_mtz(hklin, [fsigf, freer, phases])

        args = []
        args += ["HKLIN", hklin]
        args += ["XYZIN", xyzin]
        args += ["HKLOUT", hklout]
        args += ["XYZOUT", xyzout]
        args += ["XMLOUT", xmlout]

        stdin = []
        labin = "FP=" + fsigf.label(0)
        labin += " SIGFP=" + fsigf.label(1)
        labin += " FREE=" + freer.label()
        if phases is not None:
            if isinstance(phases, ABCD):
                labin += " HLA=" + phases.label(0)
                labin += " HLB=" + phases.label(1)
                labin += " HLC=" + phases.label(2)
                labin += " HLD=" + phases.label(3)
            else:
                labin += " PHIB=" + phases.label(0)
                labin += " FOM=" + phases.label(1)
        stdin.append("LABIN " + labin)
        stdin.append("NCYCLES %d" % cycles)
        stdin.append("MAKE HYDR NO")
        if twinned:
            stdin.append("TWIN")
        stdin.append("MAKE NEWLIGAND NOEXIT")
        stdin.append("PHOUT")
        stdin.append("PNAME modelcraft")
        stdin.append("DNAME modelcraft")
        stdin.append("END")

        self.run("refmac5", args, stdin)

        self._set_rfactors(xmlout)

        self.structure = gemmi.read_structure(xyzout)
        mtz = gemmi.read_mtz_file(hklout)
        self.fsigf = FsigF(mtz, fsigf.label())
        self.abcd = ABCD(mtz, "HLACOMB,HLBCOMB,HLCCOMB,HLDCOMB")
        self.fphi_best = PhiFom(mtz, "FWT,PHWT")
        self.fphi_diff = PhiFom(mtz, "DELFWT,PHDELWT")
        self.fphi_calc = PhiFom(mtz, "FC_ALL,PHIC_ALL")

        self.finish()

    def _set_rfactors(self, path: str) -> None:
        xml = ET.parse(path).getroot()
        rworks = list(xml.iter("r_factor"))
        rfrees = list(xml.iter("r_free"))
        self.initial_rwork = float(rworks[0].text) * 100
        self.initial_rfree = float(rfrees[0].text) * 100
        self.final_rwork = float(rworks[-1].text) * 100
        self.final_rfree = float(rfrees[-1].text) * 100