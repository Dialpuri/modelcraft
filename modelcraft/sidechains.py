from modelcraft.coordinates import CoordinateFile
from modelcraft.job import Job
import os
import shutil


class Sidechains(Job):
    def __init__(self, directory, xyzin, hklin):
        super().__init__(directory)
        path = os.path.join(os.path.dirname(__file__), "coot", "prune.py")
        with open(path) as f:
            script = f.read()
        path = os.path.join(os.path.dirname(__file__), "coot", "sidechains.py")
        with open(path) as f:
            script += f.read()
        script += "fix_side_chains(0, 1, 2)\n"
        script += "write_pdb_file(0, '%s')\n" % self.path("xyzout.pdb")
        script += "exit()"
        with open(self.path("script.py"), "w") as f:
            f.write(script)
        arguments = [
            xyzin.path,
            hklin.path,
            "--no-graphics",
            "--no-guano",
            "--no-state-script",
            "--script",
            self.path("script.py"),
        ]
        self.run("coot", arguments)
        self.xyzout = CoordinateFile(self.path("xyzout.pdb"))
        shutil.rmtree("coot-backup", ignore_errors=True)
        shutil.rmtree("coot-download", ignore_errors=True)
