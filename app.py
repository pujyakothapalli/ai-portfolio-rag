import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import runpy
runpy.run_path("src/ui/app.py", run_name="__main__")