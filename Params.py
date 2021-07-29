import sys
import importlib
import os.path as path
from types import MethodType, ModuleType

if len(sys.argv) <= 1:
    module_name = 'Default'
else:
    module_name = sys.argv[1]

print(f"Importing parameters from {module_name}")
Params = importlib.import_module('.' + module_name, package='experiments')
Params.params_file_path = path.join('experiments', module_name)

# Utility function. Checks if given property is set to True AND properties which are not defined are assumed false.
def isTrue(self, attrname):
    return hasattr(self,attrname) and getattr(self,attrname)
Params.isTrue = MethodType(isTrue, Params)
