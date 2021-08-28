import sys
import importlib

module = importlib.import_module('.' + sys.argv[2], package='eval')
EvalClass = getattr(module, sys.argv[2])

t = EvalClass()
t.eval()
