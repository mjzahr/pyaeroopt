
from pyaeroopt.test.naca0012.pyao.aerof_cls import AerofHdm, AerofRom, AerofGnat
from pyaeroopt.test.naca0012 import home_dir, aerof_exec

# AEROF class
aerof_hdm = [
  AerofHdm(bin=aerof_exec,
           desc=['Steady', 'Constant', None, None, None]),
  AerofHdm(bin=aerof_exec,
           desc=['Steady', 'Linear', None, None, None]),
  AerofHdm(bin=aerof_exec,
           desc=['Steady', 'ShapeOptimization', None, None, None]) ]
