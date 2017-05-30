
from pyaeroopt.test.naca0012.pyao import frg, hpc
from pyaeroopt.test.naca0012.pyao.factory import aerof_hdm

#db = initialize_database('out/aerof_hdm.db')
#for x in aerof_hdm:
#    x.set_database(db)

aerof_hdm = [x.set_mach(0.5).set_alpha(0.0).set_beta(0.0) for x in aerof_hdm]
for x in aerof_hdm:
    x.execute(None, [None], hpc)
