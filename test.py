#%%
from drivers import WFSManager, MlaRes

import numpy as np
import math
import matplotlib.pyplot as plt
# %%
wfs = WFSManager(MlaRes.Res512, exp_time=0.029)
wfs.initialize()

opt_exp_time, _ = wfs.optimize_exposure_time_and_gain()
if 0.022 < opt_exp_time < 80:
    wfs.exposure_time = opt_exp_time
    
opt_exp_time
# %%
wfs.take_image()
# %%
x,y = wfs.get_spot_deviation()
plt.imshow(x)
# %%
