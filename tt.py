import os
import pandas as pd
import numpy as np
from utils import histogram_plot

def threshold_serch(similarities, freq=100):
    best_differ, best_th = 1000000, 0.0
    th_array = np.linspace(0, 0.5, 10000)
    
    for i, th in enumerate(th_array):
        sp = similarities[similarities >= th]
        sn = similarities[similarities < th]
        differ = abs(len(sp) - len(sn))
       
        if differ < best_differ:
            best_differ = differ
            best_th = th

        if (i+1) % freq == 0 or (i+1) == len(th_array):
            print('Progress {}/{}'.format((i+1),len(th_array)), best_differ, best_th)
    return best_differ, best_th
        


dir1 = 'work_dir/graphcodebert-base.arcface.upg_ff_fw.batch256_epoch_20_s64_wisk1.0_sp_sn_m0.5_class2_sampler_re2_dropout0.1'
dir2 = 'work_dir/graphcodebert-base.arcface.upg_ff_fw.batch256_epoch_20_s64_wisk1.0_sp_sn_m0.5_class2_sampler_re2_dropout0.1_reverse_tokens'
submission1 = pd.read_csv(os.path.join(dir1, 'submission_raw2.csv'))
similarities1 = np.array(submission1['similar'])

submission2 = pd.read_csv(os.path.join(dir2, 'submission_raw2.csv'))
similarities2 = np.array(submission2['similar'])

similarities = (similarities1 + similarities2) / 2.0
# sp = similarities[similarities >= th]
# sn = similarities[similarities < th]

# print(len(sp), len(sn), len(similarities))
# print(len(sp) / len(similarities), len(sn) / len(similarities))

print(similarities.min())

best_differ, best_th = threshold_serch(similarities)
print(best_differ, best_th)
histogram_plot(similarities, 1000, save_name='hist2.png')

# th = 0.25
similarities = (np.array(similarities) >= best_th).astype(int)
submission1['similar'] = similarities
# submission1.to_csv(os.path.join(dir1, 'submission{}.csv'.format(th)), index=False)
submission1.to_csv('submission{}.csv'.format(best_th), index=False)
# print(os.path.join(dir1, 'submission{}.csv'.format(th)))