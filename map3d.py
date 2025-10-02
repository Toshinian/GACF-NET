
from GACF.utils.evaluation import generate_kitti_3d_detection, evaluate_python
import os
label_path=r""
result_path=r""
label_split_file=r""

det_results, ret_dict = evaluate_python(label_path=label_path,
                                result_path=result_path,
                                label_split_file=label_split_file,
                                current_class=0,
                                metric='R40')
print ('\n' + det_results)
