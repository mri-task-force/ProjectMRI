import os

import settings


try:
    os.system('rm -r {}'.format(settings._DIR_trained_model))
except Exception as e:
    print(e)

try:
    os.system('rm -r {}'.format(settings._DIR_logs))
except Exception as e:
    print(e)

try:
    os.system('rm -r {}'.format(settings._DIR_patient_result))
except Exception as e:
    print(e)

try:
    os.system('rm -r {}'.format(settings._DIR_tblogs))
except Exception as e:
    print(e)