import subprocess as sbp

for i in range(1,11):
    print("###################### Starting for the fold number:",i)
    sbp.call("python3 bfrm_clf.py --num=%d"%i, shell=True)
    print("###################### Completed the fold:",i)
