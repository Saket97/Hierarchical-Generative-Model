import subprocess as sbp

for i in range(1,2):
    print("###################### Starting for the fold number:",i)                                                                                                                          
    sbp.call("python3 maxLik_L1.py --logdir=/opt/data/saket/gene_data/model --num=%d"%i, shell=True)
    print("###################### Completed the fold:",i)
