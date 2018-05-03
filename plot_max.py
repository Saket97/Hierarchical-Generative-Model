import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
import pickle as pkl
#import mpl_toolkits.axes_grid1 as axes_grid1
#import seaborn as sns
plt.switch_backend("agg")
A = np.load("A1.npy")
#B = np.load("B1.npy")
D = np.load("delta_inv1.npy")
closs = np.load("clf_loss_list1.npy")
vtp = np.load("vtp_loss1.npy")

# Already taken mean while saving
#A = np.mean(A,axis=0)
#B = np.mean(B,axis=0)
#D = np.mean(D,axis=0)

plt.figure()
plt.plot(vtp)
plt.xlabel("Iterations")
plt.ylabel("-V($\phi $)")
plt.title("-ELBO")
plt.savefig("elbo.png")
plt.close()


plt.figure(figsize=(20,15))
plt.subplot(211)
if np.shape(A)[1] > 50:
    plt.boxplot(A[:,0:50])
    plt.subplot(212)
    plt.boxplot(A[:,50:])
    plt.title("A 50-99")
else:
    plt.boxplot(A)
plt.title("A 0-60")
plt.xlabel("Columns")
#plt.figure(figsize=(20,5))
plt.savefig("AboxPlot.png")
plt.close()

#plt.figure()
#plt.boxplot(B)
#plt.title("B")
#plt.xlabel("Columns")
#plt.savefig("BboxPlot.png")
#plt.close()

plt.figure()
plt.plot(closs[:,0])
plt.xlabel("Iterations")
plt.ylabel("Cross Entropy Loss")
plt.title("Classification Loss")
plt.savefig("closs.png")
plt.close()

#a = np.load("test_prob.npy")
#L_test = np.load("/opt/data/saket/gene_data/data/data_label.npy")
#L_test = L_test[160:]
#b = np.amax(a, axis=2)
#plt.figure()
#plt.subplot(211)
#plt.title("Probabilities for the max probability label in Test Set")
#plt.boxplot(b[:,0:26])
#plt.subplot(212)
#plt.boxplot(b[:,26:])
#plt.xlabel("Subjects")
#plt.xticks(np.arange(1,27),np.arange(27,53))
#plt.ylabel("Probabilities")
#plt.savefig("test_prob.png")
#plt.close()

test_lik = np.load("test_lik.npy")
plt.figure()
plt.plot(test_lik)
plt.xlabel("Iterations")
#plt.xticks(np.arange(0,60,5),1000*np.arange(0,60,5))
plt.ylabel("p(x|z,A,B,DELTA")
plt.title("Test Set lok likelihood")
plt.savefig("test_lik.png")
plt.close()

test_acc = np.load("test_acc.npy")
plt.figure()
plt.plot(test_acc)
plt.xlabel("Iterations")
plt.title("Test accuracy")
plt.savefig("test_acc.png")
plt.close()

M = np.load("M1.npy")
l = []
for i in range(M.shape[1]):
    count = 0
    for j in range(M.shape[0]):
        if M[j][i] < 0.1:
            count += 1
    l.append(count)
plt.figure()
plt.plot(l,'ro')
plt.xlabel("columns")
plt.ylabel("Number of zeros")
plt.title("M 0:20")
plt.savefig("M.png")
plt.close()
print("M:",M)
print("l:",l)

plt.figure()
plt.hist(M.flatten(),bins=30)
plt.savefig("M_hist.png")
plt.close()

#x = np.load("/opt/data/saket/gene_data/data/mod_total_data.npy")
#print(x.shape)
#x = np.power(10,x)
#plt.figure()
#plt.hist(x.flatten(),bins=1000)
#plt.savefig("x_count_hist.png")
#plt.close()
#count = 0
#count1 = 0
#count2 = 0
#count3 = 0
#for i in range(x.shape[0]):
#    for j in range(x.shape[1]):
#        if x[i][j] > 100000:
#            count += 1
#        if x[i][j] > 10000:
#            count1 += 1
#        if x[i][j] > 1000:
#            count2 += 1
#        if x[i][j] > 100:
#            count3 += 1
#print(">10^5:",count)
#print(">10^4:",count1)
#print(">10^3:",count2)
#print(">10^2:",count3)

auc = np.load("auc.npy")
plt.figure()
#plt.plot(auc[:,0],label="HIV(+) vs HIV(-)")
plt.plot(auc[:,0],label="SIRS vs SeS")
plt.plot(auc[:,1],label="SIRS vs Se")
plt.plot(auc[:,2],label="SeD vs SeS")
plt.legend()
plt.title("AUC (ROC)")
plt.xlabel("Iterations")
plt.ylabel("AUC")
plt.savefig("auc.png")
plt.close()

#x = np.load("/opt/data/saket/gene_data/data/mod_capsod_data.npy")
#plt.figure()
#plt.hist(x.flatten(),bins=15)
#plt.savefig("x_hist.png")
#plt.close()

Mtb = np.load("Mtb.npy")
Mactive = np.load("Mactive.npy")
Mlatent = np.load("Mlatent.npy")
#Mhiv = np.load("Mhiv.npy")
plt.figure()
plt.hist(Mactive)
plt.savefig("Mactive.png")
plt.close()
plt.figure()
plt.hist(Mlatent)
plt.savefig("Mlatent.png")
plt.close()
plt.figure()
plt.hist(Mtb)
plt.savefig("Mtb.png")
plt.close()
#plt.figure()
#plt.hist(Mhiv)
#plt.savefig("Mhiv.png")
#plt.close()
