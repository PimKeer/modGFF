import numpy as np
from matplotlib import pyplot as plt

# The most of the plots in the thesis were made here.

# BERNOULLI: Nk = 10000, Narr = [4,8,16,32,64], parr = [0.55,0.56,0.57,0.58,0.59,0.60]
# Nk = 10000 # Amount of samples made for the average of gamma.
# Narr = np.array([4,8,16,32,64]) # N's to use.
# parr = np.array([0.55,0.56,0.57,0.58,0.59,0.60]) # Cuts we try
# Nh = len(parr) # Amount of cuts h we try.
#
# R = [np.array([10.52165569, 10.79816344, 11.02177382, 11.40513737, 11.75259497, 12.03953147])]
# R.append(np.array([9.80390147, 10.54181822, 11.03252552, 11.65929797, 12.15509036, 12.86487864]))
# R.append(np.array([8.71084153, 9.62645873, 10.47795988, 11.37841274, 12.60346446, 13.61675366]))
# R.append(np.array([7.24929179, 8.25772302, 9.58586203, 10.9319688, 12.88692155, 14.43525915]))
# R.append(np.array([5.66553281, 6.35530585, 7.80218117, 9.82827623, 12.65738267, 15.62710257]))
#
# E = [np.array([0.28100089, 0.24206031, 0.19752345, 0.33725682, 0.32491211, 0.32271224])]
# E.append(np.array([0.27603494, 0.25055059, 0.35357458, 0.31942171, 0.21577639, 0.19277705]))
# E.append(np.array([0.16537669, 0.23316627, 0.23477479, 0.18667965, 0.43470381, 0.28769129]))
# E.append(np.array([0.23299286, 0.15791748, 0.24758114, 0.36136118, 0.25633, 0.26350624]))
# E.append(np.array([0.11131287, 0.11912703, 0.14214923, 0.39367185, 0.20374559, 0.36777518]))

# BERNOULLI: Nk = 10000, Narr = [4,8,16,32,64], parr = [0.585,0.59,0.595,0.60]
# Nk = 10000 # Amount of samples made for the average of gamma.
# Narr = np.array([4,8,16,32,64]) # N's to use.
# parr = np.array([0.585,0.59,0.595,0.60]) # Cuts we try
# Nh = len(parr) # Amount of cuts h we try.
#
# R = [np.array([11.62541972, 11.88721889, 12.05449164, 11.83858186])]
# R.append(np.array([11.79711957, 12.23952533, 12.50382139, 12.80913765]))
# R.append(np.array([11.98221715, 12.45962321, 13.13717292, 13.39640561]))
# R.append(np.array([11.90929436, 12.72470214, 13.70753318, 14.40981056]))
# R.append(np.array([11.18083376, 12.65738267, 14.11207353, 15.62710257]))
#
# E = [np.array([0.38131414, 0.38198198, 0.40438002, 0.31662146])]
# E.append(np.array([0.33971207, 0.29197755, 0.28729844, 0.27017602]))
# E.append(np.array([0.3365562, 0.30643767, 0.36607269, 0.24235462]))
# E.append(np.array([0.17658106, 0.32163449, 0.26023736, 0.26549843]))
# E.append(np.array([0.269744, 0.20374559, 0.35870609, 0.36777518]))

# SHEFFIELD: Nk = 5000, Narr = [10,20,40], parr = [0.13,0.135,0.14,0.145,0.15,0.155,0.16]
# Nk = 5000 # Amount of samples made for the average of gamma.
# Narr = np.array([10,20,40]) # N's to use.
# parr = np.array([0.13,0.135,0.14,0.145,0.15,0.155,0.16]) # Cuts we try
# Nh = len(parr) # Amount of cuts h we try.
#
# R = [np.array([20.64388545,21.1511516,22.59138739,23.37893987,24.8844019,25.10520398,27.05304619])]
# R.append(np.array([20.21384727,21.18986604,22.20661928,23.12598579,25.320884,27.1979515,28.62694308]))
# R.append(np.array([18.44732418,19.64954528,21.87738331,23.88929185,25.17581676,28.04525445,29.98856354]))
#
# E = [np.array([0.51791915,0.70315389,0.77278936,1.30246325,1.0829812,0.73156913,1.11234073])]
# E.append(np.array([0.52736901,0.91623963,0.77008539,0.71437123,0.8516745,0.88460973,0.9024745]))
# E.append(np.array([0.5899398,0.67496123,0.58256453,1.24159805,1.27367734,0.73931034,1.19331514]))

# CHAFAI: Nk = 10000, Narr = [5,10], parr = [0.07,0.08,0.09,0.10,0.11,0.12,0.13]
Nk = 10000 # Amount of samples made for the average of gamma.
Narr = np.array([5,10]) # N's to use.
parr = np.array([0.07,0.08,0.09,0.10,0.11,0.12,0.13]) # Cuts we try
Nh = len(parr) # Amount of cuts h we try.

R = [np.array([13.0403329,  14.34139121, 13.51491848, 15.15065728, 16.46203714, 18.03731747, 17.29526863])]
R.append(np.array([12.56989869, 13.37190201, 14.25082112, 15.53510871, 16.4649759, 17.84960198, 19.34041402]))

E = [np.array([0.24666015, 0.36365816, 0.34195492, 0.30722813, 0.46461569, 0.4985514, 0.45354863])]
E.append(np.array([0.26914328, 0.33109815, 0.27279675, 0.36641951, 0.45401415, 0.50682696, 0.83331686]))

# CG: e = 1e-70,1e-80, Nk = 5000, Narr = [5,10], parr = [0.07,0.08,0.09,0.10,0.11,0.12,0.13]
# Nk = 5000 # Amount of samples made for the average of gamma.
# Narr = np.array([5,10]) # N's to use.
# parr = np.array([0.07,0.08,0.09,0.10,0.11,0.12,0.13]) # Cuts we try
# Nh = len(parr) # Amount of cuts h we try.
#
# R = [np.array([12.36035755, 13.68787072,12.90630165,14.5842637,15.59143965,17.33773761,17.39893222])]
# R.append(np.array([11.64409947,12.69812603,14.00211495,14.85741072,16.75515689,18.08623824,19.95481881]))
#
# E = [np.array([0.45653549,0.64265026,0.42038999,0.46664066,0.68584527,0.63290613,0.8369731])]
# E.append(np.array([0.62934975,1.40844379,1.22724212,1.44208512,1.56760868,1.18923711,1.62955433]))

# CG: e = 1e-70,1e-90,1e-110 Nk = 5000, Narr = [5], parr = [0.07,0.08,0.09,0.10,0.11,0.12,0.13]
# Nk = 5000 # Amount of samples made for the average of gamma.
# Narr = np.array([5]) # N's to use.
# parr = np.array([0.07,0.08,0.09,0.10,0.11,0.12,0.13]) # Cuts we try
# Nh = len(parr) # Amount of cuts h we try.
# earr = ['$10^{-70}$','$10^{-90}$','$10^{-110}$']
#
# R = [np.array([12.36035755,13.68787072,12.90630165,14.5842637,15.59143965,17.33773761,17.39893222])]
# R.append(np.array([13.99247821,15.65110923,15.18675192,16.9292063,17.70691953,20.39172161,20.11134935]))
# R.append(np.array([15.50803683,17.32721945,16.87932196,18.52191975,20.14299199,22.83359233,22.45835082]))
#
# E = [np.array([0.45653549,0.64265026,0.42038999,0.46664066,0.68584527,0.63290613,0.8369731])]
# E.append(np.array([0.44078261,0.72971737,0.67835925,0.76992978,0.93263141,1.08189858,1.05622572]))
# E.append(np.array([0.61718292,0.56051415,0.95827251,0.85396261,0.95980132,1.27452245,1.10596057]))

# CG: e = 1e-70,1e-90,1e-110 Nk = 5000, Narr = [5], parr = [0.07,0.08,0.09,0.10,0.11,0.12,0.13]
# Nk = 5000 # Amount of samples made for the average of gamma.
# Narr = np.array([10]) # N's to use.
# parr = np.array([0.07,0.08,0.09,0.10,0.11,0.12,0.13]) # Cuts we try
# Nh = len(parr) # Amount of cuts h we try.
# earr = ['$10^{-70}$','$10^{-90}$','$10^{-110}$']
#
# R = [np.array([10.05656199,10.72698078,11.77154349,12.61214524,13.72407825,15.04868971,16.65883322])]
# R.append(np.array([11.64409947,12.69812603,14.00211495,14.85741072,16.75515689,18.08623824,19.95481881]))
# R.append(np.array([13.20012578,15.0594913,15.99327804,17.18466664,19.38347609,20.99524816,22.61862151]))
#
# E = [np.array([0.51607336,0.85288986,0.79946905,0.85402838,1.14772341,1.12351745,0.86350222])]
# E.append(np.array([0.62934975,1.40844379,1.22724212,1.44208512,1.56760868,1.18923711,1.62955433]))
# E.append(np.array([0.76293282,1.45395872,1.32623659,1.72965105,2.02499512,1.71413599,1.2036772]))
#
# col = ['r', 'g', 'b', 'y', 'k']
# line = ['-', '--', '-.', ':', '-']
#
# size = 14
#
# plt.rc('font', size=size)
# plt.rc('axes', titlesize=size)
# plt.rc('axes', labelsize=size)
# plt.rc('xtick', labelsize=size)
# plt.rc('ytick', labelsize=size)
# plt.rc('legend', fontsize=size)
# plt.rc('figure', titlesize=size)
#
# for i in range(len(earr)):
#     qm = np.poly1d(np.polyfit(parr, R[i], 2))
#     x = np.linspace(parr[0], parr[-1], 10000)
#     plt.plot(x,qm(x), linestyle=line[i], color=col[i], label="$\epsilon$ = "+str(earr[i]))
#     plt.errorbar(parr,R[i],yerr=E[i],capsize=5,linestyle='None',ecolor=col[i])
#     # plt.xticks([0.585,0.59,0.595,0.60])
#     plt.xlabel("$p$")
#     plt.ylabel("$R_N$")
#     plt.legend()
# plt.show()


# CG VS CHAFAI: e = 1e-70, Nk = 5000 vs 10000, Narr = [5], parr = [0.07,0.08,0.09,0.10,0.11,0.12,0.13]
# R = [np.array([13.0403329,  14.34139121, 13.51491848, 15.15065728, 16.46203714, 18.03731747, 17.29526863])]
# R.append(np.array([12.36035755, 13.68787072,12.90630165,14.5842637,15.59143965,17.33773761,17.39893222]))
#
# E = [np.array([0.24666015, 0.36365816, 0.34195492, 0.30722813, 0.46461569, 0.4985514, 0.45354863])]
# E.append(np.array([0.45653549,0.64265026,0.42038999,0.46664066,0.68584527,0.63290613,0.8369731]))

# CG VS CHAFAI: e = 1e-80, Nk = 5000 vs 10000, Narr = [10], parr = [0.07,0.08,0.09,0.10,0.11,0.12,0.13]
# R = [np.array([12.56989869, 13.37190201, 14.25082112, 15.53510871, 16.4649759, 17.84960198, 19.34041402])]
# R.append(np.array([11.64409947,12.69812603,14.00211495,14.85741072,16.75515689,18.08623824,19.95481881]))
#
# E = [np.array([0.26914328, 0.33109815, 0.27279675, 0.36641951, 0.45401415, 0.50682696, 0.83331686])]
# E.append(np.array([0.62934975,1.40844379,1.22724212,1.44208512,1.56760868,1.18923711,1.62955433]))
#
#
# col = ['r', 'g', 'b', 'y', 'k']
# line = ['-', '--', '-.', ':', '-']
#
# size = 14
#
# plt.rc('font', size=size)
# plt.rc('axes', titlesize=size)
# plt.rc('axes', labelsize=size)
# plt.rc('xtick', labelsize=size)
# plt.rc('ytick', labelsize=size)
# plt.rc('legend', fontsize=size)
# plt.rc('figure', titlesize=size)
#
# parr = np.array([0.07,0.08,0.09,0.10,0.11,0.12,0.13])
# qm1 = np.poly1d(np.polyfit(parr, R[0], 2))
# qm2 = np.poly1d(np.polyfit(parr, R[1], 2))
# x = np.linspace(parr[0], parr[-1], 10000)
# # plt.plot(x,qm1(x), linestyle=line[0], color=col[0], label="Chafai")
# # plt.plot(x,qm2(x), linestyle=line[1], color=col[1], label="CG")
# plt.errorbar(parr,R[0],yerr=E[0],capsize=5,linestyle='None',ecolor=col[0], label="Chafaï")
# plt.errorbar(parr,R[1],yerr=E[1],capsize=5,linestyle='None',ecolor=col[1], label="CG")
# # plt.xticks([0.585,0.59,0.595,0.60])
# plt.xlabel("$p$")
# plt.ylabel("$R_N$")
# plt.legend()
# plt.show()


# CHECKERBOARD: e = 1e-90, Nk = 500, Narr = [10], parr = [0.05,0.075,0.1,0.125,0.15,0.175,0.2]
# Nk = 500 # Amount of samples made for the average of gamma.
# Narr = np.array([10]) # N's to use.
# parr = np.array([0.05,0.075,0.1,0.125,0.15,0.175,0.2]) # Cuts we try
# Nh = len(parr) # Amount of cuts h we try.
# carr = np.array([0.5,0.6,0.7,0.8,0.9,1.0])
#
# R = [np.array([12.32040179,15.5215578,21.31846093,25.09552846,28.41863112,33.26086535,36.8618765])]
# R.append(np.array([15.98565244,20.44331841,27.21029633,28.5084934,33.647455,38.42773148,38.96408807]))
# R.append(np.array([14.52863555,16.65290282,22.19161463,26.37787317,30.01223466,36.39202084,41.25517907]))
# R.append(np.array([12.40186752,17.1720763,18.21859256,23.05259171,28.73752452,31.72758961,38.29380806]))
# R.append(np.array([11.39964714,11.77757169,18.20512079,20.06155321,25.63913414,27.67127204,33.07417979]))
# R.append(np.array([10.46247742,12.75824617,16.92825122,19.18135881,24.24077954,27.32113228,30.63275556]))
# R.append(np.array([9.39732325,10.05587753,15.10488867,15.84222418,20.99547092,24.30651597,29.70878918]))
#
# E = [np.array([1.34451805,2.33695534,4.10346812,2.63864411,4.28264193,2.6463801,4.0122902])]
# E.append(np.array([2.25233402,4.25235405,6.83269201,4.37933598,5.58159638,4.31056046,5.81562647]))
# E.append(np.array([3.16877998,2.55245026,4.85565131,3.67845442,3.61224869,4.07303236,4.28365608]))
# E.append(np.array([2.05513768,5.13723935,3.29101326,3.69591941,5.79152403,3.50894212,3.9067064]))
# E.append(np.array([1.43574699,2.06192596,3.98719148,2.16811126,3.71976396,3.84268508,4.8554143]))
# E.append(np.array([1.22886764,2.1919939,3.39114915,2.90677832,2.9114883,6.22231869,4.53961725]))
# E.append(np.array([1.20599309,1.43731497,2.98378162,2.29674982,2.37286698,2.20189871,4.38027433]))

Nk = 500 # Amount of samples made for the average of gamma.
Narr = np.array([10]) # N's to use.
parr = np.array([0.05,0.075,0.1,0.125,0.15,0.175,0.2]) # Cuts we try
Nh = len(parr) # Amount of cuts h we try.
carr = np.array([0.65])

R = [np.array([12.32040179,15.5215578,21.31846093,25.09552846,28.41863112,33.26086535,36.8618765])]
R.append(np.array([12.12046458,16.00107176,20.94014584,24.10393164,27.00438787,33.73017425,34.38041728]))

E = [np.array([1.34451805,2.33695534,4.10346812,2.63864411,4.28264193,2.6463801,4.0122902])]
E.append(np.array([1.63078717,2.87542076,3.80210757,2.92841593,4.20682798,3.09440351,3.92178457]))


col = ['r', 'g', 'b', 'y', 'k', 'm', 'c', 'pink']
line = ['-', '--', '-.', ':', '-', '--', '-.', ':']

size = 14

plt.rc('font', size=size)
plt.rc('axes', titlesize=size)
plt.rc('axes', labelsize=size)
plt.rc('xtick', labelsize=size)
plt.rc('ytick', labelsize=size)
plt.rc('legend', fontsize=size)
plt.rc('figure', titlesize=size)

qm = np.poly1d(np.polyfit(parr, R[0], 2))
x = np.linspace(parr[0], parr[-1], 10000)
plt.plot(x,qm(x), linestyle=line[0], color=col[0], label="Checkerboard")
plt.errorbar(parr,R[0],yerr=E[0],capsize=5,linestyle='None',ecolor=col[0])
for i in range(0,len(carr)):
    qm = np.poly1d(np.polyfit(parr, R[i+1], 2))
    x = np.linspace(parr[0], parr[-1], 10000)
    plt.plot(x,qm(x), linestyle=line[i+1], color=col[i+1], label="$c$ = "+str(carr[i]))
    plt.errorbar(parr,R[i+1],yerr=E[i+1],capsize=5,linestyle='None',ecolor=col[i+1])
    # plt.xticks([0.585,0.59,0.595,0.60])
    plt.xlabel("$p$")
    plt.ylabel("$R_N$")
    plt.legend()
plt.show()
