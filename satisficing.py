import numpy as np
import scipy.stats as stats

# initializes Setting 6
mean = np.array([[4], [3], [2], [1]])
thrs = np.array([[2.5]])

# initializes Setting 7
# mean = [4, 3, 2, 1]
# mean = [[i, j, k] for i in mean for j in mean for k in mean]
# mean = np.array(mean)
# thrs = np.array([[2.5, 2.5, 2.5]])

A, D = mean.shape

K = 100 #10
T = 100000 #20000000
TT = 1 #200
reg = np.zeros((D, K, T//TT))

for k in range(K):
    print("k:", k)

    # initializes updates and estimates (each arm is played once)
    U = np.random.normal(mean)
    N = np.ones((A,1))

    # regret of the initialization
    for a in range(A):
        for d in range(D):
            if mean[a,d] < thrs[0,d]:
                reg[d,k,a//TT] += thrs[0,d] - mean[a,d]

    t = A
    while t < T:

        # arm selection
        diff = U - thrs
        cond = diff > -np.sqrt(4*np.log(N)/N)
        if cond.all(1).any():
            a = np.argwhere(cond.all(1))
            a = np.random.choice(a.flatten())

            # samples rewards, updates estimates and counters
            X = np.random.normal(mean)
            U[a] = (X[a] + N[a]*U[a]) / (N[a]+1)
            N[a] += 1

            # regret
            for d in range(D):
                if mean[a,d] < thrs[0,d]:
                    reg[d,k,t//TT] += thrs[0,d] - mean[a,d]

            t += 1

        else:
            #round-robin play
            for a in range(A):

                # samples rewards, updates estimates and counters
                X = np.random.normal(mean)
                U[a] = (X[a] + N[a]*U[a]) / (N[a]+1)
                N[a] += 1

                # regret
                for d in range(D):
                    if mean[a,d] < thrs[0,d]:
                        reg[d,k,t//TT] += thrs[0,d] - mean[a,d]

                t += 1

    np.save("res/sat.npy", [(A,D),mean,thrs,(K,T),reg])

### UCL IMPLEMENTATION
if D == 1:

    reg = np.zeros((1, K, T//TT))

    nppf = [0] * T
    for t in range(T):
        nppf[t] = stats.norm.ppf(1-1/(t+1))

    for k in range(K):
        print("k:", k)

        # initializes estimates
        U = np.zeros(A)
        N = np.zeros(A)
        for a in range(A):
            X = np.random.normal(mean)
            U[a] = (X[a] + N[a]*U[a]) / (N[a]+1)
            N[a] += 1

        # regret of the initialization
        for a in range(A):
            if mean[a,0] < thrs[0,0]:
                reg[0,k,a//TT] += thrs[0,0] - mean[a,0]

        for t in range(A,T):

            # arm selection
            Q = U + np.sqrt(1/N) * nppf[t]
            cond = Q >= thrs
            if cond.any():
                a = np.argwhere(cond)
                a = np.random.choice(a.flatten())
            else:
                a = np.argmax(Q)

            # samples rewards, updates estimates
            X = np.random.normal(mean)
            U[a] = (X[a] + N[a]*U[a]) / (N[a]+1)
            N[a] += 1

            # regret
            if mean[a,0] < thrs[0,0]:
                reg[0,k,t//TT] += thrs[0,0] - mean[a,0]

        np.save("res/sat-ucl.npy", [(A,1),mean,thrs,(K,T),reg])
