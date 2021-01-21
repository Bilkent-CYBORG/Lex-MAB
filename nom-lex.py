import numpy as np

# initializes Setting 1
mean = np.array([[0.50, 0.50], [0.50, 0.40], [0.40, 0.90]])

# initializes Setting 2
# mean = np.array([[0.50, 0.50], [0.50, 0.40], [0.40, 0.50]])

# initializes Setting 3
# mean = np.array([[0.50, 0.50], [0.50, 0.40], [0.40, 0.10]])

# initializes Setting 4
# mean = [0.90, 0.50, 0.40, 0.10]
# mean = [[i, j, k] for i in mean for j in mean for k in mean \
#     if (i<0.50) \
#     or (i==0.50 and j<0.50) \
#     or (i==0.50 and j==0.50 and k<0.50) \
#     or (i==0.50 and j==0.50 and k==0.50)]
# mean = np.array(mean)

# initializes Setting 5
# mean = [0.90, 0.50, 0.40, 0.10]
# mean = [[i, j, k] for i in mean for j in mean for k in mean \
#     if (i<0.50) \
#     or (i==0.50 and j<0.50) \
#     or (i==0.50 and j==0.50 and k<0.50) \
#     or (i==0.50 and j==0.50 and k==0.50)]
# mean = [[i, j, k] for [i, j, k] in mean if j >= 0.50]
# mean = np.array(mean)

mean_set = mean
A, D = mean.shape
_, D_set = mean_set.shape

# initializes NOM-LEX 1
eta = np.array([[0.45, 0.45]])

# initializes NOM-LEX 2
# eta = np.array([[0.40+1e-6, 0.40+1e-6]])

# initializes NOM-LEX 3
# eta = np.array([[0.50-1e-6, 0.50-1e-6]])

# initializes NOM-LEX 4
# eta = np.array([[0.45, 0.45, 0.45]])

# initializes NOM-LEX 5
# eta = np.array([[0.45, -1e6, 0.45]])

# converts to the single-objective case
# D = 1
# mean = mean[:,0,None]
# eta = eta[:,0,None]

K = 100
T = 100000
reg = np.zeros((D_set, K, T))

# set True for priority-free regrets
reg_alt = False

for k in range(K):
    print("k:", k)

    # initializes updates and estimates (each arm is played once)
    U = (np.random.uniform(size=(A,D)) <= mean).astype(float)
    N = np.ones((A,1))

    # regret of the initialization
    for a in range(A):
        for d in range(D_set):
            if mean_set[0,d] != mean_set[a,d]:
                reg[d,k,a] = mean_set[0,d] - mean_set[a,d]
                if not reg_alt:
                    break

    t = A
    while t < T:

        # arm selection
        diff = U - eta
        cond = diff > -np.sqrt(4*np.log(N)/N)
        if cond.all(1).any():
            a = np.argwhere(cond.all(1))
            a = np.random.choice(a.flatten())

            # samples rewards, updates estimates and counters
            X = (np.random.uniform(size=(1,D)) <= mean[a]).astype(float)
            U[a] = (X + N[a]*U[a]) / (N[a]+1)
            N[a] += 1

            # regret
            for d in range(D_set):
                if mean_set[0,d] != mean_set[a,d]:
                    reg[d,k,t] = mean_set[0,d] - mean_set[a,d]
                    if not reg_alt:
                        break

            t += 1

        else:
            # round-robin play
            for a in range(A):

                # samples rewards, updates estimates and counters
                X = (np.random.uniform(size=(1,D)) <= mean[a]).astype(float)
                U[a] = (X + N[a]*U[a]) / (N[a]+1)
                N[a] += 1

                # regret
                for d in range(D_set):
                    if mean_set[0,d] != mean_set[a,d]:
                        reg[d,k,t] = mean_set[0,d] - mean_set[a,d]
                        if not reg_alt:
                            break

                t += 1

np.save("res/nom-lex.npy", [(A,D),mean,eta,(K,T),reg])
