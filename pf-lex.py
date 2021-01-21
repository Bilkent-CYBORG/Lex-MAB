import numpy as np
import sys

uid = sys.argv[1]

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

A, D = mean.shape

K = 100 #10
T = 100000 #500000000
TT = 1 #1000
reg = np.zeros((D, K, T//TT))

# initializes PF-LEX 1
dlt = T ** (-0.20)
eps = T ** (-0.20)

# initializes PF-LEX 2
# dlt = T ** (-0.10)
# eps = T ** (-0.10)

# initializes PF-LEX 3
# dlt = T ** (-0.33)
# eps = T ** (-0.33)

for k in range(K):
    print('k:', k)

    # initializes estimates and counters
    M = np.zeros((A,D))
    N = np.zeros((A,1))

    linked = np.ones((D,A,A))
    chained = np.ones((D,A,A))
    a = 0 # dummy arm for round 1

    for t in range(T):
        print('k:', k, 't:', t)

        # calculates confidence intervals
        C = np.sqrt((1+N)/(N**2) * (1+2*np.log((A*D*np.sqrt(1+N))/dlt)))
        U = M + C
        L = M - C

        # updates 'linked' relations of arm a
        # linked[i,a1,a2]==1 iff a1 links to a2 in objective i
        linked_n = np.copy(linked)
        for i in range(D):
            for a1 in range(A):
                linked_n[i,a,a1] = (U[a,i]>=L[a1,i] and L[a,i]<=U[a1,i]) \
                    or (U[a1,i]>=L[a,i] and L[a1,i]<=U[a,i])
                linked_n[i,a1,a] = linked_n[i,a,a1]

        # old implementation
        # linked = np.zeros((D,A,A))
        # for d in range(D):
        #     for a in range(A):
        #         for b in range(A):
        #             linked[d,a,b] = \
        #                 (U[a,d]>=L[b,d] and L[a,d]<=U[b,d]) \
        #                 or (U[b,d]>=L[a,d] and L[b,d]<=U[a,d])

        # calculates 'chained' relations if 'linked' relations change
        # chained[i,a1,a2]==1 iff a1 chains to a2 in objective i
        def dfsutil(i,a1,a2):
            chained[i,a1,a2] = 1
            for a3 in range(A):
                if linked[i,a2,a3] and not chained[i,a1,a3]:
                    dfsutil(i,a1,a3)
        if np.any(linked != linked_n):
            linked = linked_n
            chained = np.zeros((D,A,A))
            for i in range(D):
                for a1 in range(A):
                    dfsutil(i,a1,a1)

        # old implementation
        # chained = np.copy(linked)
        # for d in range(D):
        #     for c in range(A):
        #         for a in range(A):
        #             for b in range(A):
        #                 chained[d,a,b] = \
        #                     chained[d,a,b] \
        #                     or (chained[d,a,c] and chained[d,c,b])

        # arm selection
        a0 = np.argmax(U[:,0])
        A0 = [a for a in range(A) if chained[0,a,a0]]
        if np.any(C[A0] > eps/2):
            a = np.argwhere(C[A0] > eps/2)
            a = A0[np.random.choice(a.flatten())]
        else:
            Ai = A0
            for i in range(1, D-1):
                ai = Ai[np.argmax(U[Ai,i])]
                Ai = [a for a in Ai if chained[i,a,ai]]
            a = Ai[np.argmax(U[Ai,D-1])]

        # observes rewards and updates estimates and counters
        X = (np.random.uniform(size=(1,D)) <= mean[a]).astype(float)
        M[a] = (X + N[a]*M[a]) / (N[a]+1)
        N[a] += 1

        # regret
        for d in range(D):
            if mean[0,d] != mean[a,d]:
                reg[d,k,t//TT] += mean[0,d] - mean[a,d]
                break

    np.save('res/pf-lex.npy'.format(uid), [(A,D),mean,(dlt,eps),(K,T),reg])
