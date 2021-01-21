# Lex-MAB

Implementations of PF-LEX, OM-LEX, and NOM-LEX for the multi-armed bandit with lexicographically ordered objectives (Lex-MAB).

In all implementations:
- `A` is the number of arms,
- `D` is the number of objectives,
- `means` is an AxD matrix, where `means[a,i]` is the expected reward of arm a in objective i, and arm 0 is assumed to be lexicographic optimal,
- `K` is the number of individual runs,
- `T` is the number of rounds,
- `reg` is a DxKxT matrix, where `reg[i,k,t]` is the regret incurred in objective i, individual run k, and round t.

In the implementation of PF-LEX:
- `dlt` is the confidence term,
- `eps` is proportional to the suboptimality that the learner aims to tolerate,
- `TT` is the period in which the regrets are recorded.
Note that this script takes an argument called `uid`, which is a unique identifier with which the final results are saved. This is to allow for parallel execution of the script.

In the implementation of NOM-LEX:
- `eta` is a 1xD matrix, where `eta[0,i]` is the near lexicographic optimal expected reward in objective i.

# Sat-MAB

In addition to algorithms for Lex-MAB, 'satisficing.py' includes an adapted version of NOM-LEX for the multi-armed bandit with satisficing objectives (Sat-MAB), along with an implementation of Satisficing-In-Mean-Rewards UCL (Reverdy et al., "Satisficing in multi-armed bandit problems," 2017).
