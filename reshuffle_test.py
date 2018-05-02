import pandas as pd
import tracemalloc
import numpy as np
import sklearn

df = pd.read_csv('/Users/amir/projects/farone_test/faraone/rand_train.csv')

tracemalloc.start(10)

print("beginning the reshuffling")
time1 = tracemalloc.take_snapshot()
length = len(df.index)
for i in range(300):
    print(i)
    df = df.sample(frac=1).reset_index(drop=True)
    # df = sklearn.utils.shuffle(df)

    # df.sample(frac=1).reset_index(inplace=True, drop=True)
    # df.reindex(np.random.permutation(df.index))
    # df = df.iloc[np.random.permutation(length)]
    # df = sklearn.utils.shuffle(df)

time2 = tracemalloc.take_snapshot()
stats = time2.compare_to(time1, 'lineno')

for stat in stats[:1000]:
    print(stat)


