import pandas
import numpy as np


def create_random_data(csv_name, data, data_size):
    data_np = np.zeros((data_size, len(data)))
    for j, row in data.iterrows():
        print(j)
        stype = row['Storage Type']
        mu, sigma = row['Mean'], row['Std. Dev.']
        lower, upper = row['Min'], row['Max']
        if stype == 'byte':
            dp = np.random.randint(lower, upper + 1, data_size)
        elif stype == 'float':
            dp = np.random.normal(mu, sigma, data_size)
        data_np[:, j] = dp
    np.savetxt(csv_name, data_np, delimiter=',', fmt="%1.4f")


data = pandas.read_csv('vars.csv', sep=';')

create_random_data('rand_train.csv', data, 75000)
create_random_data('rand_valid.csv', data, 15000)

# np.save('ran_data.npy', data_np)


