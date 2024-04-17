import numpy as np
from rpy2.robjects import r

# Function to read RDS file and convert it to numpy array                                           
def read_rds_to_numpy(file_path):
    rds_data = r['readRDS'](file_path)

    # Convert the R object to a numpy array                                                                         
    np_array = np.array(rds_data)

    return np_array

file_path = 'src/data/max_stable_sims.rds'
normalise = True

data = read_rds_to_numpy(file_path)
n_data = data.shape[0]

data = np.log(data) # Convert to Gumbel scale

mean_all = np.median(data)
std_all =  np.median(np.abs(data - mean_all)) * 1.5

for tt in range(n_data):
   
        # Standardise the sample panels (if specified)
        if normalise:
            data[tt, :, :] = (data[tt, :, :] - mean_all) / std_all

np.save('src/intermediates/max_stable_data_Gumbel.npy', data)
np.save('src/intermediates/max_stable_data_Gumbel_mean_std.npy', [mean_all, std_all])