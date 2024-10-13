import numpy as np
import pywt

###########################################################################
###### Algoritms to generate Features from wavelets overlapping layers ####
###### Wavelet-Based Adaptive Layered algorithm (WALE-a)####

def sliding_ts_v2(data, window):
    #Receive a 1d array
    out = np.array([])

    #Convert to 2d arra
    data = data[None,:]
    #Set variables
    s = data.shape
    col = s[1]
    col_extent = col - window + 1
    row_extent = 1

    #Create indice to sliding
    start_idx = np.arange(1)[:, None] * col + np.arange(window)
    offset_idx = np.arange(row_extent)[:, None] * col + np.arange(col_extent)

    #MOUNT MATRIX WITH SLIDING WINDOWS
    out = np.take(data, start_idx.ravel()[:, None] + offset_idx.ravel())

    #Return array with all windows already slided
    return out.T

def wave_layer_ts_v2(input_ts, win_len, wl_name=['Haar'], t_pool=0):
    # Perform Wavelet-Based Adaptive Layered Transformation
    # The proposed algorithm utilizes Wavelet decompositions within a sliding window framework to extract features using statistical metrics.
    # These extracted metrics are subsequently consolidated through a pooling layer, resulting in a concise and powerful representation
    # of patterns within the time series data.

    # Perform convolution and extraction layer
    conv_layer = convolutional_wave_layer_ts(input_ts, win_len, wl_name, t_pool, pad=0)

    # Perform Pooling layer

    return conv_layer

def convolutional_wave_layer_ts(input_ts, win_len, wl_name, t_pool, pad=0):

    #instance variables
    #Create array of paddings variables
    num_layers = len(wl_name)
    paddings = np.zeros(num_layers, dtype = np.int32)
    # Set padding
    if pad == 1:
        paddings = np.where(np.random.randint(2, size=num_layers) == 1, (win_len - 1) // 2, 0)

    #Set the qtd of examples
    num_examples = input_ts.shape[0]

    #Set output of the wavelets layer
    fea_map_final = []

    #Perform convolution-process (wavelet transformations) in all events (rows)
    for i in range(num_examples):
        # Set list to receive each layer
        fea_map_wave = []

        # Perform transformations to each wavelet
        # The transformation are applied into sliding windows, according
        for j_idx, j_name in enumerate(wl_name):

            # Set list to receive each layer
            fea_map_tmp = []

            # perform sliding
            windows = sliding_ts_v2(input_ts[i], win_len)

            # Add the sliding windows without transformation, only time###
            if j_idx == 0:
                fea_map_tmp.append(windows)

            #Padded input TS
            windows_pad = np.pad(windows, ((0, 0), (paddings[j_idx], paddings[j_idx])), mode='constant')

            # Perform wavelet transformation - Strategie 2
            sig_dec = pywt.wavedec(windows_pad, j_name)
            # Separating all coeffs approximation and details (cA, Cd1,cD2,cD3...)
            fea_map_tmp.append(sig_dec)

            # Perform wavelet transformation - Strategie 3
            sig_dec = pywt.wavedec(windows_pad, j_name)
            # Reconstruction of signal, only last level (cA + cD)
            fea_map_tmp.append(pywt.waverec(sig_dec[:2], j_name))

            ###Perform the process extraction layer###

            ###Pooling###
            tmp = extraction_wavelet_layer_v1([fea_map_tmp], win_len, t_pool)

            #Mount of list with extracted features
            fea_map_wave.append(tmp)


        #Constructor OUTPUT with all features maps of all examples
        concatenated_arrays = []
        for sublist in fea_map_wave:
            concatenated_sublist = np.hstack(sublist)
            concatenated_arrays.append(concatenated_sublist)
        result = np.hstack(concatenated_arrays)
        fea_map_final.append(result)

    return np.vstack(fea_map_final)

def extraction_wavelet_layer_v1(list_all_fea_map, win_len, t_pool):

    #Perform pooling
    # The input (list_all_fea_map) must be list-of-list

    #Here receive decomposition list with array 2-d with windows x metrics, and
    #Perform the extraction of features by many statistics
    for element in list_all_fea_map:
        decomp_list = get_stat_features(element, win_len)

    #Perform pooling strategie
    # The tmp_list have all measures of sliding windows
    # The Pooling strategies is seleted
    fea_map_pooling = poolins_v1(decomp_list, t_pool)

    return fea_map_pooling

def poolins_v1(conv_data, t_pool):
    if t_pool == 0:
        #Perform pooling layer - strategies #1 with MEAN  Pooling
        pooling = pooling_mean(conv_data)
    elif t_pool == 1:
        # Perform pooling layer - strategies #2 with MAX Pooling
        pooling = pooling_max(conv_data)
    elif t_pool == 2:
        # Perform pooling layer - strategies #4 MEAN and PPV Pooling
        pooling = pooling_mean_ppv(conv_data)
    elif t_pool == 3:
        # Perform pooling layer - strategies #5 MAX and PPV Pooling
        pooling = pooling_max_ppv(conv_data)
    else:
        quit()

    return pooling

def pooling_mean(list_all_fea_map):
    #perform MEAN pooling
    fea_map_pooling = [np.mean(arr,axis=0) for arr in list_all_fea_map]

    return fea_map_pooling

def pooling_mean_ppv(list_all_fea_map):
    #perform MEAN and PPV pooling

    fea_map_pooling = []
    fea_map_pooling_ppv = [np.mean(arr > 0,axis=0) for arr in list_all_fea_map]  # Compute the PPV for each kernel
    fea_map_pooling_mean = [np.mean(arr,axis=0) for arr in list_all_fea_map] #Compute mean pooling for each kernel

    for arr1, arr2 in zip(fea_map_pooling_ppv,fea_map_pooling_mean):
        fea_map_pooling.append(arr1)
        fea_map_pooling.append(arr2)

    return fea_map_pooling

def pooling_mean_mpv(list_all_fea_map):
    #perform MEAN and MPV pooling

    fea_map_pooling = []
    fea_map_pooling_mpv = [np.divide(np.sum(arr[arr > 0], axis=0),np.where(np.sum(arr > 0, axis=0) > 0, np.sum(arr > 0, axis=0), 1)) if np.any(arr > 0) else -1 for arr in list_all_fea_map] # Compute the PPV for each kernel
    fea_map_pooling_mean = [np.mean(arr,axis=0) for arr in list_all_fea_map] #Compute MEAN pooling for each kernel

    for arr1, arr2 in zip(fea_map_pooling_mpv,fea_map_pooling_mean):
        fea_map_pooling.append(arr1)
        fea_map_pooling.append(arr2)

    return fea_map_pooling

def pooling_max_ppv(list_all_fea_map):
    #perform MAX and PPV pooling

    fea_map_pooling = []
    fea_map_pooling_ppv = [np.mean(arr > 0,axis=0) for arr in list_all_fea_map]  # Compute the PPV for each kernel
    fea_map_pooling_max = [np.max(arr,axis=0) for arr in list_all_fea_map] #Compute max pooling for each kernel

    for arr1, arr2 in zip(fea_map_pooling_ppv,fea_map_pooling_max):
        fea_map_pooling.append(arr1)
        fea_map_pooling.append(arr2)

    return fea_map_pooling

def pooling_max(list_all_fea_map):
    #perform MAX pooling
    fea_map_pooling = [np.max(arr,axis=0) for arr in list_all_fea_map]

    return fea_map_pooling

def pooling_sum(list_all_fea_map):
    #perform MEAN pooling
    fea_map_pooling = [np.sum(arr,axis=0) for arr in list_all_fea_map]

    return fea_map_pooling

def get_stat_features(data_list, win_len):
    list_features = []
    # Compute some measures to each element's list
    for element in data_list:

        # Check if the element is a numpy array
        if isinstance(element, np.ndarray):
            measures = calculate_measures_v2(element, win_len)
            list_features.append(measures)

        # Check if the element is a list
        elif isinstance(element, list):
            # Recursively call the function for the sublist
            sublist = get_stat_features(element, win_len)
            list_features.extend(sublist)

    return list_features

def calculate_measures_v2(data_array, win_len):
    # Calculate measures - MEAN
    mean = np.mean(data_array, axis=1)

    # Calculate measures - STD
    std = np.std(data_array, axis=1)

    # Calculate measures - MAX
    max_value = np.max(data_array, axis=1)

    # Calculate measures - MIN
    min_value = np.min(data_array, axis=1)

    # Calculate area under windows - Trapezoidal rule
    area = np.trapz(data_array, axis=1)

    # Calculate RMS under windows
    rms = np.sqrt(np.sum(data_array ** 2, axis=1) / win_len)

    # Calculate MAV of windows
    mav = np.sum(np.abs(data_array), axis=1) / win_len

    # Calculate Zero crossing of windows
    zero_crossing = np.sum(np.diff(data_array > 0), axis=1)

    # Calculate Average Power (AP)
    average_power = np.sum(data_array ** 2, axis=1) / win_len

    # Calculate WINAMP
    u_windows = np.mean(np.abs(np.diff(data_array)), axis=1)[:, None]
    winamp = np.sum(np.abs(np.diff(data_array)) >= u_windows, axis=1)

    # Calculate Wavelength
    wavelength = np.sum(np.abs(np.diff(data_array)), axis=1)

    # Calculate measures - PERCENTILE 5%
    #percentile_5 = np.percentile(data_array, 5, axis=1)

    # Stack the calculated measures into a single array
    list_measures = np.column_stack((mean, std, max_value, min_value, area, rms, mav, zero_crossing, average_power, winamp, wavelength))

    return list_measures

def zero_pad_ts(data, pad):
    # Perform padded
    return np.pad(data, pad)

def get_max_levels(lst,win_len):
    max_levels = {}

    for element in lst:
        wavelet = pywt.Wavelet(element)
        filter_len = wavelet.dec_len
        max_levels[element] = pywt.dwt_max_level(win_len,filter_len)

    return max_levels

###########################################################################
