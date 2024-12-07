#%%
import sys
import os
sys.path.append(os.path.abspath(".."))
import glob

import pickle 
import numpy as np
import pandas as pd
from moabb.datasets import BI2013a

from preprocessing.power import FRMS
from preprocessing.data_processing import get_clean_epochs, Lagger
from preprocessing.data_processing_iterative import AltFilters



def extract_peaks(dataset=BI2013a(), db_name = "BI2013a"):
    """
    Extracts peaks data (n_channels x 1) for Target and Non-Target responses
    and saves them as .pkl files (one for each subject)
    """


    XY = {"ERP":[], "Target":[], "Timeidx":[], "Session":[], "Subject":[], "Epoch":[]}

    for subj in dataset.subject_list:
        #
        # Get data and preprocess
        #
        epochs = get_clean_epochs(dataset, subjects_list = [subj])
        unique_sessions = np.unique(epochs.metadata['session'])

        for session in unique_sessions:
            session_epochs = epochs[epochs.metadata == session]

            #preprocess
            alt_filter = AltFilters(session_epochs, p=4)
            del session_epochs
            filtered_epochs, _ = alt_filter.fit_and_apply(class_="Target", plot_it=False)
            del alt_filter
            #Average filtered and lag-corrected target FRMS 
            lagger = Lagger(filtered_epochs["Target"])
            lag_corrected_epochs_tg = lagger.compute_and_correct_lags()

            #Average filtered and lag-corrected non-target FRMS 
            lagger = Lagger(filtered_epochs["NonTarget"])
            lag_corrected_epochs_ntg = lagger.compute_and_correct_lags()
            del lagger, filtered_epochs
            
            #
            #Extract peaks from FRMS of avg ERP
            #
            #FRMS
            frms_tg = FRMS(lag_corrected_epochs_tg.average())
            frms_ntg = FRMS(lag_corrected_epochs_ntg.average())

            #peaks idx
            peaks_idx_tg, _ = frms_tg.peaks_idx()
            peaks_idx_ntg, _ = frms_ntg.peaks_idx()

            del frms_tg, frms_ntg
            tg_samples = lag_corrected_epochs_tg.get_data()[:,:,peaks_idx_tg]
            ntg_samples = lag_corrected_epochs_ntg.get_data()[:,:,peaks_idx_ntg]

            XY["ERP"] = XY["ERP"] + list(tg_samples.transpose(0, 2, 1).reshape(-1, tg_samples.shape[1], 1)) #reshape to n_epochs*n_times x n_channels x 1 (put n_channels as last dim to conserve channels grouping)
            XY["Target"] = XY["Target"] + [1]*tg_samples.shape[0]*tg_samples.shape[-1] #n_epochs*n_times
            XY["Session"] = XY["Session"] + [session]*tg_samples.shape[0]*tg_samples.shape[-1] #n_epochs*n_times
            XY["Subject"] = XY["Subject"] + [subj]*tg_samples.shape[0]*tg_samples.shape[-1] #n_epochs*n_times
            XY["Timeidx"] = XY["Timeidx"] + list(peaks_idx_tg)*tg_samples.shape[0] #peaks_idx_tg has len==n_times. We should then multiply by n_epochs

            XY["ERP"] = XY["ERP"] + list(ntg_samples.transpose(0, 2, 1).reshape(-1, ntg_samples.shape[1], 1)) #reshape to n_epochs*n_times x n_channels x 1 (put n_channels as last dim to conserve channels grouping)
            XY["Target"] = XY["Target"] + [0]*ntg_samples.shape[0]*ntg_samples.shape[-1] #n_epochs*n_times
            XY["Session"] = XY["Session"] + [session]*ntg_samples.shape[0]*ntg_samples.shape[-1] #n_epochs*n_times
            XY["Subject"] = XY["Subject"] + [subj]*ntg_samples.shape[0]*ntg_samples.shape[-1] #n_epochs*n_times
            XY["Timeidx"] = XY["Timeidx"] + list(peaks_idx_ntg)*ntg_samples.shape[0] #peaks_idx_ntg has len==n_times. We should then multiply by n_epochs

            del lag_corrected_epochs_tg, lag_corrected_epochs_ntg


        #save
        path = os.path.join(os.path.dirname(__file__), '../data/peak_data/{}'.format(db_name))
        #verify if folder exists, create otherwise
        os.makedirs(path, exist_ok=True)
        # Filepath for the image
        dict_path = os.path.join(path, 'XY_{}_subj{}.pkl'.format(db_name,subj))
        with open(dict_path, 'wb') as f:
            pickle.dump(XY, f)


def extract_window_mean(dataset=BI2013a(), db_name = "BI2013a"):
    """
    Extracts peaks data (n_channels x 1) for Target and Non-Target responses
    and saves them as .pkl files (one for each subject)
    """
    XY = {"ERP":[], "Target":[], "Timeidx":[], "Session":[], "Subject":[], "Epoch":[]}

    for subj in dataset.subject_list[3:]:
        #
        # Get data and preprocess
        #
        epochs = get_clean_epochs(dataset, subjects_list = [subj])
        sfreq = epochs.info['sfreq'] #sampling frequency
        unique_sessions = np.unique(epochs.metadata['session'])

        #treat one session at a time
        for session in unique_sessions:
            session_epochs = epochs[epochs.metadata.session.values == session]

            #preprocess
            alt_filter = AltFilters(session_epochs, p=4)
            del session_epochs
            filtered_epochs, _ = alt_filter.fit_and_apply(class_="Target", plot_it=False)
            del alt_filter
            #Average filtered and lag-corrected target FRMS 
            lagger = Lagger(filtered_epochs["Target"])
            lag_corrected_epochs_tg = lagger.compute_and_correct_lags()

            #Average filtered and lag-corrected non-target FRMS 
            lagger = Lagger(filtered_epochs["NonTarget"])
            lag_corrected_epochs_ntg = lagger.compute_and_correct_lags()
            del lagger, filtered_epochs

            #
            # Extract window mean
            #
            # Parameters
            #get number of samples corresponding to ~t_interval ms
            #t_interval: time window duration in ms
            t_interval = 30
            len_window = int(t_interval*1e-3*sfreq) # Length of the window ~ around t_interval as number of samples
            step = 3         # Step size for advancing the window

            tg_samples = lag_corrected_epochs_tg.get_data()
            ntg_samples = lag_corrected_epochs_ntg.get_data()

            _, n_channels, n_times = tg_samples.shape
            # Slide the window over the time axis
            for start in range(0, n_times - len_window + 1, step):
                end = start + len_window
                # Compute the average within the window and reshape to (n_epochs, n_channels, 1)
                window_avg_tg = np.mean(tg_samples[:, :, start:end], axis=2) #(n_epochs, n_channels,)
                window_avg_tg = np.expand_dims(window_avg_tg, 2) #(n_epochs,n_channels,1)

                window_avg_ntg = np.mean(ntg_samples[:, :, start:end], axis=2) #(n_epochs, n_channels,)
                window_avg_ntg = np.expand_dims(window_avg_ntg, 2) #(n_epochs,n_channels,1)

                XY["ERP"] = XY["ERP"] + list(window_avg_tg) #len = n_epochs
                XY["Target"] = XY["Target"] + [1]*window_avg_tg.shape[0] #n_epochs
                XY["Session"] = XY["Session"] + [session]*window_avg_tg.shape[0] #n_epochs
                XY["Subject"] = XY["Subject"] + [subj]*window_avg_tg.shape[0] #n_epochs
                XY["Timeidx"] = XY["Timeidx"] + [(start, end)]*window_avg_tg.shape[0] 
                XY["Epoch"] = XY["Epoch"] + list(lag_corrected_epochs_tg.metadata.index) #n_epochs


                XY["ERP"] = XY["ERP"] + list(window_avg_ntg)
                XY["Target"] = XY["Target"] + [0]*window_avg_ntg.shape[0] #n_epochs
                XY["Session"] = XY["Session"] + [session]*window_avg_ntg.shape[0] #n_epochs
                XY["Subject"] = XY["Subject"] + [subj]*window_avg_ntg.shape[0] #n_epochs
                XY["Timeidx"] = XY["Timeidx"] + [(start, end)]*window_avg_ntg.shape[0] 
                XY["Epoch"] = XY["Epoch"] + list(lag_corrected_epochs_ntg.metadata.index) #n_epochs

            del lag_corrected_epochs_tg, lag_corrected_epochs_ntg

        #save
        path = os.path.join(os.path.dirname(__file__), '../data/window_mean_data/{}/'.format(db_name))
        #verify if folder exists, create otherwise
        os.makedirs(path, exist_ok=True)
        # Filepath for the image
        dict_path = os.path.join(path, 'XY_{}_subj{}.pkl'.format(db_name,subj))
        with open(dict_path, 'wb') as f:
            pickle.dump(XY, f)




def read_pkl_files(folder = os.path.join("..", "data", "window_mean_data"), db_name = '*'):
    """
    Reads .pkl files in folder and returns df with peak data 
    for db_name directory 
        Params: 
            db_name (e.g. db_name = 'BI2013a' or '*' for all db directories in folder)
        Returns: df_peaks 
    """
    path = os.path.join(os.path.dirname(__file__), folder)
    print(path)
    # List files inside all directories in folder
    pkls_all_subj = [d for d in glob.glob(os.path.join(path, db_name))]
    dict_subj = {}
    for d in pkls_all_subj:
        #read pickle files as df and concatenate to dict_subj
        dict_subj.update(pd.read_pickle(d))
    #for all subjects    
    df = pd.DataFrame(dict_subj)
    return df


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    #main()
    #extract_peaks()
    df = read_pkl_files(folder= os.path.join("..", "data", "window_mean_data"))





# %%
