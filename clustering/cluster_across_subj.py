#%%
import sys
import os
sys.path.append(os.path.abspath(".."))
import glob

import pickle 
import numpy as np
import pandas as pd
from moabb.datasets import BI2013a

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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


def read_pkl_files(folder = os.path.join("..", "data", "window_mean_data"), db_name = '*', subject = "*"):
    """
    Reads .pkl files in folder and returns df with peak data 
    for db_name directory 
        Params: 
            db_name: (e.g. db_name = 'BI2013a' or '*' for all db directories in folder)
            subject: string number of subject file inside folder/db_name. subject =  "*" if all subjects are desired.
        Returns: df 
    """
    path = os.path.join(os.path.dirname(__file__), folder)
    if subject != "*":
        subject = "XY_" + db_name + "_subj" + subject + ".pkl"
    # List desired files inside desired db folder
    pkls_all_subj = [d for d in glob.glob(os.path.join(path, db_name, subject))]
    print("Reading the files: \n", pkls_all_subj)
    dict_subj = {}
    for d in pkls_all_subj:
        #read pickle files as df and concatenate to dict_subj
        dict_subj.update(pd.read_pickle(d))
    #for all subjects    
    df = pd.DataFrame(dict_subj)
    return df


def get_XY_data(subject = "1", db_name = 'BI2013a', session = "0"):
    """
        Reads pickle files and returns X and y data
        Params:
            db_name: (e.g. db_name = 'BI2013a' or '*' for all db directories in folder)
            subject: string number of subject file inside folder/db_name. subject =  "*" if all subjects are desired.
        Returns:
            X, y
    """

    folder = os.path.join("..", "data", "window_mean_data")
    df = read_pkl_files(folder = folder, db_name = db_name, subject = subject)

    df_session = df[df["Session"] == session]
    # Flatten the 'ERP' n_channelsx1 arrays into n_channels-element vectors
    df_session['ERP'] = df_session['ERP'].apply(lambda x: np.array(x).flatten())

    df_agg_epochs = df_session.groupby(by="Epoch").agg({
        "Session": lambda x: x.unique()[0] if len(x.unique()) == 1 else ValueError("Multiple sessions found"),
        "ERP": lambda x: np.vstack(x.to_list()),  # Ensure values can be stacked
        "Timeidx": lambda x: list(x.unique()),    # Return unique values as a list
        "Subject": lambda x: list(x.unique())[0] if len(x.unique()) == 1 else ValueError("Multiple subjects found"),    # Return unique values as a list
        "Target": lambda x: list(x.unique())[0]  if len(x.unique()) == 1 else ValueError("Multiple targets in an epoch found")    # Return unique values as a list
        })

    tg_ntg_counts = df_agg_epochs["Target"].value_counts()
    num_epochs_ntg, num_epochs_tg = tg_ntg_counts[tg_ntg_counts.index==0]; tg_ntg_counts[tg_ntg_counts.index==1]
    print("Number of (Target, NonTarget) Epochs in the session: ", num_epochs_tg, num_epochs_ntg)
    print("Number of Epochs in the session: ", len(df_agg_epochs))

    #Get data features and fit SVM
    
    # Stack the flattened ERP into a feature matrix
    X = np.vstack(df_session['ERP'].values)  # Shape: (n_samples, 16)
    y = df_session['Target'].values  # Class labels

    # Normalize the feature matrix
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def plot_roc_subj_session(X, y):
    """
    Plot ROC curve for SVM model of as subject in a session. Prediction probabilities are
    the mean of prediction probabilities obtained in cross validation. These mean predictions are
    used for tracing the ROC curve.
    """

    # Set up the SVM classifier and try to compensate class imbalance
    svm = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')

    # cross-validation and AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba = cross_val_predict(svm, X, y, cv=cv, method='predict_proba')[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    roc_auc = auc(fpr, tpr)
    y_predict = 



    # Plot AUC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("SVM ROC Curve" + db_name + " subject " + subject + " session " + session)
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

def acc_on_time(X, y):
    """
    Use it for a single subject and session df
    """
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(Data, Target, test_size=0.3, random_state=0, stratify=Target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)



if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    #main()
    #extract_peaks()

 #function that recovers data and transoform into normalized X and y
# %%
