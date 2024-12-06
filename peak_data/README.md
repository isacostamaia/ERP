This data was obtained from clutering/cluster_across_subjects.py script.
For each subject the data was preprocessed and an average Target and non-Target ERP was obtained. For each one of these we computed the FRMS (Target and Non-Target). We obtain the FRMS peak idx for each one of the two conditions. Then we use those peak idx to get the EEG samples for each conditions.

XY = {"ERP":[], "Target":[], "Subject":[]}

ERP_i.shape == n_channels x n_peaks