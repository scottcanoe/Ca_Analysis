import h5py
import numpy as np
import xarray as xr
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

import utils

if __name__ == '__main__':

    n_iters = 100
    cmats = {0: [], 5: []}
    for day in (0, 5):

        # Collect time-averaged responses for all events
        event_to_data = {}
        event = 1
        for sequence in ('ABCD', 'ABBD', 'ACBD'):
            seq_data = utils.load_traces(day, sequence)
            for arr in utils.split_by_event(seq_data):
                arr = arr.isel(time=slice(2, None)).mean('time')
                event_to_data[event] = arr
                event += 1

        for it in range(n_iters):
            print(it)
            # Randomly pick which trials are for training and which
            # are for testing
            n_trials = 500
            split_size = 250
            train_ids = np.random.choice(n_trials, split_size, replace=False)
            test_ids = np.setdiff1d(np.arange(n_trials, dtype=int), train_ids)

            # Split responses into train/test groups.
            X_train, X_test = [], []
            y_train, y_test = [], []
            for event, arr in event_to_data.items():
                train = arr.isel(trial=train_ids)
                X_train.append(train)
                y_train.append(np.full(split_size, event))
                test = arr.isel(trial=test_ids)
                X_test.append(test)
                y_test.append(np.full(split_size, event))

            X_train = xr.concat(X_train, 'trial')
            y_train = np.concatenate(y_train)
            X_test = xr.concat(X_test, 'trial')
            y_test = np.concatenate(y_test)

            clf = SVC()
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)

            # confusion matrix
            cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
            cmats[day].append(cm)


    with h5py.File('data/decode_events.h5', 'w') as f:
        mat = np.stack(cmats[0]).astype(int)
        f.create_dataset('day_0', data=mat)
        mat = np.stack(cmats[5]).astype(int)
        f.create_dataset('day_5', data=mat)

