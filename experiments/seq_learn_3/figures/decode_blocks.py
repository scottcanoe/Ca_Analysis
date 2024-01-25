import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import xarray as xr

from ca_analysis.stats import *
from seq_learn_3.utils import *

"""
See how well the decoder can tell which block an event came from.
"""

# def decode(event, day) -> np.ndarray:



def get_cmat(day, event, kernel='rbf'):

    sessions = get_sessions(day=day, fs=0)
    # apply_roi_filter(sessions, 'visual')
    ev = schema.get(event=event)
    arr = flex_split(sessions, ev)
    arr = arr.isel(time=slice(-4, None))
    arr = arr.mean('time')

    X_train, X_test = [], []
    y_train, y_test = [], []
    for i in range(0, 500, 100):
        block = arr.isel(trial=slice(i, i + 100))
        train_inds = np.random.choice(100, 50, replace=False)
        test_inds = np.setdiff1d(np.arange(100), train_inds)
        X_train.append(block.isel(trial=train_inds))
        y_train.append(np.full(50, i))
        X_test.append(block.isel(trial=test_inds))
        y_test.append(np.full(50, i))

    X_train = xr.concat(X_train, 'trial')
    y_train = np.concatenate(y_train)
    X_test = xr.concat(X_test, 'trial')
    y_test = np.concatenate(y_test)

    clf = SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    cm = normalize_confusion_matrix(cm)
    return cm


events = schema.events[1:]
events = [ev for ev in events if not ev.name.endswith('_')]
# events = [ev for ev in events if ev.name.endswith('_')]

cmats = []
for ev in events:
    cmats.append(get_cmat(0, ev))
cm_0 = np.stack(cmats).mean(axis=0)

cmats = []
for ev in events:
    cmats.append(get_cmat(5, ev))
cm_5 = np.stack(cmats).mean(axis=0)

plt.rcParams['font.size'] = 7
fig, axes = plt.subplots(1, 2, figsize=(4.5, 2))

ax = axes[0]
im = ax.imshow(cm_0, cmap='inferno', vmin=0, vmax=0.8)
ticks = np.arange(5)
tick_labels = ['1', '2', '3', '4', '5']
ax.set_xticks(ticks)
ax.set_xticklabels(tick_labels)
ax.set_yticks(ticks)
ax.set_yticklabels(tick_labels)
ax.set_xlabel('predicted block')
ax.set_ylabel('true block')
# plt.colorbar(im)

ax = axes[1]
im = ax.imshow(cm_5, cmap='inferno', vmin=0, vmax=0.8)
ticks = np.arange(5)
tick_labels = ['1', '2', '3', '4', '5']
ax.set_xticks(ticks)
ax.set_xticklabels(tick_labels)
ax.set_yticks(ticks)
ax.set_yticklabels(tick_labels)
# ax.set_xlabel('predicted block')
# ax.set_ylabel('true block')
plt.colorbar(im)

fig.tight_layout(pad=1)
plt.show()


fig.savefig('figures/decode_blocks.eps')


acc_0 = np.diag(cm_0).mean()
acc_5 = np.diag(cm_5).mean()
print(f'day 0 accuracy: {acc_0}')
print(f'day 5 accuracy: {acc_5}')
