import numpy as np
import Data, Model, Parameter
import csv, os, datetime, time
from tabulate import tabulate
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score

kf = KFold(n_splits=Parameter.num_folds, shuffle=True, random_state=42)

metrics = ['accuracy', 'mcc', 'f1']
models = {
    'Hinge': Model.Hinge_SVM(C=1),
    'GPIN': Model.GPIN_SVM(tau1=0.3, tau2=0.2, epsilon1=0.1, epsilon2=0.015, C=1),
    'TPIN': Model.TPIN_SVM(tau=0.3, alpha1=3, alpha2=3/5, C=1, num_iterations=10),
    'ATGP': Model.ATGP_SVM(tau1=0.1, tau2=0.3, epsilon1=0.1, epsilon2=0.015, alpha1=1, alpha2=1/5, C=1, num_iterations=10)
}

results = {name: {m: [] for m in metrics} for name in models}
times = {name: [] for name in models}

print('(Round 1)')
for k, (train_idx, test_idx) in enumerate(kf.split(Data.X), start=1):
    X_train, X_test = Data.X[train_idx], Data.X[test_idx]
    y_train, y_test = Data.y[train_idx], Data.y[test_idx]
    data = []

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        times[name].append(time.time() - start)
        y_pred = model.predict(X_test)
        results[name]['accuracy'].append(accuracy_score(y_test, y_pred))
        results[name]['mcc'].append(matthews_corrcoef(y_test, y_pred))
        results[name]['f1'].append(f1_score(y_test, y_pred))

    data.append(['Accuracy'] + [f"{results[m]['accuracy'][-1]:.4f}" for m in models])
    data.append(['MCC'] + [f"{results[m]['mcc'][-1]:.4f}" for m in models])
    data.append(['F1 scores'] + [f"{results[m]['f1'][-1]:.4f}" for m in models])
    print(tabulate(data, headers=[f"Fold: {k}"] + list(models.keys()), tablefmt="grid"))

summary_data = []
for metric in metrics:
    row = [metric.capitalize()]
    for name in models:
        avg = np.mean(results[name][metric]) * 100
        std = np.std(results[name][metric]) * 100
        row.append(f"{avg:.2f} (±S.D. {std:.2f})")
    summary_data.append(row)

print(tabulate(summary_data, headers=["Average"] + list(models.keys()), tablefmt="grid"))

all_data = []
for name in models:
    all_data.extend([
        [f"{name}, Noise level = {Parameter.noise_level}"],
        ["Average Accuracy", f"{np.mean(results[name]['accuracy']) * 100:.2f} ± {np.std(results[name]['accuracy']) * 100:.2f}"],
        ["Average MCC", f"{np.mean(results[name]['mcc']) * 100:.2f} ± {np.std(results[name]['mcc']) * 100:.2f}"],
        ["Average F1 scores", f"{np.mean(results[name]['f1']) * 100:.2f} ± {np.std(results[name]['f1']) * 100:.2f}"],
        []
    ])

file_name = os.path.splitext(os.path.basename(Data.data_file_path))[0]
date_str = datetime.datetime.now().strftime("%d_%m_%Y")
csv_path = f"{file_name}_{date_str}.csv"

with open(csv_path, 'a', newline='', encoding='utf-8') as f:
    csv.writer(f).writerows(all_data)

for name in models:
    print(f"Average time for {name} SVM: {np.mean(times[name]):.4f} seconds")
