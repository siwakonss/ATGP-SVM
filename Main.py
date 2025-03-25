import numpy as np
import Data
import Model
import Parameter
import csv
import os
import datetime
import time
from tabulate import tabulate
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score







# Train the model

# Initialize lists to store evaluation metrics for each fold
fold_accuracies_Hinge = []
fold_mcc_scores_Hinge = []
fold_f1_scores_Hinge = []
fold_accuracies_GPIN = []
fold_mcc_scores_GPIN = []
fold_f1_scores_GPIN = []
fold_accuracies_TPIN = []
fold_mcc_scores_TPIN = []
fold_f1_scores_TPIN = []
fold_accuracies_ATGP = []
fold_mcc_scores_ATGP = []
fold_f1_scores_ATGP = []
time_Hinge = []
time_GPIN = []
time_TPIN = []
time_eTPIN = []
time_TGP = []
time_ATGP = []


kf = KFold(n_splits=Parameter.num_folds, shuffle=True, random_state=42)


start_time = time.time()    
for i in range(1):
    k=1
    print(f'(Round {i+1})')

    for train_index, test_index in kf.split(Data.X):
        data = []
        # Split the data into training and testing sets for the current fold
         

        X_train, X_test = Data.X[train_index], Data.X[test_index]
        #print(X_train.shape) 
        y_train, y_test = Data.y[train_index], Data.y[test_index]

        # Train your custom SVM model on the current fold (as you were doing)
        classifier_Hinge = Model.Hinge_SVM(C=1)
        classifier_GPIN = Model.GPIN_SVM(tau1=0.3, tau2=0.2 , epsilon1=0.1, epsilon2=0.015, C=1)
        classifier_TPIN = Model.TPIN_SVM(tau=0.3, alpha1=3, alpha2=3/5, C=1, num_iterations=10)
        classifier_ATGP = Model.ATGP_SVM(tau1 = 0.1 , tau2 = 0.3, epsilon1 = 0.1, epsilon2 = 0.015, alpha1=1, alpha2=1/5, C=1, num_iterations=10)
        




        # Train your custom SVM model on the current fold (as you were doing)
        start_time = time.time() 
        w_Hinge, b_Hinge  = classifier_Hinge.fit(X_train, y_train)
        time_Hinge.append(time.time() - start_time)
        start_time = time.time()
        w_GPIN, b_GPIN  = classifier_GPIN.fit(X_train, y_train)
        time_GPIN.append(time.time() - start_time)
        start_time = time.time()
        w_TPIN, b_TPIN  = classifier_TPIN.fit(X_train, y_train)
        time_TPIN.append(time.time() - start_time)
        start_time = time.time()
        w_ATGP, b_ATGP  = classifier_ATGP.fit(X_train, y_train)
        time_ATGP.append(time.time() - start_time)
        

        


        # Create an instance of your custom SVM classifier with '(w, b)'
        

        
        
        # Make predictions on the test set
        y_pred_Hinge = classifier_Hinge.predict(X_test)
        y_pred_GPIN = classifier_GPIN.predict(X_test)
        y_pred_TPIN = classifier_TPIN.predict(X_test)
        y_pred_ATGP = classifier_ATGP.predict(X_test)


        
        # Calculate accuracy for the current fold
        fold_accuracy_Hinge = accuracy_score(y_test, y_pred_Hinge)
        fold_accuracy_GPIN = accuracy_score(y_test, y_pred_GPIN)
        fold_accuracy_TPIN = accuracy_score(y_test, y_pred_TPIN)

        fold_accuracy_ATGP = accuracy_score(y_test, y_pred_ATGP)

        
        
        #print('Fold:', k, 'Accuracy (TPIN): ', fold_accuracy_TPIN, '   | (TGP):', fold_accuracy_TGP)
        fold_accuracies_Hinge.append(fold_accuracy_Hinge)
        fold_accuracies_GPIN.append(fold_accuracy_GPIN)
        fold_accuracies_TPIN.append(fold_accuracy_TPIN)
        fold_accuracies_ATGP.append(fold_accuracy_ATGP)
    

        # Calculate MCC for the current fold
        fold_mcc_Hinge = matthews_corrcoef(y_test, y_pred_Hinge)
        fold_mcc_GPIN = matthews_corrcoef(y_test, y_pred_GPIN) 
        fold_mcc_TPIN = matthews_corrcoef(y_test, y_pred_TPIN)
        fold_mcc_ATGP = matthews_corrcoef(y_test, y_pred_ATGP)
        #print('Fold:', k, 'MCC (TPIN):      ', fold_mcc_TPIN, '   | (TGP):', fold_mcc_TGP)
        fold_mcc_scores_Hinge.append(fold_mcc_Hinge)
        fold_mcc_scores_GPIN.append(fold_mcc_GPIN)
        fold_mcc_scores_TPIN.append(fold_mcc_TPIN)
        fold_mcc_scores_ATGP.append(fold_mcc_ATGP)

        # Calculate F1 score for the current fold
        fold_f1_Hinge = f1_score(y_test, y_pred_Hinge)
        fold_f1_GPIN = f1_score(y_test, y_pred_GPIN)
        fold_f1_TPIN = f1_score(y_test, y_pred_TPIN)

        fold_f1_ATGP = f1_score(y_test, y_pred_ATGP)
        #print('Fold:', k, 'F1 scores (TPIN):', fold_f1_TPIN, '   | (TGP):', fold_f1_TGP)
        fold_f1_scores_Hinge.append(fold_f1_Hinge)
        fold_f1_scores_GPIN.append(fold_f1_GPIN)
        fold_f1_scores_TPIN.append(fold_f1_TPIN)
        fold_f1_scores_ATGP.append(fold_f1_ATGP)
        row = ['Accuracy', f' {fold_accuracy_Hinge}', f' {fold_accuracy_GPIN}', f' {fold_accuracy_TPIN}', f' {fold_accuracy_ATGP}']
        data.append(row)
        
        row = ['MCC',f' {fold_mcc_Hinge}', f' {fold_mcc_GPIN}', f' {fold_mcc_TPIN}', f' {fold_mcc_ATGP}']
        data.append(row)

        row = ['F1 scores', f' {fold_f1_Hinge}', f' {fold_f1_GPIN}', f' {fold_f1_TPIN}', f'{fold_f1_ATGP}']
        data.append(row)

        # Define the headers for your table
        headers = [f'Fold: {k}',"Hinge", "GPIN", "TPIN", "ATGP"]

        # Use the tabulate function to format the data into a table
        table = tabulate(data, headers, tablefmt="grid")

        # Print the formatted table
        print(table)
        k += 1
        
    data = []
    # Calculate the average accuracy, MCC, and F1 score across all folds
    average_accuracy_Hinge = np.mean(fold_accuracies_Hinge)*100
    average_mcc_Hinge = np.mean(fold_mcc_scores_Hinge)*100
    average_f1_Hinge = np.mean(fold_f1_scores_Hinge)*100
    average_accuracy_GPIN = np.mean(fold_accuracies_GPIN)*100
    average_mcc_GPIN = np.mean(fold_mcc_scores_GPIN)*100
    average_f1_GPIN = np.mean(fold_f1_scores_GPIN)*100
    average_accuracy_TPIN = np.mean(fold_accuracies_TPIN)*100
    average_mcc_TPIN = np.mean(fold_mcc_scores_TPIN)*100
    average_f1_TPIN = np.mean(fold_f1_scores_TPIN)*100
    average_accuracy_ATGP = np.mean(fold_accuracies_ATGP)*100
    average_mcc_ATGP = np.mean(fold_mcc_scores_ATGP)*100
    average_f1_ATGP = np.mean(fold_f1_scores_ATGP)*100

    # Print or store the evaluation metrics
    # Calculate the standard deviation for accuracy, MCC, and F1 scores across all folds
    std_accuracy_Hinge = np.std(fold_accuracies_Hinge)*100
    std_mcc_Hinge = np.std(fold_mcc_scores_Hinge)*100
    std_f1_Hinge= np.std(fold_f1_scores_Hinge)*100
    std_accuracy_GPIN = np.std(fold_accuracies_GPIN)*100
    std_mcc_GPIN = np.std(fold_mcc_scores_GPIN)*100
    std_f1_GPIN = np.std(fold_f1_scores_GPIN)*100
    std_accuracy_TPIN = np.std(fold_accuracies_TPIN)*100
    std_mcc_TPIN = np.std(fold_mcc_scores_TPIN)*100
    std_f1_TPIN = np.std(fold_f1_scores_TPIN)*100
    std_accuracy_ATGP = np.std(fold_accuracies_ATGP)*100
    std_mcc_ATGP = np.std(fold_mcc_scores_ATGP)*100
    std_f1_ATGP = np.std(fold_f1_scores_ATGP)*100

    # Convert standard deviation to a percentage
    std_accuracy_Hinge_percent = (std_accuracy_Hinge)
    std_mcc_Hinge_percent = (std_mcc_Hinge ) 
    std_f1_Hinge_percent = (std_f1_Hinge ) 
    std_accuracy_GPIN_percent = (std_accuracy_GPIN )
    std_mcc_GPIN_percent = (std_mcc_GPIN ) 
    std_f1_GPIN_percent = (std_f1_GPIN ) 
    std_accuracy_TPIN_percent = (std_accuracy_TPIN ) 
    std_mcc_TPIN_percent = (std_mcc_TPIN ) 
    std_f1_TPIN_percent = (std_f1_TPIN ) 
    std_accuracy_ATGP_percent = (std_accuracy_ATGP ) 
    std_mcc_ATGP_percent = (std_mcc_ATGP ) 
    std_f1_ATGP_percent = (std_f1_ATGP ) 
    
    # Add rows for "Average Accuracy," "Average MCC," and "Average F1 scores"
    data.append(['Accuracy', f'{average_accuracy_Hinge:.2f} (±S.D. {std_accuracy_Hinge_percent:.2f})',f'{average_accuracy_GPIN:.2f} (±S.D. {std_accuracy_GPIN_percent:.2f})', f'{average_accuracy_TPIN:.2f} (±S.D. {std_accuracy_TPIN_percent:.2f})', f'{average_accuracy_ATGP:.2f} (±S.D. {std_accuracy_ATGP_percent:.2f})'])
    data.append(['MCC',f'{average_mcc_Hinge:.2f} (±S.D. {std_mcc_Hinge_percent:.2f})', f'{average_mcc_GPIN:.2f} (±S.D. {std_mcc_GPIN_percent:.2f})', f'{average_mcc_TPIN:.2f} (±S.D. {std_mcc_TPIN_percent:.2f})', f'{average_mcc_ATGP:.2f} (±S.D. {std_mcc_ATGP_percent:.2f})'])
    data.append(['F1 scores', f'{average_f1_Hinge:.2f} (±S.D. {std_f1_Hinge_percent:.2f})', f'{average_f1_GPIN:.2f} (±S.D. {std_f1_GPIN_percent:.2f})', f'{average_f1_TPIN:.2f} (±S.D. {std_f1_TPIN_percent:.2f})',  f'{average_f1_ATGP:.2f} (±S.D. {std_f1_ATGP_percent:.2f})'])
    

    headers = ['Average', "Hinge", "GPIN", "TPIN",  "ATGP"]

    # Use the tabulate function to format the data into a table
    table_average = tabulate(data, headers, tablefmt="grid")

    # Print the formatted table
    print(table_average)
    
    all_data = []
    models = ['Hinge', 'GPIN', 'TPIN',  'ATGP']
    for model in models:
        current_run_data = [
            [ f'{model}, Noise level = {Parameter.noise_level}'],
            ['Average Accuracy', f'{globals()[f"average_accuracy_{model}"]:.2f} ± {globals()[f"std_accuracy_{model}_percent"]:.2f}'],
            ['Average MCC', f'{globals()[f"average_mcc_{model}"]:.2f} ± {globals()[f"std_mcc_{model}_percent"]:.2f}'],
            ['Average F1 scores', f'{globals()[f"average_f1_{model}"]:.2f} ± {globals()[f"std_f1_{model}_percent"]:.2f}']
        ]
        #all_data.append([f'{model}, epsilon = {epsilon}'])
        all_data.extend(current_run_data)
        all_data.append([])

    file_name = os.path.splitext(os.path.basename(Data.data_file_path))[0]
    current_datetime = datetime.datetime.now()
    date_time_str = current_datetime.strftime("%d_%m_%Y")
    csv_file_name = f'{file_name}_{date_time_str}.csv'


    output_folder = '/Users/macintosh/Desktop/Thesis/ATGSVM/Test Model code/Main/'
    csv_file_path = output_folder + csv_file_name

    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for row in all_data:
            writer.writerow(row)   


# Calculate the average time for each method
avg_time_Hinge = np.mean(time_Hinge)
avg_time_GPIN = np.mean(time_GPIN)
avg_time_TPIN = np.mean(time_TPIN)
avg_time_ATGP = np.mean(time_ATGP)

print(f"Average time for Hinge SVM: {avg_time_Hinge:.4f} seconds")
print(f"Average time for GPIN SVM: {avg_time_GPIN:.4f} seconds")
print(f"Average time for TPIN SVM: {avg_time_TPIN:.4f} seconds")
print(f"Average time for ATGP SVM: {avg_time_ATGP:.4f} seconds")


