from preprocess import folder_of_images_to_array_of_images
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib


cat_folder_path = './Datasets/Petimages/Cat'
dog_folder_path = './Datasets/Petimages/Dog'
random_folder_path = './Datasets/data'
 
cats = folder_of_images_to_array_of_images(cat_folder_path, (160, 160))
dogs = folder_of_images_to_array_of_images(dog_folder_path, (160, 160))
random = folder_of_images_to_array_of_images(random_folder_path, (160, 160))
 
cats_label = ["Koty"] * cats.shape[0]  
dogs_label = ["Psy"] * dogs.shape[0]   
random_label = ["Losowe"] * random.shape[0]  
 
 
Search_space = np.vstack((cats, dogs, random))
labels = np.array(cats_label + dogs_label + random_label)
 
X_train, X_test, y_train, y_test = train_test_split(Search_space, labels, test_size=0.2, random_state=42)


#Debug

print("zacząłem liczyć")

log_reg_lbfsg_noregularization= LogisticRegression(
    max_iter=500, 
    random_state=42,
    solver='lbfgs'
)

log_reg_lbfsg_noregularization.fit(X_train, y_train)

#Debug

print("1 - policzone")

log_reg_saga_noregularization = LogisticRegression(
    solver='saga',
    max_iter=500,      
    random_state=42,
)

log_reg_saga_noregularization.fit(X_train, y_train)

#Debug

print("2 - policzone")
 
log_reg_saga_l1 = LogisticRegression(
    solver='saga',
    penalty='l1',             
    C=1.0,                    # Regularization strength
    max_iter=500,                 
    random_state=42
)

log_reg_saga_l1.fit(X_train, y_train)

#Debug

print("3 - policzone")
 
log_reg_saga_l2 = LogisticRegression(
    solver='saga',
    penalty='l2',             
    C=1.0,                    # Regularization strength
    max_iter=500,                  
    random_state=42
)

log_reg_saga_l2.fit(X_train, y_train)

#Debug

print("4 - policzone")
 
joblib.dump(log_reg_lbfsg_noregularization, 'calculated_models/log_reg_lbfsg_noregularization.joblib')
print('Trained and saved Logistic Regression model.')
joblib.dump(log_reg_saga_noregularization, 'calculated_models/log_reg_saga_noregularization.joblib')
print('Trained and saved Logistic Regression model.')
 
joblib.dump(log_reg_saga_l1, 'calculated_models/log_reg_saga_l1.joblib')
print('Trained and saved Logistic Regression model.')
joblib.dump(log_reg_saga_l2, 'calculated_models/log_reg_saga_l2.joblib')
print('Trained and saved Logistic Regression model.')
 

np.save('test_data/Logreg_X_test.npy', X_test)
np.save('test_data/Logreg_Y_test.npy', y_test)