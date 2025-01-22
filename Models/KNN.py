import numpy as np
from preprocess import folder_of_images_to_array_of_images
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib


cat_folder_path = './Datasets/Petimages/Cat'
dog_folder_path = './Datasets/Petimages/Dog'
random_folder_path = './Datasets/data'


cats=folder_of_images_to_array_of_images(cat_folder_path,(320,320))
dogs=folder_of_images_to_array_of_images(dog_folder_path,(320,320))
random=folder_of_images_to_array_of_images(random_folder_path,(320,320))

cats_label = ["Koty"] * cats.shape[0]
dogs_label = ["Psy"] * dogs.shape[0]
random_label = ["Losowe"] * random.shape[0]

Search_space = np.vstack((cats,dogs,random))
labels = np.array(cats_label+dogs_label+random_label)

X_train, X_test, y_train, y_test = train_test_split(Search_space, labels, test_size=0.2 , random_state=42)

k = 1
knn1 = KNeighborsClassifier(n_neighbors=k)
knn1.fit(X_train, y_train)
print('policzyłem 1')
k = 5
knn5 = KNeighborsClassifier(n_neighbors=k)
knn5.fit(X_train, y_train)
print('policzyłem 5')
k = 10
knn10 = KNeighborsClassifier(n_neighbors=k)
knn10.fit(X_train, y_train)
print('policzyłem 10')
joblib.dump(knn1, 'Models/calculated_models/knn1_model.joblib')
joblib.dump(knn5, 'Models/calculated_models/knn5_model.joblib')
joblib.dump(knn10, 'Models/calculated_models/knn10_model.joblib')

np.save('Models/test_data/Knn_X_test',X_test)
np.save('Models/test_data/Knn_Y_test',y_test)

