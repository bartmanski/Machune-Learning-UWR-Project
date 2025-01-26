import numpy as np
from xgboost import XGBClassifier
from preprocess import folder_of_images_to_array_of_images
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
cat_folder_path = './Datasets/Petimages/Cat'
dog_folder_path = './Datasets/Petimages/Dog'
random_folder_path = './Datasets/data'


cats=folder_of_images_to_array_of_images(cat_folder_path,(160,160))
dogs=folder_of_images_to_array_of_images(dog_folder_path,(160,160))
random=folder_of_images_to_array_of_images(random_folder_path,(160,160))

cats_label = [0] * cats.shape[0]
dogs_label = [1] * dogs.shape[0]
random_label = [2] * random.shape[0]

Search_space = np.vstack((cats,dogs,random))
labels = np.array(cats_label+dogs_label+random_label)

X_train, X_test, y_train, y_test = train_test_split(Search_space, labels, test_size=0.2 , random_state=42)

'''
    Czemu używamy hist a nie domyślengo exact:
    For small datasets (e.g., <10,000 rows), the computational overhead of exact is negligible, and the method benefits from its precision.
    For large datasets (e.g., >1M rows), exact becomes computationally prohibitive, and hist is far more practical.
    Why it's not the default: Users often start with small datasets during testing and experimentation, where exact might still be preferred.
'''

xgboost = XGBClassifier(
    #zapisywanie drzew jako histogramy   
    tree_method='hist',
    # jak dużo drzew w xd booscie
    n_estimators=750,
    # maksymalna głębokość drzewa      
    max_depth=5,
    # learning rate w booscie
    learning_rate=0.1,
    # na czym tworzymy drzewa (chce na losowych 80% żeby drzewa były bardzo różne) 
    subsample=0.8,
    # seed losowy żeby dało się powtórzyć wynik
    random_state=42
)

# ściaga labeli 
# 0- kot
# 1- pies
# 2- random

#Debug
print('zacząłem liczyć')

xgboost.fit(X_train,y_train)

#Debug
print('skończyłem liczyć')

xgboost.save_model('Models/calculated_models/xg_boost_750_160_model.json')

#Debug
print('już powinienem być zapisany')


'''
np.save('Models/test_data/Knn_X_test_160',X_test)
np.save('Models/test_data/Knn_Y_test_160',y_test)
'''
y_pred = xgboost.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")