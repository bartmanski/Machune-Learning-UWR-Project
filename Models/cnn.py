import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Przygotowanie danych
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# dataset treningowy (Cats / Dogs / Random)
train_generator = train_datagen.flow_from_directory(
    'Datasets/Data/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
# dataset walidacyjny (Cats / Dogs / Random)
validation_generator = validation_datagen.flow_from_directory(
    'Datasets/Data/validation',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Budowanie CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')  # Trzy klasy, aktywacja softmax 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("zaczynam trening cnn")

# Trening modelu

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(train_generator, epochs=20, validation_data=validation_generator, callbacks=[early_stopping])

print("trening zakończony")

# Zapisanie modelu
model.save('cat_dog_random_classifier.h5')

print("model zapisany")
