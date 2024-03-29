import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(400, 400, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(26, activation='softmax'))  


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


datagen = ImageDataGenerator(rescale=1./255,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             validation_split=0.2)  


train_generator = datagen.flow_from_directory(
    'data',
    target_size=(400, 400),  
    batch_size=60,
    class_mode='categorical',
    subset='training',
    interpolation = 'nearest'
)

validation_generator = datagen.flow_from_directory(
    'data',
    target_size=(400, 400),  
    batch_size=60,
    class_mode='categorical',  
    subset='validation',
    interpolation = 'nearest'
)


history = model.fit(train_generator,
                    epochs=10,  
                    validation_data=validation_generator)


model.save('trainedModel.h5')  


