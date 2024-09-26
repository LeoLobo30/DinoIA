import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
from keras._tf_keras.keras.applications import ResNet50
from keras._tf_keras.keras.layers import Dense, GlobalAveragePooling2D
from keras._tf_keras.keras.models import Model
import numpy as np
from PIL import Image

def createModel():
  model = models.Sequential()

  # Camada de input
  model.add(layers.Input(shape=(150, 150, 3)))

  # Primeira camada de convolução
  model.add(layers.Conv2D(32, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))

  # Segunda camada de convolução
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))

  # Terceira camada de convolução
  model.add(layers.Conv2D(128, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))

  # Quarta camada de convolução
  model.add(layers.Conv2D(128, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))

  # Camada totalmente conectada
  model.add(layers.Flatten())
  model.add(layers.Dense(512, activation='relu'))

  # Camada de saída (número de classes depende do seu dataset)
  model.add(layers.Dense(3, activation='softmax'))

  return model

def loadModel():
  model = tf.keras.models.load_model('model.keras')
  return model

def trainModel():
  # Caminho para os dados de treino e validação
  train_dir = 'dataset/train'
  validation_dir = 'dataset/validation'

  # Geradores de dados com aumento de dados (data augmentation)
  train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )

  validation_datagen = ImageDataGenerator(rescale=1./255)

  # Gerar imagens com tamanhos padronizados
  train_generator = train_datagen.flow_from_directory(
      train_dir,
      target_size=(150, 150),
      batch_size=32,
      class_mode='categorical'  # Para classificação multiclasse
  )

  validation_generator = validation_datagen.flow_from_directory(
      validation_dir,
      target_size=(150, 150),
      batch_size=32,
      class_mode='categorical'
  )

  # pode ser trocado
  #model = loadModel()
  #model = createModel()
  model = trainModelBaseInModel()

  # Compilar o modelo
  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  # Ajuste os hiperparâmetros
  epochs = 47
  steps_per_epoch = train_generator.samples // train_generator.batch_size
  validation_steps = validation_generator.samples // validation_generator.batch_size

  # Treinamento do modelo
  history = model.fit(
      train_generator,
      steps_per_epoch=20,
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps=validation_steps
  )

  test_loss, test_acc = model.evaluate(train_generator, verbose=2)
  print(f"Test accuracy: {test_acc}")

  # Salvar o modelo
  model.save('model.keras')

def trainModelBaseInModel() -> Model:
  # Carregar o modelo ResNet50 pré-treinado
  base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

  # Congelar as camadas iniciais
  for layer in base_model.layers:
      layer.trainable = False

  # Adicionar camadas para a sua tarefa
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  predictions = Dense(3, activation='softmax')(x)

  # Criar o modelo final
  return Model(inputs=base_model.input, outputs=predictions)



def predictClass(model: Model, img_path: str) -> int:
  """
  Dado um modelo e um caminho de imagem, retorna a classe prevista
  """  
  # Carregar a imagem
  img = Image.open(img_path)
  img = img.resize((150, 150))
  img_array = np.array(img)
  img_array = img_array / 255.0
  img_array = np.expand_dims(img_array, 0)

  # Fazer a previsao
  predictions = model.predict(img_array)
  return np.argmax(predictions)

class_names = ["cactos", "dinossauro", "passaro"]

previsao = predictClass(loadModel(), 'dataset/train/cactos/cacto.png')

print('Classe prevista: ' + str(previsao) + '/nclasse real: ' + class_names[previsao])