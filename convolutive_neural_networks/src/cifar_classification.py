import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras import optimizers
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2

# -----------------------------------------------------------------------------
# PARÁMETROS
# -----------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 50
PATIENCE_ES = 3
PATIENCE_LR = 2
INITIAL_LR_FACTOR = 0.5
IMAGE_SIZE = (128, 128)

# -----------------------------------------------------------------------------
# GENERADOR DE AUMENTO DE DATOS
# -----------------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
valid_datagen = ImageDataGenerator(rescale=1./255)

# -----------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# -----------------------------------------------------------------------------
def cifar_classification(data_dir: Path):
    """
    data_dir debe apuntar a la carpeta que contiene 'train', 'valid' y 'test'.
    Ejemplo:
      D:/.../gesture_recognition/data
    """
    # Definir rutas de directorios
    base_dir = Path(data_dir)
    train_dir = base_dir / 'train'
    valid_dir = base_dir / 'valid'
    test_dir = base_dir / 'test'

    # Comprobar existencia de directorios
    for d in (train_dir, valid_dir, test_dir):
        if not d.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {d}")

    # Generadores de datos desde directorio
    train_gen = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    valid_gen = valid_datagen.flow_from_directory(
        directory=valid_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    test_gen = valid_datagen.flow_from_directory(
        directory=test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    n_classes = train_gen.num_classes

    # -------------------------------------------------------------------------
    # TRANSFER LEARNING: MobileNetV2
    # -------------------------------------------------------------------------
    base_model = MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.summary()

    # Compilar
    model.compile(
        optimizer=optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # -------------------------------------------------------------------------
    # CALLBACKS
    # -------------------------------------------------------------------------
    timestamp = int(datetime.now().timestamp())
    save_dir = Path(__file__).parent.parent / 'trained_model_parameters'
    save_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=PATIENCE_ES,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=INITIAL_LR_FACTOR,
            patience=PATIENCE_LR,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(save_dir / f'best_model_{timestamp}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # -------------------------------------------------------------------------
    # ENTRENAMIENTO
    # -------------------------------------------------------------------------
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=valid_gen,
        validation_steps=valid_gen.samples // BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )

    # -------------------------------------------------------------------------
    # GUARDAR Y EVALUAR
    # -------------------------------------------------------------------------
    final_path = save_dir / 'final_model.h5'
    model.save(str(final_path))
    print(f"Modelo final guardado en: {final_path}")

    # Verificar que el número de pasos no sea 0
    steps = test_gen.samples // BATCH_SIZE
    if steps == 0:
        steps = 1  # Asegúrate de que siempre sea al menos 1

    # Evaluar el modelo en el conjunto de prueba
    loss, acc = model.evaluate(test_gen, steps=steps, verbose=2)
    print(f'Test loss: {loss:.4f}, Test accuracy: {acc:.4f}')

# -----------------------------------------------------------------------------
# EJECUCIÓN
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python main.py <ruta_al_directorio_data>")
        sys.exit(1)
    data_path = sys.argv[1]
    cifar_classification(Path(data_path))
