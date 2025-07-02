from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def construir_modelo(input_dim):
    modelo = Sequential()

    # Capa 1
    modelo.add(Dense(192, activation='relu', input_dim=input_dim))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.3))

    # Capa 2
    modelo.add(Dense(64, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.3))

    # Capa 3
    modelo.add(Dense(16, activation='relu'))

    # Salida
    modelo.add(Dense(1))  # salida: precio

    # Compilación
    opt = Adam(learning_rate=0.01)
    modelo.compile(optimizer=opt, loss='mse', metrics=['mae'])

    return modelo
