import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

def read_data(filename):
    df = pd.read_csv(filename, delimiter=';', encoding='cp1251', decimal=',')
    return df

def build_model(layers_array, optimizer='adam', loss='mse'):
    model = keras.Sequential(layers_array)
    model.compile(optimizer=optimizer, loss=loss)
    return model

def learn_model(model, train_df, test_df, epochs=300, batch_size=32, verbose=0):
    history = model.fit(
        train_df.drop('Цена', axis=1), train_df['Цена'],
        validation_data=(test_df.drop('Цена', axis=1), test_df['Цена']),
        epochs=epochs, batch_size=batch_size, verbose=verbose
    )
    return history

def draw_scattering_diagram(test_df, test_predictions, model_number):
    plt.scatter(test_df['Цена'], test_predictions, label=f'Model {model_number}')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.show()

def draw_loss_diagram(history, model_number):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model {model_number} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()

def save_model(model, dir):
    model.save(dir)

def save_predictions(test_data, predicted_prices, dir):
    result = pd.DataFrame({
        'Цена': test_data['Цена'],
        'Цена_OUT': predicted_prices.flatten()
    })

    # AE
    result['AE'] = np.abs(result['Цена'] - result['Цена_OUT'])
    
    # SP
    result['SP'] = result['AE'] ** 2
    
    # APE
    result['APE'] = result['AE'] / result['Цена']
    
    # SPE
    result['SPE'] = result['APE'] ** 2

    result.to_excel(dir, index=False)

# Загрузка данных из файла
df = read_data('./lab1/kv.csv')

# Разделение данных на обучающую и тестовую выборки
train_df, test_df = train_test_split(df, test_size=0.2)

# Нормализация данных
scaler = StandardScaler()
train_df_scaled = scaler.fit_transform(train_df.drop('Цена', axis=1))
train_df_scaled = pd.DataFrame(train_df_scaled, columns=train_df.columns[:-1])
train_df_scaled['Цена'] = train_df['Цена'].values
test_df_scaled = scaler.transform(test_df.drop('Цена', axis=1))
test_df_scaled = pd.DataFrame(test_df_scaled, columns=test_df.columns[:-1])
test_df_scaled['Цена'] = test_df['Цена'].values


# Построение первой модели
model1 = build_model(
    layers_array = [
        layers.Dense(64, activation='relu', input_shape=[len(train_df.columns) - 1]),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ],
    optimizer='adam',
    loss='mse'
)
save_model(model1, './lab1/model1')

# Обучение первой модели
history1 = learn_model(model1, train_df_scaled, test_df_scaled, epochs=300, batch_size=32, verbose=0)

# Построение второй модели
model2 = build_model(
    layers_array = [
        layers.Dense(128, activation='linear', input_shape=[len(train_df.columns) - 1]),
        layers.Dense(64, activation='linear'),
        layers.Dense(1)
    ],
    optimizer='rmsprop',
    loss='mse'
)
save_model(model2, './lab1/model2')

# Обучение второй модели
history2 = learn_model(model2, train_df_scaled, test_df_scaled, epochs=300, batch_size=32, verbose=0)

# Получение предсказаний для тестовой выборки
test_predictions1 = model1.predict(test_df_scaled.drop('Цена', axis=1)).flatten()
test_predictions2 = model2.predict(test_df_scaled.drop('Цена', axis=1)).flatten()

save_predictions(test_df, test_predictions1, './lab1/predictions1.xlsx')
save_predictions(test_df, test_predictions2, './lab1/predictions2.xlsx')

# Построение диаграммы рассеивания первой модели
draw_scattering_diagram(test_df, test_predictions1, 1)

# Построение диаграммы рассеивания второй модели
draw_scattering_diagram(test_df, test_predictions2, 2)

# Оценка моделей
train_mse1 = model1.evaluate(train_df_scaled.drop('Цена', axis=1), train_df_scaled['Цена'], verbose=0)
test_mse1 = model1.evaluate(test_df_scaled.drop('Цена', axis=1), test_df_scaled['Цена'], verbose=0)
train_mse2 = model2.evaluate(train_df_scaled.drop('Цена', axis=1), train_df_scaled['Цена'], verbose=0)
test_mse2 = model2.evaluate(test_df_scaled.drop('Цена', axis=1), test_df_scaled['Цена'], verbose=0)

print(f'Model 1: train_mse={train_mse1:.4f}, test_mse={test_mse1:.4f}')
print(f'Model 2: train_mse={train_mse2:.4f}, test_mse={test_mse2:.4f}')

# График изменения функции потерь на обучающей и тестовой выборках для первой модели
draw_loss_diagram(history1, 1)

# График изменения функции потерь на обучающей и тестовой выборках для второй модели
draw_loss_diagram(history2, 2)