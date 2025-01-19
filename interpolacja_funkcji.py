import matplotlib.pyplot as plt
import numpy as np


# Przykładowe funkcje
def f1(x):
  return np.sin(x)

def f2(x):
  return np.sin(1/x)

def f3(x):
  return np.sign(np.sin(8*x))

  # Interpolacja 1D
def interpolacja1d(x_interpolowany: np.array, x_mierzony: np.array, y_mierzony: np.array, kernel: callable) -> np.array:
  return y_mierzony @ [kernel((x_interpolowany-offset)/(x_mierzony[1]-x_mierzony[0])) for offset in x_mierzony]

# dla kazdej wartości w wektorze x_mierzony wykonuje funkcje kernela która puźniej jest mnorzona macierzowo z y mierzony

# Funkcje jądrowe

def kernel_sample_hold(t: np.array) -> np.array:
    # kernel interpolacji typu Sample hold
    return (t >= 0) * (t < 1)

def kernel_najblizszy_sasiad(t: np.array) -> np.array:
    # kernel interpolacji typu metoda najbliższego sąsiada
    return (t >= -1/2) * (t < 1/2)

def kernel_liniowy(t: np.array) -> np.array:
    # kernel interpolacji liniowej
    return (np.abs(t) <= 1) * (1-np.abs(t))

def kernel_sin(x: np.array) -> np.array:
    # kernel z interpolacją z sinusem
    return (np.abs(x) < np.inf) * np.sinc(x)

# Funkcja wyliczająca mse
def mse(y_true, y_pred):
    return  np.mean((y_true - y_pred)*(y_true - y_pred))

ils = 100
przewidywany = ils*10
x = np.linspace(-10, 10, ils)
x_przewidywany = np.linspace(-10, 10, przewidywany)
y = f3(x) #przypisujemy y wartosci jakiejś funkcji
y_prawdziwy = f3(x_przewidywany)
y_interpolowany = interpolacja1d(x_interpolowany=x_przewidywany, x_mierzony=x, y_mierzony=y, kernel=kernel_sin)



plt.figure(figsize=(14, 10))

plt.scatter(x, y, color='red', label='Punkty badane')
plt.plot(x_przewidywany, y_prawdziwy, label='Prawdziwa')
plt.plot(x_przewidywany, y_interpolowany, color='green', label='interpolacja')
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show



print(f"Mean Squared Error (MSE) : {mse(y_prawdziwy, y_interpolowany):.8f}")

