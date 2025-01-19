import numpy as np
from numpy.typing import NDArray
from PIL import Image
from skimage.util import view_as_windows
from google.colab import files

#ładowanie obrazu
def load_image(file_path: str) -> NDArray:
    """
    Wczytuje obraz w skali szarości z pliku.

    :param file_path: ścieżka do pliku obrazu
    :return: obraz jako 2D NDArray
    """
    with Image.open(file_path) as img:
        return np.array(img.convert("L"))

#zapisywanie obrazu
def save_image(image: NDArray, file_path: str):
    """
    Zapisuje obraz do pliku.

    :param image: obraz jako 2D NDArray
    :param file_path: ścieżka do pliku wynikowego
    """
    img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    img.save(file_path)

#funkcje jadrowe 2d
def kernel2d_sample_hold(t: NDArray) -> NDArray:
    x, y = t[:, 0], t[:, 1]
    return (x >= 0) * (x < 1) * (y >= 0) * (y < 1)


def kernel2d_najblizszy_sasiad(t: NDArray) -> NDArray:
    x, y = t[:, 0], t[:, 1]
    return (x >= (-1 / 2)) * (x < (1 / 2)) * (y >= (-1 / 2)) * (y < (1 / 2))


def kernel2d_liniowy(t: NDArray) -> NDArray:
    x, y = t[:, 0], t[:, 1]

    return ((1 - np.abs(x)) * (1 - np.abs(y))) * (np.abs(x) < 1) * (np.abs(y) < 1)


def kernel2d_sin(t: NDArray) -> NDArray:
    x, y = t[:, 0], t[:, 1]

    return  (np.abs(x) < np.inf) *  (np.abs(y) < np.inf) * (np.sinc(x) * np.sinc(y))

#upsampling przy uzyciu interpolacji
def image_interpolate2d(image: NDArray, ratio: int, kernel: callable) -> NDArray:


    # Rozmiar jądra
    w = 1.0

    # Rozmiar obrazu wynikowego
    target_shape = (image.shape[0] * ratio, image.shape[1] * ratio)

    # Siatka współrzędnych obrazu wejściowego
    image_grid_x, image_grid_y = np.meshgrid(
        np.arange(image.shape[1]), np.arange(image.shape[0])
    )
    image_grid = np.stack([image_grid_x.ravel(), image_grid_y.ravel()], axis=1)

    # Siatka współrzędnych dla interpolacji
    interpolate_grid_x, interpolate_grid_y = np.meshgrid(
        np.linspace(0, image.shape[1] - 1, target_shape[1]),
        np.linspace(0, image.shape[0] - 1, target_shape[0])
    )
    interpolate_grid = np.stack([interpolate_grid_x.ravel(), interpolate_grid_y.ravel()], axis=1)

    interpolated_image = np.zeros(target_shape)
    for point, value in zip(image_grid, image.ravel()):
        kernel_value = value * kernel((interpolate_grid-point)/w)
        interpolated_image += kernel_value.reshape(target_shape)

    return interpolated_image

#downsampling przy użyciu jądra uśredniającego
def downsample(image: NDArray, kernel_size: int = 2) -> np.ndarray:

    # Tworzenie jądra uśredniającego
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

    # Tworzenie okien
    windows = view_as_windows(image, window_shape=(kernel_size, kernel_size), step=kernel_size)

    downsampled_image = np.zeros(windows.shape[:2])  # Inicjalizacja wynikowego obrazu
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            downsampled_image[i, j] = np.sum(windows[i, j] * kernel)  # Operacja splotu

    return downsampled_image

# Funkcja wyliczająca mse
def mse(y_true, y_pred):
    return  np.mean((y_true - y_pred)*(y_true - y_pred))

print("Załaduj obraz do Colab")
uploaded = files.upload()  # Użytkownik przesyła plik

for file_name in uploaded.keys():
    print(f"Wczytywanie pliku: {file_name}")
    input_image = load_image(file_name)

    normal_file = "normal_image.png"
    upsampled_file = "upsampled_image.png"
    downsampled_file = "downsampled_image.png"
    up_and_down_sampled_file = "up_and_downsampled_image.png"
    upsampled_image = image_interpolate2d(input_image, ratio=2, kernel=kernel2d_sin)
    downsampled_image = downsample(input_image, 2)
    up_and_down_sampled_image = downsample(upsampled_image, 2)

    save_image(up_and_down_sampled_image, up_and_down_sampled_file)
    save_image(downsampled_image, downsampled_file)
    save_image(upsampled_image, upsampled_file)
    save_image(input_image, normal_file)

    # Pobranie pliku wynikowego
    files.download(weird_file)
    files.download(downsampled_file)
    files.download(upsampled_file)
    files.download(normal_file)
