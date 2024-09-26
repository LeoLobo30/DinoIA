import cv2
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from PIL import Image
from io import BytesIO
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import easyocr

def capture_screen_loop():
        while True:
            capture_screen()

def open_dino_runner():
    """Abre o jogo Dino Runner usando Selenium."""
    global driver
    # Abre o Chrome usando o WebDriver
    driver = webdriver.Chrome()
    driver.get("https://dinorunner.com/pt/")
    driver.implicitly_wait(2)

def jump():
    """Simula um pulo no jogo pressionando a tecla 'Espaço'."""
    body = driver.find_element(by=By.TAG_NAME, value="body")
    body.send_keys(Keys.SPACE)

def restart_game():
    """Simula a ação de reiniciar o jogo pressionando 'Enter'."""
    body = driver.find_element(by=By.TAG_NAME, value="body")
    body.send_keys(Keys.RETURN)  # Ou Keys.SPACE se o jogo reiniciar com 'Espaço'

def capture_screen():
    try:
        canvas = driver.find_element(by=By.TAG_NAME, value="canvas")
        screenshot = Image.open(BytesIO(canvas.screenshot_as_png))
        screenshot.save("screenshot.png")
    except NoSuchElementException:
        pass
    return screenshot

def extract_text_from_image() -> list[str]:
    reader = easyocr.Reader(['pt', 'en'])
    """Extrai texto de uma imagem usando OCR (Tesseract)."""
    # Carrega a imagem usando OpenCV
    image = cv2.imread("screenshot.png")

    # Converte a imagem para escala de cinza para melhorar a precisão do OCR
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Opcional: Aplicar limiarização para melhorar o contraste (ajuda na detecção de texto)
    _, threshold_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    #remove a little part of upper the image
    #h, w = threshold_image.shape
    #threshold_image = threshold_image[20:h, 0:w]

    #salvar imagem redimensionada
    cv2.imwrite("resized_image.png", threshold_image)

    # Usa o pytesseract para extrair texto da imagem
    result = reader.readtext(threshold_image)

    list_text = []
    for detection in result:
        bbox, text, confidence = detection
        print(f'Texto detectado: {text}, Confiança: {confidence}')
        list_text.append(text)

    return list_text