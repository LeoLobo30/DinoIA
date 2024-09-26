import threading
from automation import *

def main():
    """Executa o loop principal do jogo."""
    open_dino_runner()

    thread = threading.Thread(target=capture_screen_loop)
    thread.daemon = True
    thread.start()

    for text in extract_text_from_image():
        if "COMEÇAR" or "COMEGAR" or "COMECAR" in text:
            jump()

    while True:
        for text in extract_text_from_image():
            if "A M E" or "M E" in text:
                restart_game() 

if __name__ == "__main__":
    main()
