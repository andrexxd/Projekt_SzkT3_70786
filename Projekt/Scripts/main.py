import cv2
import multiprocessing as mp
from functools import wraps
import os
import numpy as np
import threading
from tqdm import tqdm
import subprocess
import time
import imageio_ffmpeg
from queue import Empty
import sys
import glob

# ---------- dekoratory ----------
def filter(order=0):
    def decorator(func):
        func._is_filter = True
        func._order = order
        return func
    return decorator

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[TIMER] {func.__name__} wykonana w {elapsed:.4f} s")
        return result
    return wrapper

# ---------- proces roboczy ----------

def process_chunk(jpeg_chunk, filters, counter, lock):
    """Otrzymuje listę bajtów JPEG. Dekoduje, stosuje filtry, ponownie koduje do JPEG."""
    processed = []
    for jpeg_bytes in jpeg_chunk:
        # Dekompresja do surowego BGR
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        # Zastosuj filtry (np. pencilSketch)
        for f in filters:
            frame = f(frame)
        # Ponowna kompresja wyniku
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if success:
            processed.append(buffer.tobytes())
        else:
            processed.append(b'')
    with lock:
        counter.value += len(processed)
    return processed

# ---------- metaklasa ----------
class VideoPipelineMeta(type):
    def __new__(mcs, name, bases, namespace):
        filters = []
        for attr_name, attr_value in namespace.items():
            if callable(attr_value) and hasattr(attr_value, '_is_filter'):
                filters.append(attr_value)
        filters.sort(key=lambda f: f._order)

        @timed
        def run(self):
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                raise IOError(f"Nie można otworzyć pliku: {self.input_path}")

            # Dynamiczna nazwa pliku wyjściowego
            input_basename = os.path.basename(self.input_path)
            output_dir = "Video_Out"
            os.makedirs(output_dir, exist_ok=True)
            self.output_path = os.path.join(output_dir, input_basename)

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            chunk_size = getattr(self, 'chunk_size', 150) #Domyślna wartość chunk_size jeśli nie określono w klasie MyVideoProcesor
            num_workers = getattr(self, 'num_workers', 8) #Domyślna wartość num_workers jeśli nie określono w klasie MyVideoProcesor



            # ---------- Pierwsza klatka (ustalenie wymiarów) ----------
            ret, first_frame = cap.read()
            if not ret:
                cap.release()
                return
            sample = first_frame.copy()
            for f in self._filters:
                sample = f(sample)
            # # Upewniamy się, że mamy obraz 3-kanałowy BGR (wymagane przez VideoWriter)
            if len(sample.shape) == 2:
                sample = cv2.cvtColor(sample, cv2.COLOR_GRAY2BGR)
            out_h, out_w = sample.shape[:2]

            # Kompresja pierwszej klatki (dla spójności z resztą)
            _, sample_jpg = cv2.imencode('.jpg', sample, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if _:
                sample = cv2.imdecode(sample_jpg, cv2.IMREAD_COLOR)

            # Tymczasowy plik wyjściowy (bez dźwięku)
            base, _ = os.path.splitext(self.output_path)
            temp_output = base + 'temp.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (out_w, out_h))
            out.write(sample)

            pbar = tqdm(total=total_frames, desc="Przetwarzanie", unit="klatka")
            pbar.update(1)

            stop_pbar = threading.Event()
            def pbar_updater():
                while not stop_pbar.is_set():
                    # if frame_counter.value > pbar.n:
                    pbar.update(frame_counter.value - pbar.n)
                    stop_pbar.wait(1.0)



            result_queue = mp.Queue()
            pool = mp.Pool(processes=num_workers)

            # Bufor na wyniki (id paczki -> lista bajtów JPEG)
            buffer = {}
            next_to_write = 1
            chunk_id = 0

            manager = mp.Manager()
            frame_counter = manager.Value('i', 0)
            counter_lock = manager.Lock()

            def read_compressed_chunk():
                """Odczytuje do chunk_size klatek i od razu kompresuje je do JPEG."""
                chunk = []
                for _ in range(chunk_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    success, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if success:
                        chunk.append(jpeg.tobytes())
                return chunk

            current_chunk = read_compressed_chunk()
            if current_chunk:
                chunk_id += 1
                pool.apply_async(process_chunk, (current_chunk, self._filters, frame_counter, counter_lock),
                                 callback=lambda res, cid=chunk_id: result_queue.put((cid, res)))


            updater = threading.Thread(target=pbar_updater, daemon=True)
            updater.start()


            # ---------- Główna pętla ----------
            while next_to_write <= chunk_id or not result_queue.empty() or current_chunk:

                # 1. Odbierz gotowe paczki
                while not result_queue.empty():
                    try:
                        cid, processed = result_queue.get_nowait()
                        buffer[cid] = processed
                    except Empty:
                        break

                # 2. Zapisuj kolejne paczki w kolejności
                while next_to_write in buffer:
                    for frame_bytes in buffer[next_to_write]:
                        nparr = np.frombuffer(frame_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            out.write(frame)
                    del buffer[next_to_write]
                    next_to_write += 1


                # 3. Wczytaj nową paczkę, jeśli jest miejsce w buforze
                if current_chunk and len(buffer) < num_workers * 2:
                    next_chunk = read_compressed_chunk()
                    if next_chunk:
                        chunk_id += 1
                        pool.apply_async(process_chunk, (next_chunk, self._filters, frame_counter, counter_lock),
                                         callback=lambda res, cid=chunk_id: result_queue.put((cid, res)))
                        current_chunk = next_chunk
                    else:
                        current_chunk = None   # koniec pliku

                # 4. Małe opóźnienie, aby nie katować CPU
                time.sleep(0.01)

            stop_pbar.set()
            updater.join()
            if pbar.n < total_frames:
                pbar.update(frame_counter.value - pbar.n)

            pbar.close()
            pool.close()
            pool.join()
            cap.release()
            out.release()
            cv2.destroyAllWindows()

            # ---------- Dodanie ścieżki dźwiękowej (ffmpeg) ----------
            print("\nDodawanie ścieżki dźwiękowej...")
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, '-i', temp_output, '-i', self.input_path,
                '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0?',
                '-y', self.output_path
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"Zapisano finalny plik z dźwiękiem: {self.output_path}")
                os.remove(temp_output)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Nie udało się dodać dźwięku: {e.stderr if e.stderr else e}")
                print(f"Plik bez dźwięku pozostał jako: {temp_output}")

            manager.shutdown()

        namespace['run'] = run
        namespace['_filters'] = filters

        return super().__new__(mcs, name, bases, namespace)


# ---------- klasa użytkownika ----------
class MyVideoProcessor(metaclass=VideoPipelineMeta):
    chunk_size = 150
    num_workers = 8

    @filter(order=1)
    def pencil_sketch_manual(frame):
        # 1. Skala szarości
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 2. Negatyw
        inverted = cv2.bitwise_not(gray)
        # 3. Rozmycie negatywu (im większy kernel, tym grubsze linie)
        blurred = cv2.GaussianBlur(inverted, (95, 93), sigmaX=0, sigmaY=0)
        # 4. Rozjaśnianie (color dodge) – podzielenie szarości przez odwrócony rozmyty
        #    Używamy dzielenia float, aby uniknąć obcięcia wartości
        sketch = cv2.divide(gray, 255 - blurred, scale=256.0)
        # 5. Konwersja do 8-bit i na 3 kanały BGR
        sketch_8u = np.uint8(sketch)
        return cv2.cvtColor(sketch_8u, cv2.COLOR_GRAY2BGR)


    # Przygotowane filtry do wyboru:

    # @filter(order=1)
    # def sepia(frame):
    #     """Nadaje obrazowi ciepły, sepiowy odcień."""
    #     # Macierz transformacji kolorów dla efektu sepii
    #     kernel = np.array([
    #         [0.272, 0.534, 0.131],
    #         [0.349, 0.686, 0.168],
    #         [0.393, 0.769, 0.189]
    #     ])
    #     sepia_frame = cv2.transform(frame, kernel)
    #     # Upewnienie, że wartości są w zakresie 0-255
    #     sepia_frame = np.clip(sepia_frame, 0, 255).astype(np.uint8)
    #     return sepia_frame
    #
    # @filter(order=1)
    # def negative(frame):
    #     """Odwraca kolory, tworząc negatyw."""
    #     return cv2.bitwise_not(frame)
    #
    # @filter(order=1)
    # def canny_edges(frame):
    #     """Wykrywa krawędzie metodą Canny'ego, zwraca obraz BGR (białe krawędzie na czarnym tle)."""
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     edges = cv2.Canny(gray, 100, 200)
    #     # Konwersja z obrazu binarnego na 3-kanałowy BGR
    #     edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    #     return edges_bgr
    #
    # @filter(order=1)
    # def cartoon(frame):
    #     """Efekt kreskówki poprzez stylizację i podkreślenie krawędzi."""
    #     # Lekkie rozmycie przed stylizacją
    #     blurred = cv2.medianBlur(frame, 5)
    #     # Użyj stylizacji z biblioteki OpenCV (funkcja stylization)
    #     styled = cv2.stylization(blurred, sigma_s=60, sigma_r=0.6)
    #     # Wykryj krawędzie na oryginalnym obrazie
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     edges = cv2.adaptiveThreshold(
    #         cv2.medianBlur(gray, 7), 255,
    #         cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2
    #     )
    #     edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    #     # Połącz stylizowany obraz z krawędziami
    #     cartoon = cv2.bitwise_and(styled, edges_bgr)
    #     return cartoon
    #
    # @filter(order=1)
    # def oil_painting(frame):
    #     """Daje efekt malarstwa olejnego (wymaga OpenCV z modułem ximgproc, jeśli niedostępne, użyj prostszej symulacji)."""
    #     try:
    #         oil = cv2.bilateralFilter(frame, 15, 80, 80)
    #         return oil
    #     except ImportError:
    #         # Alternatywna implementacja przybliżona
    #         blurred = cv2.medianBlur(frame, 7)
    #         # Kwantyzacja kolorów - redukcja palety
    #         quantized = (blurred // 64) * 64
    #         return quantized
    #
    # @filter(order=1)
    # def gaussian_blur(frame):
    #     """Rozmycie Gaussa (jako przykład delikatnego wygładzenia)."""
    #     return cv2.GaussianBlur(frame, (15, 15), 0)
    #
    # @filter(order=1)
    # def sharpen(frame):
    #     """Wyostrzenie obrazu za pomocą filtra unsharp mask."""
    #     blurred = cv2.GaussianBlur(frame, (0, 0), 3)
    #     sharpened = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
    #     return sharpened



if __name__ == "__main__":
    input_dir = "Video_In"
    if not os.path.isdir(input_dir):
        print(f"Foler {input_dir} nie istnieje")
        sys.exit(1)

    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not video_files:
        print(f"Brak plików wideo w folderze {input_dir}")
        sys.exit(1)

    print(f"Liczba znalezionych plików do przetworzenia: {len(video_files)}")

    for f in video_files:
        print(f" - {f}")

    p = MyVideoProcessor()
    for video_path in video_files:
        print(f"\n>>> Przetwarzanie: {video_path}")
        p.input_path = video_path

        p.run()
        print(f">>> Zakończono: {os.path.basename(video_path)}")
    print("Wszystkie pliki przetworzone!")