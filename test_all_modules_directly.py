"""
Тестовый скрипт для проверки Face Mesh pipeline с Head Pose и EAR
"""

import cv2
import time
import mediapipe as mp
from collections import deque
from datetime import datetime

from face_detection_and_emotion_recognition import EmotionRecognizer
from analyze_head_pose import HeadPoseEstimator, classify_attention_state, HEAD_POSE_LANDMARKS
from analyze_ear import EyeAspectRatioAnalyzer, classify_attention_by_ear, LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS

# Инициализация MediaPipe (в этом скрипте отдельная реализация детектора лица через Face Mesh)
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Ключевые landmarks (точки от Face Mesh) для визуализации (Head Pose + Eyes)
KEY_LANDMARKS = HEAD_POSE_LANDMARKS + LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS


class FaceMeshAnalyzer:
    """
    Анализатор с Face Mesh (детекция лица - отдельная реализация данного скрипта), 
    Head Pose, EAR и Emotion Recognition
    """

    def __init__(self, device='cpu'):
        """
        Args:
            device: 'cpu' или 'cuda' для emotion recognizer
        """
        print("Инициализация FaceMeshAnalyzer...")

        # Детектор и точки
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Анализаторы
        self.head_pose_estimator = HeadPoseEstimator()
        self.ear_analyzer = EyeAspectRatioAnalyzer(ear_threshold=0.25, consec_frames=3)
        self.emotion_recognizer = EmotionRecognizer(
            device=device,
            window_size=15,
            confidence_threshold=0.55,
            ambiguity_threshold=0.15
        )

        # Состояние визуализации точек (визуализация+дебаг): 0=off, 1=key points, 2=all points
        self.landmark_viz_mode = 0

        print("FaceMeshAnalyzer инициализирован")

    def _get_face_bbox(self, face_landmarks, image_width, image_height, margin=20):
        """
        Вычисляет bounding box лица из landmarks.

        Args:
            face_landmarks: Face Mesh landmarks
            image_width: Ширина изображения
            image_height: Высота изображения
            margin: Отступ вокруг bbox

        Returns:
            (x1, y1, x2, y2) координаты bbox
        """
        # Все координаты landmarks
        x_coords = [lm.x * image_width for lm in face_landmarks.landmark]
        y_coords = [lm.y * image_height for lm in face_landmarks.landmark]

        # min/max координат
        x_min = int(min(x_coords))
        x_max = int(max(x_coords))
        y_min = int(min(y_coords))
        y_max = int(max(y_coords))

        # Добавление отступа (margin) к рамке bbox
        x1 = max(0, x_min - margin)
        y1 = max(0, y_min - margin)
        x2 = min(image_width, x_max + margin)
        y2 = min(image_height, y_max + margin)

        return x1, y1, x2, y2

    def _crop_face(self, image, bbox):
        """
        Вырезает лицо из изображения по bbox.

        Args:
            image: Исходное изображение
            bbox: (x1, y1, x2, y2)

        Returns:
            Вырезанное изображение лица
        """
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]


    def _draw_landmarks(self, image, face_landmarks, image_width, image_height):
        """
        Рисует landmarks в зависимости от выбранного режима визуализации.

        Args:
            image: Изображение для рисования
            face_landmarks: Face Mesh landmarks
            image_width: Ширина изображения
            image_height: Высота изображения
        """

        if self.landmark_viz_mode == 0:                 # off - пропуск, возврат из метода
            return
        elif self.landmark_viz_mode == 1:               # key - только ключевые точки (+ вокруг глаз)
            for idx in KEY_LANDMARKS:
                lm = face_landmarks.landmark[idx]
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        elif self.landmark_viz_mode == 2:               # all - все точки Face Mesh
            for lm in face_landmarks.landmark:
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)


    def toggle_landmark_visualization(self):
        """Переключает режим визуализации landmarks: off -> key -> all -> off"""
        
        self.landmark_viz_mode = (self.landmark_viz_mode + 1) % 3
        mode_names = ["OFF", "KEY POINTS", "ALL POINTS"]
        print(f"Landmark visualization: {mode_names[self.landmark_viz_mode]}")

    def analyze_frame(self, image):
        """
        Анализирует кадр: Face Mesh, Head Pose, EAR, Emotion.

        Args:
            image: Входное BGR изображение

        Returns:
            Словарь с результатами и аннотированное изображение
        """
        h, w = image.shape[:2]
        annotated_image = image.copy()

        # Обработка Face Mesh
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        mesh_results = self.face_mesh.process(rgb_image)
        rgb_image.flags.writeable = True

        results = {
            'face_detected': False,
            'bbox': None,
            'emotion': None,
            'emotion_confidence': None,
            'head_pose': None,
            'eyes': None
        }

        if mesh_results.multi_face_landmarks:
            face_landmarks = mesh_results.multi_face_landmarks[0]

            # 1) Получение bounding box
            bbox = self._get_face_bbox(face_landmarks, w, h, margin=20)
            results['bbox'] = bbox

            # 2) Вырезание лица для emotion recognition
            face_crop = self._crop_face(image, bbox)

            # 3) Распознавание эмоции
            if face_crop.size > 0:
                emotion, conf = self.emotion_recognizer.predict(face_crop)
                results['emotion'] = emotion
                results['emotion_confidence'] = conf

            # 4) Вычисление Head Pose Estimation 
            head_pose = self.head_pose_estimator.estimate(face_landmarks, w, h)
            if head_pose:
                results['head_pose'] = head_pose
                results['head_pose']['attention_state'] = classify_attention_state(
                    head_pose['pitch'], head_pose['yaw'], head_pose['roll']
                )

            # 5) Вычисление Eye Aspect Ratio (EAR)
            # по умолчанию задан ID лица с индексом 0 для отслеживания в историю
            ear_result = self.ear_analyzer.analyze(face_landmarks, w, h, face_id=0)
            if ear_result:
                # Расчёт частоты моргания (предполагаемые ~30 FPS)
                # TODO: уточнить метрику
                blink_rate = ear_result['blink_count'] * 2
                results['eyes'] = ear_result
                results['eyes']['attention_state'] = classify_attention_by_ear(ear_result['avg_ear'], blink_rate)

            results['face_detected'] = True

            # 6) Визуализация
            self._visualize_results(annotated_image, results, face_landmarks, w, h)

        return results, annotated_image


    def _visualize_results(self, image, results, face_landmarks, w, h):
        """
        Визуализирует результаты анализа на изображении.

        Args:
            image: Изображение для рисования
            results: Словарь с результатами анализа
            face_landmarks: Face Mesh landmarks
            w, h: Размеры изображения
        """
        # Рисование landmarks
        self._draw_landmarks(image, face_landmarks, w, h)

        # Рисование bounding box
        if results['bbox']:
            x1, y1, x2, y2 = results['bbox']
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Текст с результатами
            y_offset = y1 - 10

            # Эмоция
            if results['emotion']:
                emotion_text = f"{results['emotion']}: {results['emotion_confidence']:.2f}"
                cv2.putText(image, emotion_text, (x1, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                y_offset -= 25

            # Head Pose
            if results['head_pose']:
                hp = results['head_pose']
                pose_text = f"P:{hp['pitch']:.1f} Y:{hp['yaw']:.1f} R:{hp['roll']:.1f}"
                cv2.putText(image, pose_text, (x1, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_offset -= 20

                attention_text = f"Att: {hp['attention_state']}"
                cv2.putText(image, attention_text, (x1, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_offset -= 20

            # EAR
            if results['eyes']:
                eyes = results['eyes']
                ear_text = f"EAR:{eyes['avg_ear']:.2f} Blinks:{eyes['blink_count']}"
                cv2.putText(image, ear_text, (x1, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                y_offset -= 20

                eye_state_text = f"Eyes: {eyes['attention_state']}"
                cv2.putText(image, eye_state_text, (x1, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    def close(self):
        """Освобождение ресурсов (закрытие анализатора)"""
        self.face_mesh.close()


# Доп. логирование в файл
def log_frame_results(log_file, frame_count, results, fps, timestamp):
    """
    Записывает результаты анализа кадра в лог-файл.

    Args:
        log_file: Открытый файл для записи
        frame_count: Номер кадра
        results: Словарь с результатами анализа
        fps: Текущий FPS
        timestamp: Время обработки кадра
    """
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"Frame: {frame_count} | Time: {timestamp} | FPS: {fps:.2f}\n")
    log_file.write(f"{'='*80}\n")

    if results['face_detected']:
        # Bounding box
        if results['bbox']:
            x1, y1, x2, y2 = results['bbox']
            log_file.write(f"Face BBox: ({x1}, {y1}) -> ({x2}, {y2})\n")

        # Эмоция
        if results['emotion']:
            log_file.write(f"\n[EMOTION]\n")
            log_file.write(f"  Detected: {results['emotion']}\n")
            log_file.write(f"  Confidence: {results['emotion_confidence']:.4f}\n")

        # Head Pose
        if results['head_pose']:
            hp = results['head_pose']
            log_file.write(f"\n[HEAD POSE]\n")
            log_file.write(f"  Pitch (up/down): {hp['pitch']:>7.2f}°\n")
            log_file.write(f"  Yaw (left/right): {hp['yaw']:>7.2f}°\n")
            log_file.write(f"  Roll (tilt): {hp['roll']:>7.2f}°\n")
            log_file.write(f"  Attention State: {hp['attention_state']}\n")

        # Eyes / EAR
        if results['eyes']:
            eyes = results['eyes']
            log_file.write(f"\n[EYE ANALYSIS]\n")
            log_file.write(f"  Left EAR: {eyes['left_ear']:.4f}\n")
            log_file.write(f"  Right EAR: {eyes['right_ear']:.4f}\n")
            log_file.write(f"  Average EAR: {eyes['avg_ear']:.4f}\n")
            log_file.write(f"  Eyes Open: {eyes['eyes_open']}\n")
            log_file.write(f"  Is Blinking: {eyes['is_blinking']}\n")
            log_file.write(f"  Total Blinks: {eyes['blink_count']}\n")
            log_file.write(f"  Attention State: {eyes['attention_state']}\n")
    else:
        log_file.write("No face detected\n")

    log_file.flush()  # Выгрузка (запись) на диск


def main():
    print("Запуск Face Mesh Test Pipeline...")
    print("\nУправление:")
    print("  ESC - выход")
    print("  T   - переключение визуализации landmarks (off -> key -> all -> off)")
    print("  Q   - выключить визуализацию landmarks")
    print("  SPACE - скриншот")

    # Создание лог-файла с timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"test_log_{timestamp_str}.txt"
    log_file = open(log_filename, 'w', encoding='utf-8')

    # Заголовок лога
    log_file.write("="*80 + "\n")
    log_file.write("Face Mesh Test Pipeline - Analysis Log\n")
    log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("="*80 + "\n")
    log_file.flush()

    print(f"\nЛог записывается в файл: {log_filename}")

    # Проверка CUDA
    device = 'cpu'
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            device = 'cuda'
            print(f"CUDA доступна, используем GPU")
            log_file.write(f"\nDevice: CUDA (GPU)\n")
        else:
            print(f"CUDA недоступна, используем CPU")
            log_file.write(f"\nDevice: CPU\n")
    except:
        print(f"CUDA недоступна, используем CPU")
        log_file.write(f"\nDevice: CPU\n")

    log_file.write("="*80 + "\n\n")
    log_file.flush()

    # Инициализация анализатора
    analyzer = FaceMeshAnalyzer(device=device)

    # Открытие веб-камеры
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        log_file.write("\nERROR: Failed to open camera\n")
        log_file.close()
        return

    # FPS трекинг
    fps_history = deque(maxlen=30)

    try:
        frame_count = 0
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # Анализ кадра
            results, annotated = analyzer.analyze_frame(frame)

            # FPS
            fps = 1.0 / (time.time() - start_time)
            fps_history.append(fps)
            avg_fps = sum(fps_history) / len(fps_history)

            # Отображение FPS
            cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Отображение режима landmarks
            mode_names = ["OFF", "KEY", "ALL"]
            viz_mode_text = f"Landmarks: {mode_names[analyzer.landmark_viz_mode]}"
            cv2.putText(annotated, viz_mode_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Вывод детальной информации в консоль каждую секунду (~30 FPS, т.е. каждые 30 кадров)
            if frame_count % 30 == 0 and results['face_detected']:
                print(f"\n--- Frame {frame_count} ---")
                if results['emotion']:
                    print(f"Emotion: {results['emotion']} ({results['emotion_confidence']:.2f})")
                if results['head_pose']:
                    hp = results['head_pose']
                    print(f"Head Pose: pitch={hp['pitch']:.1f}° yaw={hp['yaw']:.1f}° roll={hp['roll']:.1f}°")
                    print(f"Attention (pose): {hp['attention_state']}")
                if results['eyes']:
                    eyes = results['eyes']
                    print(f"EAR: {eyes['avg_ear']:.3f} (L:{eyes['left_ear']:.3f} R:{eyes['right_ear']:.3f})")
                    print(f"Eyes open: {eyes['eyes_open']}, Blinks: {eyes['blink_count']}")
                    print(f"Attention (eyes): {eyes['attention_state']}")

            # Логирование в файл каждые 30 кадров
            if results['face_detected'] and  frame_count % 30 == 0:
                current_timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                log_frame_results(log_file, frame_count, results, avg_fps, current_timestamp)

            # Отображение
            window_name = 'Face Mesh Test | ESC - quit | T - point mods | Q - off points | SPACE - screenshot'
            cv2.imshow(window_name, annotated)

            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('t') or key == ord('T'):  # T - toggle landmarks
                analyzer.toggle_landmark_visualization()
            elif key == ord('q') or key == ord('Q'):  # Q - disable landmarks
                analyzer.landmark_viz_mode = 0
                print("Landmark visualization: OFF")
            elif key == 32:  # SPACE - screenshot
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"Скриншот сохранён: {filename}")

            frame_count += 1

    finally:
        print("\nЗавершение работы...")

        # Закрытие лог-файла с финальной записью
        log_file.write("\n" + "="*80 + "\n")
        log_file.write(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("="*80 + "\n")
        log_file.close()
        print(f"Лог сохранён: {log_filename}")

        # cleanup
        analyzer.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Работа завершена успешно!")


if __name__ == '__main__':
    main()
