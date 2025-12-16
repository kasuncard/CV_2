#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Демонстрационный скрипт для системы отслеживания объектов
"""

import cv2
import numpy as np
import argparse
import os
import sys

class SimpleObjectTracker:
    """
    Упрощенная версия трекера для демонстрации
    """
    
    def __init__(self, min_matches=10):
        self.min_matches = min_matches
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.object_kp = None
        self.object_desc = None
        self.object_bbox = None
        self.is_initialized = False
    
    def initialize(self, frame, bbox):
        """Инициализация объекта"""
        x, y, w, h = bbox
        
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            return False
        
        object_roi = frame[y:y+h, x:x+w]
        self.object_kp, self.object_desc = self.orb.detectAndCompute(object_roi, None)
        
        if len(self.object_kp) < self.min_matches:
            return False
        
        self.object_bbox = bbox
        self.is_initialized = True
        print(f"Объект инициализирован: {len(self.object_kp)} ключевых точек")
        return True
    
    def update(self, frame):
        """Обновление положения объекта"""
        if not self.is_initialized:
            return False, (0, 0, 0, 0)
        
        frame_kp, frame_desc = self.orb.detectAndCompute(frame, None)
        
        if frame_desc is None or len(frame_kp) < self.min_matches:
            return False, self.object_bbox
        
        matches = self.matcher.match(self.object_desc, frame_desc)
        
        if len(matches) < self.min_matches:
            return False, self.object_bbox
        
        matches = sorted(matches, key=lambda x: x.distance)
        
        obj_pts = np.float32([self.object_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        scene_pts = np.float32([frame_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        homography, mask = cv2.findHomography(obj_pts, scene_pts, cv2.RANSAC, 5.0)
        
        if homography is None:
            return False, self.object_bbox
        
        x, y, w, h = self.object_bbox
        obj_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        try:
            scene_corners = cv2.perspectiveTransform(obj_corners, homography)
            
            x_coords = [corner[0][0] for corner in scene_corners]
            y_coords = [corner[0][1] for corner in scene_corners]
            
            new_x = int(min(x_coords))
            new_y = int(min(y_coords))
            new_w = int(max(x_coords) - new_x)
            new_h = int(max(y_coords) - new_y)
            
            if new_w > 0 and new_h > 0:
                self.object_bbox = (new_x, new_y, new_w, new_h)
                return True, self.object_bbox
            else:
                return False, self.object_bbox
                
        except:
            return False, self.object_bbox
    
    def visualize(self, frame):
        """Визуализация результатов"""
        vis_frame = frame.copy()
        
        if self.is_initialized and self.object_bbox:
            x, y, w, h = self.object_bbox
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_frame, "Tracked Object", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_frame


def create_demo_video(output_path, duration=10):
    """Создание демонстрационного видео"""
    width, height = 640, 480
    fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    print(f"Создание демонстрационного видео: {output_path}")
    
    for frame_num in range(total_frames):
        # Создание фона
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50
        
        # Добавление текстуры
        noise = np.random.randint(0, 100, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Создание движущегося объекта
        obj_width, obj_height = 120, 80
        
        # Движение по синусоиде
        center_x = int(width/2 + 150 * np.sin(2 * np.pi * frame_num / (fps * 3)))
        center_y = int(height/2 + 50 * np.cos(2 * np.pi * frame_num / (fps * 2)))
        
        x = center_x - obj_width // 2
        y = center_y - obj_height // 2
        
        # Рисование объекта с текстурой
        obj_roi = frame[y:y+obj_height, x:x+obj_width]
        
        # Добавление паттерна
        for i in range(0, obj_width, 10):
            cv2.line(obj_roi, (i, 0), (i, obj_height), (100, 150, 200), 2)
        for i in range(0, obj_height, 10):
            cv2.line(obj_roi, (0, i), (obj_width, i), (200, 150, 100), 2)
        
        # Добавление текста
        cv2.putText(obj_roi, "BOOK", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        writer.write(frame)
        
        # Прогресс
        if frame_num % (fps * 2) == 0:
            progress = (frame_num / total_frames) * 100
            print(f"  Прогресс: {progress:.0f}%")
    
    writer.release()
    print(f"Видео создано: {output_path}")
    return output_path


def process_video_demo(video_path, output_path=None, initial_bbox=None):
    """Демонстрация обработки видео"""
    
    # Открытие видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        return
    
    # Получение информации о видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Информация о видео:")
    print(f"  Разрешение: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Всего кадров: {total_frames}")
    
    # Настройка видеозаписи
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Создание трекера
    tracker = SimpleObjectTracker()
    
    # Чтение первого кадра
    ret, first_frame = cap.read()
    if not ret:
        print("Ошибка: не удалось прочитать первый кадр")
        return
    
    # Получение начальной рамки
    if initial_bbox is None:
        print("\nВыберите объект для отслеживания:")
        print("Используйте мышь для выделения области, затем нажмите ENTER")
        initial_bbox = cv2.selectROI("Выберите объект", first_frame, False, False)
        cv2.destroyAllWindows()
        
        if initial_bbox == (0, 0, 0, 0):
            print("Объект не выбран")
            return
    
    # Инициализация трекера
    if not tracker.initialize(first_frame, initial_bbox):
        print("Ошибка: не удалось инициализировать отслеживание")
        return
    
    # Обработка кадров
    frame_count = 0
    successful_tracks = 0
    
    print("\nНачало обработки видео...")
    print("Нажмите 'q' для выхода")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Обновление отслеживания
        success, bbox = tracker.update(frame)
        
        if success:
            successful_tracks += 1
        
        # Визуализация
        vis_frame = tracker.visualize(frame)
        
        # Добавление информации
        status = "Tracking" if success else "Lost"
        cv2.putText(vis_frame, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Frame: {frame_count}/{total_frames}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Запись кадра
        if writer:
            writer.write(vis_frame)
        
        # Отображение
        cv2.imshow("Object Tracking Demo", vis_frame)
        
        # Прогресс
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Обработка: {progress:.1f}%")
        
        frame_count += 1
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Освобождение ресурсов
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Статистика
    success_rate = (successful_tracks / frame_count) * 100 if frame_count > 0 else 0
    
    print(f"\nРезультаты:")
    print(f"  Всего кадров: {frame_count}")
    print(f"  Успешно отслежено: {successful_tracks}")
    print(f"  Успешность: {success_rate:.1f}%")
    
    if output_path:
        print(f"  Видео сохранено: {output_path}")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Демонстрация системы отслеживания объектов')
    parser.add_argument('--input', '-i', help='Путь к входному видео')
    parser.add_argument('--output', '-o', help='Путь к выходному видео')
    parser.add_argument('--bbox', '-b', nargs=4, type=int, 
                       help='Начальная рамка (x y width height)')
    parser.add_argument('--create-demo', '-d', action='store_true',
                       help='Создать демонстрационное видео')
    parser.add_argument('--demo-output', default='demo_video.mp4',
                       help='Имя файла для демонстрационного видео')
    
    args = parser.parse_args()
    
    print("=== Демонстрация системы отслеживания объектов ===\n")
    
    if args.create_demo:
        # Создание демонстрационного видео
        demo_path = create_demo_video(args.demo_output, duration=10)
        
        # Обработка созданного видео
        initial_bbox = (260, 200, 120, 80)  # Центр видео
        process_video_demo(demo_path, "tracked_demo.mp4", initial_bbox)
        
    elif args.input:
        # Обработка существующего видео
        if not os.path.exists(args.input):
            print(f"Ошибка: файл не найден {args.input}")
            return
        
        initial_bbox = tuple(args.bbox) if args.bbox else None
        process_video_demo(args.input, args.output, initial_bbox)
        
    else:
        print("Использование:")
        print("  python demo.py --create-demo                    # Создать и обработать демо-видео")
        print("  python demo.py --input video.mp4                # Обработать существующее видео")
        print("  python demo.py --input video.mp4 --bbox 100 100 200 150  # С заданной рамкой")
        print("  python demo.py --input video.mp4 --output out.mp4  # С сохранением результата")


if __name__ == "__main__":
    main()