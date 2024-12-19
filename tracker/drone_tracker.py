import ultralytics
import torch
import cv2
import warnings
import sys

from pathlib import Path
from dataclasses import dataclass 
from time import time

from ultralytics import YOLO, RTDETR

sys.path.append("./")
from drone_display import Display
from utils.paths import latest_video_file_num
from utils.video_utils import get_frame_size
from utils.logs import logger

warnings.filterwarnings('ignore')


@dataclass
class YoloConfig:
    tracker: str =  r'botsort.yaml'
    verbose: bool = True
    persist: bool = True
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_type: ultralytics = YOLO #| RTDETR
    # model_path: str = r"D:\runs\tracker_vt4\weights\best.pt"
    model_path: str = r"C:\Users\1nr_operator_24\Desktop\TrackerVT\Обученные модели\trackerVT_all_detector\weights\best.pt"
    save_video: bool = True 
    saved_video_folder: str = r"output_videos"
    saved_video_name: str = r"output_video" 
    saved_video_format: str = r".mp4"


    
class YoloModel(YoloConfig):
    """Базовый класс для работы с YOLO моделью"""
    def __init__(self):
        self.model = self.load_model()
    
    def load_model(self) -> YOLO:
        model = self.model_type(self.model_path)
        model = model.to(self.device)
        model.fuse()
        return model    
    


class YoloTracker(YoloModel):
    """Класс для задачи трекинга на видеопотоке"""


    def __call__(self, input_video_path) -> None:
        """Запуск работы модели на видеопотоке. Рекомендуется запускать на CUDA"""

        def click_event(event, x: float, y: float, flags, param):
            nonlocal check_all_objs, check_obj_idx          
            """Кликаем на box, получаем его id"""
            if event == cv2.EVENT_LBUTTONDOWN:
                boxes = results[0].boxes
                xywh = boxes.xywh
                ids = boxes.id

                for i in range(len(boxes)):                
                    x_c, y_c, w, h = xywh[i]
                    x_c, y_c, w, h = int(x_c), int(y_c), int(w), int(h)
            
                    if ((x >= x_c - w // 2) and (x <= x_c + w // 2)) and ((y >= y_c - h // 2) and (y <= y_c + h // 2)):
                        check_all_objs = False 
                        check_obj_idx = int(ids[i].numpy().astype(int))
                        break 
            
            elif event == cv2.EVENT_RBUTTONDOWN:
                check_all_objs = True        
        
        check_all_objs = True   # Пока что смотрим все объекты
        check_obj_idx = None    # Индекс объекта в списке, за которым мы наблюдаем

        # Открытие видео файла 
        # cap = cv2.VideoCapture(input_video_path)
        cap = cv2.VideoCapture(cv2.CAP_DSHOW)
        # Размеры экрана
        frame_width, frame_height = get_frame_size(cap)
        # Отображение доп. информации на экран
        cls_display = Display(frame_width, frame_height)

        assert cap.isOpened(), 'Ошибка при открытии видео-файла'  
        # Определяем, куда будем записывать видеопоток
        if self.save_video:
            Path(self.saved_video_folder).mkdir(parents=True, exist_ok=True)
            
            video_num = latest_video_file_num(YoloConfig())
            save_path = f'{self.saved_video_folder}/{self.saved_video_name}{video_num}{self.saved_video_format}'

            save_fps = int(cap.get(cv2.CAP_PROP_FPS))
            save_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            save_out = cv2.VideoWriter(f'{save_path}', save_fourcc, save_fps, (frame_width, frame_height))


        # Цикл по видео фрейму
        while cap.isOpened():
            cls_display.start_time = time()
            # Считывание фрейма из видео
            success, frame = cap.read()            
            if not success:
                break

            # Запуск трекера
            results = self.model.track(frame,
                                       persist=self.persist,
                                       tracker=self.tracker,
                                       verbose=self.verbose)
                        
            boxes = results[0].boxes.cpu()
            obj_count = len(boxes)  # Кол-во распознанных объектов

            try:
                # Выход модели для данного frame
                xyxys = list(boxes.xyxy.numpy().astype(int))
                ids = list(boxes.id.numpy().astype(int))      
                cls = list(boxes.cls.numpy().astype(int))
                conf = list(boxes.conf.numpy().astype(float))

                if check_all_objs is True:
                    for idx in range(len(xyxys)):
                        cls_display.draw_boxing(frame, idx, xyxys, ids, cls, conf)
                
                else:
                    idx_target = ids.index(check_obj_idx)

                    # Отображение на экран приоритетной цели
                    cls_display.draw_boxing(frame, idx_target, xyxys, ids, cls, conf)
                    cls_display.priority_aim_display(frame, xyxys, idx_target, check_obj_idx)

            except (AttributeError, ValueError):  # Если id-шник пропал или нет объектов на экране
                check_all_objs = True


            # Вывод информации на дисплей
            cls_display.working_time_display(frame)
            cls_display.fps_display(frame)
            cls_display.logotype_display(frame)
            cls_display.project_name_display(frame)  
            cls_display.target_aiming_display(frame)
            cls_display.object_count_display(frame, obj_count)
            
            # Вывод кадра на экран
            cv2.imshow('Tracker', frame)
            # Реагирование на клик 
            cv2.setMouseCallback('Tracker', click_event)

            if self.save_video:
                save_out.write(frame)
        
            # Остановка цикла при нажатии "q"
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        # Сохранение видеопотока в указанную папку
        cap.release()
        if self.save_video:
            save_out.release()
            logger.info(f"Сохранённый видеопоток: {save_path}")

        cv2.destroyAllWindows()