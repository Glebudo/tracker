import sys
import warnings
import cv2


from drone_tracker import YoloTracker

warnings.filterwarnings('ignore')

sys.path.append(r"./")
from utils.logs import logger



if __name__ == '__main__':

    
    tracker = YoloTracker()

    logger.info(f'Загружена модель: {tracker.model_path}')
    logger.info(f'Используемый девайс: {tracker.device}')

    input_video_path = "http://localhost:8080"
    
    tracker(input_video_path) # Запуск трекера
    logger.info('Завершение работы трекера')