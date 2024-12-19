import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def convert_box(size, box):
        """Перевод Bounding Box из VisDrone формата в YOLO"""
        dw, dh = 1. / size[0], 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh


def visdrone2yolo(dir) -> None:
    """Функция для перевода датасета из VisDrone формата в YOLO"""
    (dir / 'labels').mkdir(parents=True, exist_ok=True)  # Создание директори labels
    pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
    for f in pbar:
        img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size
        lines = []
        with open(f, 'r') as file:  # Прочтение annotations.txt
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0': 
                    continue
                cls = int(row[5]) - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(str(f).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}'), 'w') as fl:
                    fl.writelines(lines)  # Запись labels.txt




# if __name__ == '__main__': 
    # dir = Path(r"D:\Трекер на обучение\visdrone")  # Путь до папки с датасетом



    # for d in ['train', 'test', 'valid']:
    #     visdrone2yolo(dir / d)