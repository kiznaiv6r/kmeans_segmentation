import numpy as np
import cv2
import os
import glob
import uuid
import base64
import logging
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, FileResponse
from django.conf import settings
from django.views.decorators.csrf import ensure_csrf_cookie
from .utils import KMeans, find_optimal_k, segment_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ensure_csrf_cookie
def index(request):
    logger.info("Запрос на /: %s", request.method)
    if request.method == "POST":
        logger.info("POST-запрос получен")
        logger.info("request.FILES: %s", request.FILES)
        if 'image' not in request.FILES:
            logger.error("Изображение не выбрано")
            return JsonResponse({"error": "Изображение не выбрано"}, status=400)
        
        image_file = request.FILES['image']
        logger.info(f"Выбрано изображение: {image_file.name}")
        if not image_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            logger.error("Недопустимый формат файла")
            return JsonResponse({"error": "Разрешены только PNG или JPEG"}, status=400)
        
        # Очистка старых временных файлов и сессии
        for old_file in glob.glob(os.path.join(settings.MEDIA_ROOT, 'Temp', 'labels_*.npy')):
            try:
                os.remove(old_file)
                logger.info(f"Удалён старый файл: {old_file}")
            except:
                pass
        request.session.flush()
        request.session['palette_history'] = []
        
        image_path = os.path.join(settings.MEDIA_ROOT, 'Uploads', image_file.name)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)
        logger.info(f"Сохранение изображения: {image_path}")

        k = request.POST.get('k', '0')
        k = None if k == '0' else int(k)
        logger.info(f"Используем k: {k if k else 'Auto'}")

        try:
            output_path = os.path.join(settings.MEDIA_ROOT, 'Uploads', 'segmented_image.png')
            segmented_image, labels, centers, (height, width), segmented_image_bgr, original_image = segment_image(
                image_path, k=k, output_path=output_path
            )
            labels_file = os.path.join(settings.MEDIA_ROOT, 'Temp', f'labels_{uuid.uuid4().hex}.npy')
            os.makedirs(os.path.dirname(labels_file), exist_ok=True)
            np.save(labels_file, labels)
            logger.info(f"Метки сохранены в: {labels_file}, shape={labels.shape}")
            
            request.session['labels_file'] = labels_file
            request.session['color_palette'] = centers.tolist()
            request.session['image_path'] = image_path
            request.session['image_height'] = height
            request.session['image_width'] = width
            request.session['palette_history'] = [centers.tolist()]
            request.session.modified = True
            logger.info(f"Ключи сессии: {request.session.keys()}")
        except ValueError as e:
            logger.error(f"Ошибка сегментации: {str(e)}")
            return JsonResponse({"error": str(e)}, status=400)
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            return JsonResponse({"error": str(e)}, status=500)

        try:
            _, buffer = cv2.imencode('.png', segmented_image_bgr)
            segmented_image_b64 = base64.b64encode(buffer).decode("utf-8")
            _, buffer = cv2.imencode('.png', cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
            original_image_b64 = base64.b64encode(buffer).decode("utf-8")
            logger.info(f"Изображения закодированы, размер сегментированного: {len(segmented_image_b64)}")
        except Exception as e:
            logger.error(f"Ошибка кодирования: {e}")
            return JsonResponse({"error": "Не удалось закодировать изображение"}, status=500)

        return render(request, 'segmentation/index.html', {
            'segmented_image': segmented_image_b64,
            'original_image': original_image_b64,
            'image_path': image_path,
            'k': len(centers),
            'height': height,
            'width': width
        })

    return render(request, 'segmentation/index.html')

def recolor(request):
    logger.info("Запрос на /recolor")
    if request.method != "POST":
        logger.error("Требуется POST-запрос")
        return JsonResponse({"error": "Требуется POST-запрос"}, status=400)
    
    try:
        x = float(request.POST.get("x"))
        y = float(request.POST.get("y"))
        color_hex = request.POST.get("color")
        image_path = request.POST.get("image_path")
        k = int(request.POST.get("k"))
        height = int(request.POST.get("height"))
        width = int(request.POST.get("width"))
        canvas_width = float(request.POST.get("canvas_width"))
        canvas_height = float(request.POST.get("canvas_height"))

        labels_file = request.session.get('labels_file')
        if not labels_file or not os.path.isfile(labels_file):
            logger.error("Файл меток не найден")
            return JsonResponse({"error": "Метки не найдены в сессии"}, status=400)

        labels = np.load(labels_file)
        logger.info(f"Загружены метки, форма до reshape: {labels.shape}")
        labels = labels.reshape(height, width)
        logger.info(f"Форма labels после reshape: {labels.shape}")

        x = int(x * width / canvas_width)
        y = int(y * height / canvas_height)
        if x < 0 or x >= width or y < 0 or y >= height:
            logger.error(f"Недопустимый клик: ({x}, {y})")
            return JsonResponse({"error": "Недопустимые координаты клика"}, status=400)

        cluster_idx = labels[y, x]
        logger.info(f"Кликнут кластер: {cluster_idx}")

        color_hex = color_hex.lstrip("#")
        try:
            rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
        except:
            logger.error("Недопустимый формат цвета")
            return JsonResponse({"error": "Недопустимый формат цвета"}, status=400)

        color_palette = np.array(request.session.get('color_palette', []), dtype=np.uint8)
        if len(color_palette) != k:
            logger.error("Палитра не инициализирована")
            return JsonResponse({"error": "Палитра не инициализирована"}, status=400)
        
        # Сохраняем текущую палитру в историю
        palette_history = request.session.get('palette_history', [])
        palette_history.append(color_palette.tolist())
        if len(palette_history) > 10:
            palette_history.pop(0)
        request.session['palette_history'] = palette_history
        
        # Обновляем палитру
        color_palette[cluster_idx] = rgb
        request.session['color_palette'] = color_palette.tolist()
        request.session.modified = True
        logger.info(f"Обновлена палитра для кластера {cluster_idx}: {rgb}, история: {len(palette_history)}")

        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Не удалось загрузить изображение: {image_path}")
            return JsonResponse({"error": f"Не удалось загрузить изображение: {image_path}"}, status=400)
        image = cv2.resize(image, (width, height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logger.info(f"Форма image: {image.shape}")

        segmented_data = color_palette[labels.flatten()]
        logger.info(f"Форма segmented_data: {segmented_data.shape}")
        segmented_image = segmented_data.reshape(height, width, 3)
        segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

        output_path = os.path.join(settings.MEDIA_ROOT, 'Uploads', 'recolored_segmented_image.png')
        cv2.imwrite(output_path, segmented_image_bgr)
        _, buffer = cv2.imencode('.png', segmented_image_bgr)
        segmented_image_b64 = base64.b64encode(buffer).decode('utf-8')

        return JsonResponse({"segmented_image": segmented_image_b64})
    except Exception as e:
        logger.error(f"Ошибка перекраски: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)

def undo(request):
    logger.info("Запрос на /undo")
    if request.method != "POST":
        logger.error("Требуется POST-запрос")
        return JsonResponse({"error": "Требуется POST-запрос"}, status=400)
    
    try:
        palette_history = request.session.get('palette_history', [])
        logger.info(f"История палитр при undo: {len(palette_history)} элементов")
        if len(palette_history) <= 1:
            logger.error("Нет действий для отмены")
            return JsonResponse({"error": "Нет действий для отмены"}, status=400)

        # Удаляем последнюю палитру и берём предыдущую
        palette_history.pop()
        previous_palette = palette_history[-1] if palette_history else request.session.get('color_palette')
        request.session['color_palette'] = previous_palette
        request.session['palette_history'] = palette_history
        request.session.modified = True
        logger.info(f"Восстановлена палитра: {previous_palette}")

        labels_file = request.session.get('labels_file')
        image_path = request.session.get('image_path')
        height = request.session.get('image_height')
        width = request.session.get('image_width')

        if not labels_file or not os.path.isfile(labels_file):
            logger.error("Файл меток не найден")
            return JsonResponse({"error": "Метки не найдены"}, status=400)

        labels = np.load(labels_file).reshape(height, width)
        color_palette = np.array(previous_palette, dtype=np.uint8)

        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Не удалось загрузить изображение: {image_path}")
            return JsonResponse({"error": f"Не удалось загрузить изображение: {image_path}"}, status=400)
        image = cv2.resize(image, (width, height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        segmented_data = color_palette[labels.flatten()]
        segmented_image = segmented_data.reshape(height, width, 3)
        segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

        output_path = os.path.join(settings.MEDIA_ROOT, 'Uploads', 'recolored_segmented_image.png')
        cv2.imwrite(output_path, segmented_image_bgr)
        _, buffer = cv2.imencode('.png', segmented_image_bgr)
        segmented_image_b64 = base64.b64encode(buffer).decode('utf-8')

        return JsonResponse({"segmented_image": segmented_image_b64})
    except Exception as e:
        logger.error(f"Ошибка отмены: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)

def download_image(request):
    logger.info("Запрос на /download")
    output_path = os.path.join(settings.MEDIA_ROOT, 'Uploads', 'recolored_segmented_image.png')
    if not os.path.exists(output_path):
        logger.error("Изображение для скачивания отсутствует")
        return HttpResponse("Изображение для скачивания отсутствует", status=400)
    return FileResponse(open(output_path, 'rb'), as_attachment=True, filename='segmented_image.png')