console.log("script.js загружен");

document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM полностью загружен");

    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewCanvas = document.getElementById('previewCanvas');
    const previewCtx = previewCanvas ? previewCanvas.getContext('2d') : null;
    const dropText = document.getElementById('dropText');
    const form = document.getElementById('upload-form');
    const loading = document.getElementById('loading');
    const kSlider = document.getElementById('kValue');
    const kDisplay = document.getElementById('kDisplay');

    // Проверка наличия элементов
    if (!dropZone) console.error("dropZone не найден");
    if (!fileInput) console.error("fileInput не найден");
    if (!previewCanvas) console.error("previewCanvas не найден");
    if (!dropText) console.error("dropText не найден");
    if (!form) console.error("upload-form не найден");

    function showPreview(file) {
        console.log("Показ предпросмотра для файла:", file.name);
        if (!previewCtx) {
            console.error("previewCanvas не имеет контекста");
            return;
        }
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = new Image();
            img.src = e.target.result;
            img.onload = function() {
                dropText.classList.add('d-none');
                previewCanvas.classList.remove('d-none');
                previewCanvas.width = dropZone.offsetWidth;
                previewCanvas.height = dropZone.offsetHeight;
                const scale = Math.min(previewCanvas.width / img.width, previewCanvas.height / img.height);
                const w = img.width * scale;
                const h = img.height * scale;
                previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
                previewCtx.drawImage(img, (previewCanvas.width - w) / 2, (previewCanvas.height - h) / 2, w, h);
            };
        };
        reader.readAsDataURL(file);
    }

    if (dropZone) {
        dropZone.addEventListener('click', function(e) {
            e.preventDefault();
            console.log("Клик по dropZone");
            fileInput.click();
        });

        dropZone.addEventListener('dragover', function(e) {
            e.preventDefault();
            dropZone.classList.add('dragover');
            console.log("Dragover");
        });

        dropZone.addEventListener('dragleave', function() {
            dropZone.classList.remove('dragover');
            console.log("Dragleave");
        });

        dropZone.addEventListener('drop', function(e) {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            console.log("Файл сброшен");
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                console.log("Файл присвоен fileInput:", files[0].name);
                showPreview(files[0]);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                console.log("Файл выбран:", this.files[0].name);
                showPreview(this.files[0]);
            }
        });
    }

    if (kSlider && kDisplay) {
        kSlider.addEventListener('input', function() {
            console.log("Slider value:", kSlider.value);
            kDisplay.textContent = kSlider.value === '0' ? 'Auto' : kSlider.value;
        });
        // Инициализация отображения
        kDisplay.textContent = kSlider.value === '0' ? 'Auto' : kSlider.value;
    }

    if (form) {
        form.addEventListener('submit', function(event) {
            console.log("Форма отправлена");
            if (!fileInput.files.length) {
                event.preventDefault();
                console.error("Файл не выбран");
                const toast = new bootstrap.Toast(document.getElementById('errorToast'));
                document.getElementById('errorToast').getElementsByClassName('toast-body')[0].textContent = "Пожалуйста, выберите изображение.";
                toast.show();
            } else {
                if (loading) loading.style.display = 'block';
            }
        });
    }
});