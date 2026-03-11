// 上传区域交互
const uploadArea = document.getElementById('uploadArea');
const videoInput = document.getElementById('videoInput');
const configSection = document.getElementById('configSection');
const processBtn = document.getElementById('processBtn');
const progressSection = document.getElementById('progressSection');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');
const downloadLink = document.getElementById('downloadLink');

let selectedFile = null;

// 拖拽上传
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// 点击上传
videoInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    // 验证文件类型
    const allowedTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/x-flv'];
    if (!allowedTypes.includes(file.type)) {
        showError('不支持的视频格式！请选择 MP4, AVI, MOV, MKV 或 FLV 格式。');
        return;
    }

    // 验证文件大小（最大 500MB）
    if (file.size > 500 * 1024 * 1024) {
        showError('文件太大！请选择小于 500MB 的视频。');
        return;
    }

    selectedFile = file;
    uploadArea.innerHTML = `
        <div class="upload-icon">✅</div>
        <h3>已选择：${file.name}</h3>
        <p>大小：${(file.size / 1024 / 1024).toFixed(2)} MB</p>
    `;

    configSection.style.display = 'block';
}

// 处理视频
processBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        showError('请先选择视频文件！');
        return;
    }

    // 显示进度
    configSection.style.display = 'none';
    progressSection.style.display = 'block';

    const useKalman = document.getElementById('useKalman').checked;

    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('use_kalman', useKalman.toString());

    try {
        updateProgress(10, '正在上传视频...');

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            // 处理成功
            updateProgress(100, '处理完成！');
            setTimeout(() => {
                showResult(data);
            }, 500);
        } else {
            // 处理失败
            showError(data.error || '处理失败，请重试');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('网络错误，请检查连接后重试');
    }
});

function updateProgress(percent, text) {
    progressFill.style.width = percent + '%';
    progressText.textContent = text;
}

function showResult(data) {
    progressSection.style.display = 'none';
    resultSection.style.display = 'block';

    // 更新统计信息
    document.getElementById('totalFrames').textContent = data.stats.total_frames;
    document.getElementById('trackedFrames').textContent = data.stats.tracked_frames;
    document.getElementById('successRate').textContent = data.stats.success_rate.toFixed(1) + '%';
    document.getElementById('lostFrames').textContent = data.stats.lost_frames;

    // 设置下载链接
    downloadLink.href = data.download_url;
    downloadLink.download = data.filename;
}

function showError(message) {
    progressSection.style.display = 'none';
    configSection.style.display = 'none';
    errorSection.style.display = 'block';
    document.getElementById('errorMessage').textContent = message;
}
