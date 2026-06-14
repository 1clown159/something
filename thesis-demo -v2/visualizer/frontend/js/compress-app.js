/**
 * Compress Page Application
 * 3-step workflow: Upload → Compress → Results
 */

let nav;
let ws = null;
let currentMode = 'compress'; // 'compress' | 'decompress'
let signFile = null;
let mantFile = null;
let headersFile = null;

const $ = (id) => document.getElementById(id);

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initModeToggle();
    initUpload();
    initCompress();
    initDecompress();
    initResults();
    updateNavStatus('ready', '系统就绪');
    logMessage('compressLog', '系统就绪');
});

function initNavigation() {
    nav = new StepNavigation({
        steps: ['upload', 'compress', 'results'],
        startStep: 0,
        onStepChange: (index, name) => {
            // no-op
        }
    });
    nav.stepBtns[0].classList.remove('disabled');
}

/* ============ Mode Toggle ============ */
function initModeToggle() {
    var buttons = document.querySelectorAll('.mode-btn');
    buttons.forEach(function(btn) {
        btn.addEventListener('click', function() {
            buttons.forEach(function(b) { b.classList.remove('active'); });
            btn.classList.add('active');
            currentMode = btn.dataset.mode;
            switchMode(currentMode);
        });
    });
}

function switchMode(mode) {
    var isCompress = mode === 'compress';
    
    // Upload step
    $('uploadDesc').textContent = isCompress
        ? '上传需要压缩的地震数据文件（.sgy / .segy / .bin / .dat）'
        : '上传已压缩的 .s4rc 文件进行解压';
    
    var formats = $('uploadFormats');
    formats.innerHTML = isCompress
        ? '<span class="format-tag">.sgy</span><span class="format-tag">.segy</span><span class="format-tag">.bin</span><span class="format-tag">.dat</span>'
        : '<span class="format-tag">.s4rc</span>';
    
    $('fileInput').accept = isCompress ? '.sgy,.segy,.bin,.dat,.raw' : '.s4rc';
    $('decompressAux').style.display = isCompress ? 'none' : 'block';
    
    // Compress step
    $('step1Title').textContent = isCompress ? '执行压缩' : '执行解压';
    $('step1Desc').textContent = isCompress ? '配置参数并启动压缩任务' : '启动解压任务';
    $('compressConfig').style.display = isCompress ? 'block' : 'none';
    $('decompressConfig').style.display = isCompress ? 'none' : 'block';
    $('progressTitle').textContent = isCompress ? '压缩进度' : '解压进度';
    
    // Results step
    $('step2Title').textContent = isCompress ? '结果分析' : '解压结果';
    $('step2Desc').textContent = isCompress ? '查看压缩统计与文件对比' : '下载解压后的文件';
    $('statsTitle').textContent = isCompress ? '压缩统计' : '解压统计';
    
    // Reset
    resetUpload();
}

/* ============ Step 1: Upload ============ */
function initUpload() {
    const uploadZone = $('uploadZone');
    const fileInput = $('fileInput');
    const filePanel = $('filePanel');
    const removeBtn = $('removeFileBtn');
    const continueBtn = $('continueToCompress');

    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFileUpload(e.target.files[0]);
    });

    uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('dragover'); });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleFileUpload(e.dataTransfer.files[0]);
    });

    removeBtn.addEventListener('click', (e) => { e.stopPropagation(); resetUpload(); });
    continueBtn.addEventListener('click', () => {
        nav.setStepEnabled(1, true);
        nav.goToStep(1);
    });

    // 辅助文件
    var signInput = $('signFileInput');
    var mantInput = $('mantFileInput');
    if (signInput) signInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) { signFile = e.target.files[0]; $('signFileName').textContent = signFile.name; }
    });
    if (mantInput) mantInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) { mantFile = e.target.files[0]; $('mantFileName').textContent = mantFile.name; }
    });
    var headersInput = $('headersFileInput');
    if (headersInput) headersInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) { headersFile = e.target.files[0]; $('headersFileName').textContent = headersFile.name; }
    });
}

async function handleFileUpload(file) {
    showLoading(true);
    updateNavStatus('uploading', '上传中...');

    try {
        if (currentMode === 'decompress') {
            // 解压模式：无需上传到后端，直接保存文件引用
            AppState.setTask('decompress_' + Date.now(), file.name, file.size);
            AppState.uploadedFile = file;
            
            $('fileName').textContent = file.name;
            $('fileSize').textContent = formatFileSize(file.size);
            $('fileDimensions').textContent = '.s4rc compressed file';
            
            $('uploadZone').style.display = 'none';
            $('filePanel').style.display = 'block';
            
            $('displayTaskId').textContent = AppState.taskId;
            $('displayTaskFile').textContent = file.name;
            $('displayTaskSize').textContent = formatFileSize(file.size);
            $('taskInfo').style.display = 'block';
            
            updateNavStatus('ready', '文件就绪');
            logMessage('compressLog', '已选择压缩文件: ' + file.name);
            
            nav.setStepEnabled(1, true);
            setTimeout(function() { nav.goToStep(1); }, 400);
        } else {
            // 压缩模式：上传到后端
            const result = await api.uploadFile(file);
            AppState.setTask(result.task_id, file.name, file.size);

            $('fileName').textContent = file.name;
            $('fileSize').textContent = formatFileSize(file.size);

            if (result.is_sgy) {
                $('fileDimensions').textContent = result.dimensions || 'SEG-Y (' + (result.format || 'unknown') + ')';
            } else {
                const numFloats = file.size / 4;
                $('fileDimensions').textContent = numFloats === 40000 ? '(2 x 100 x 200)' : '(' + numFloats.toLocaleString() + ' floats)';
            }

            $('uploadZone').style.display = 'none';
            $('filePanel').style.display = 'block';

            $('displayTaskId').textContent = result.task_id;
            $('displayTaskFile').textContent = file.name;
            $('displayTaskSize').textContent = formatFileSize(file.size);
            $('taskInfo').style.display = 'block';

            updateNavStatus('ready', '上传完成');
            logMessage('compressLog', '文件上传成功: ' + file.name + (result.is_sgy ? ' (SEG-Y 已自动转换)' : ''));

            nav.setStepEnabled(1, true);
            setTimeout(function() { nav.goToStep(1); }, 600);
        }
    } catch (error) {
        updateNavStatus('error', '上传失败');
        logMessage('compressLog', '上传失败: ' + error.message, 'error');
        showNotification('上传失败，请重试', 'error');
    } finally {
        showLoading(false);
    }
}

function resetUpload() {
    AppState.reset();
    signFile = null;
    mantFile = null;
    headersFile = null;
    $('fileInput').value = '';
    $('uploadZone').style.display = 'block';
    $('filePanel').style.display = 'none';
    $('taskInfo').style.display = 'none';
    $('signFileName').textContent = '未选择';
    $('mantFileName').textContent = '未选择';
    $('headersFileName').textContent = '未选择';
    nav.reset();
    nav.setStepEnabled(1, false);
    nav.setStepEnabled(2, false);
    nav.stepBtns[1].classList.add('disabled');
    nav.stepBtns[2].classList.add('disabled');
}

/* ============ Step 2: Compress ============ */
function initCompress() {
    $('startCompressBtn').addEventListener('click', startCompression);
}

/* ============ Step 2b: Decompress ============ */
function initDecompress() {
    var btn = $('startDecompressBtn');
    if (btn) btn.addEventListener('click', startStandaloneDecompress);
}

async function startStandaloneDecompress() {
    if (!AppState.uploadedFile) {
        showNotification('请先选择 .s4rc 文件', 'warning');
        return;
    }
    if (AppState.isCompressing) return;

    AppState.isCompressing = true;
    $('startDecompressBtn').disabled = true;
    updateNavStatus('decompressing', '解压中...');
    logMessage('compressLog', '开始解压...');
    $('progressStatus').textContent = '上传文件中...';

    try {
        var formData = new FormData();
        formData.append('file', AppState.uploadedFile);
        formData.append('output_format', 'bin');
        if (signFile) formData.append('sign_file', signFile);
        if (mantFile) formData.append('mant_file', mantFile);
        if (headersFile) formData.append('headers_file', headersFile);

        var resp = await axios.post(API_BASE_URL + '/api/decompress-file', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });

        AppState.setTask(resp.data.task_id, AppState.uploadedFile.name, AppState.uploadedFile.size);
        logMessage('compressLog', '解压任务已启动, task_id=' + resp.data.task_id);

        pollUntilDone(resp.data.task_id, 'decompress', 2000);
    } catch (error) {
        AppState.isCompressing = false;
        $('startDecompressBtn').disabled = false;
        updateNavStatus('error', '解压失败');
        logMessage('compressLog', '解压失败: ' + error.message, 'error');
    }
}

async function startCompression() {
    if (!AppState.taskId) {
        showNotification('请先上传文件', 'warning');
        return;
    }
    if (AppState.isCompressing) return;

    const config = {
        feature_mode: $('featureMode').value,
        target_mode: $('targetMode').value,
        patch_shape: [parseInt($('patchH').value), parseInt($('patchW').value)],
        inference_batch: parseInt($('inferenceBatch').value),
        device: $('computeDevice').value
    };

    AppState.isCompressing = true;
    $('startCompressBtn').disabled = true;
    updateNavStatus('compressing', '压缩中...');
    logMessage('compressLog', '开始压缩...');
    $('progressStatus').textContent = '初始化...';

    try {
        const resp = await api.startCompression(AppState.taskId, config);
        logMessage('compressLog', '压缩任务已启动');

        pollUntilDone(AppState.taskId, 'compress', 2000);
    } catch (error) {
        AppState.isCompressing = false;
        $('startCompressBtn').disabled = false;
        updateNavStatus('error', '压缩失败');
        logMessage('compressLog', `压缩失败: ${error.message}`, 'error');
    }
}

/* ============ Polling utility ============ */
function pollUntilDone(taskId, mode, interval) {
    var maxAttempts = mode === 'compress' ? 7200 : 3600; // 4h / 2h
    var attempts = 0;
    var lastProgress = -1;

    function poll() {
        axios.get(API_BASE_URL + '/api/status/' + taskId)
            .then(function(resp) {
                var status = resp.data;
                attempts++;

                if (status.progress !== undefined && status.progress !== lastProgress) {
                    lastProgress = status.progress;
                    updateProgressRing(status.progress);
                    $('progressPercent').textContent = status.progress + '%';
                    $('progressStatus').textContent = status.progress < 100 ? '处理中...' : '完成';
                    $('processedVoxels').textContent = status.progress + '%';
                }

                if (mode === 'compress' && status.status === 'completed') {
                    onCompressDone(status);
                } else if (mode === 'decompress' && status.status === 'decompress_completed') {
                    onDecompressDone(status);
                } else if (status.status === 'failed') {
                    onFailed(status, mode);
                } else if (attempts < maxAttempts) {
                    setTimeout(poll, interval);
                } else {
                    updateNavStatus('error', '轮询超时');
                    logMessage('compressLog', '任务超时，但可能仍在后台运行，请检查后端日志', 'warning');
                }
            })
            .catch(function(err) {
                attempts++;
                if (attempts < maxAttempts) {
                    setTimeout(poll, interval * 2);
                }
            });
    }

    poll();
}

function onCompressDone(status) {
    AppState.isCompressing = false;
    $('startCompressBtn').disabled = false;
    updateProgressRing(100);
    $('progressPercent').textContent = '100%';
    $('progressStatus').textContent = '完成';
    updateNavStatus('ready', '压缩完成');
    logMessage('compressLog', '压缩完成!', 'success');

    if (status.output) updateResults(status.output, false);
    nav.setStepEnabled(2, true);
    nav.markStepDone(1);
    setTimeout(function() { nav.goToStep(2); }, 800);
}

function onDecompressDone(status) {
    AppState.isCompressing = false;
    var isStandalone = !status.output || status.output.compression_ratio === undefined;
    
    if (isStandalone) {
        $('startDecompressBtn').disabled = false;
        updateProgressRing(100);
        $('progressPercent').textContent = '100%';
        $('progressStatus').textContent = '完成';
        updateNavStatus('ready', '解压完成');
        logMessage('compressLog', '解压完成!', 'success');

        var totalUpload = status.file_size || 0;
        var s4rc = status.s4rc_size || 0;
        var signSz = status.sign_size || 0;
        var mantSz = status.mant_size || 0;

        var result = {
            original_size: 0,
            decompressed_size: status.decompressed_size || 0,
            bitstream_size: totalUpload,
            has_aux: !!(signFile || mantFile),
        };
        updateResults(result, true);

        if (signSz || mantSz) {
            logMessage('compressLog',
                '上传文件: s4rc=' + formatFileSize(s4rc) +
                ' + sign=' + formatFileSize(signSz) +
                ' + mant=' + formatFileSize(mantSz) +
                ' = 合计 ' + formatFileSize(totalUpload));
        }
        if (status.has_aux_files) {
            logMessage('compressLog', '已使用 sign/mant 恢复完整 float32 数据', 'success');
        } else {
            logMessage('compressLog', '未提供辅助文件，仅解压指数部分', 'warning');
        }
        if (status.reconstructed_sgy) {
            logMessage('compressLog', '✓ 已使用 headers.json 重建完整 SGY（含头文件）', 'success');
        }
        if (status.original_data_size && status.decompressed_size) {
            var matchSize = status.decompressed_size === status.original_data_size;
            logMessage('compressLog',
                (matchSize ? '✓' : '✗') + ' 大小验证: 解压 ' + formatFileSize(status.decompressed_size) +
                ' / 预期 ' + formatFileSize(status.original_data_size),
                matchSize ? 'success' : 'error');
        }
        nav.setStepEnabled(2, true);
        nav.markStepDone(1);
        setTimeout(function() { nav.goToStep(2); }, 800);
    } else {
        var match = status.verify_match ? '✓ 无损验证通过！' : '✗ 验证未通过！';
        logMessage('resultLog', match, status.verify_match ? 'success' : 'error');
        logMessage('resultLog', '解压文件大小: ' + formatFileSize(status.decompressed_size || 0));
        showNotification('解压验证完成', 'success');
        updateProgressRing(100);
        $('progressPercent').textContent = '100%';
    }
}

function onFailed(status, mode) {
    AppState.isCompressing = false;
    $('startCompressBtn').disabled = false;
    var btn = $('startDecompressBtn');
    if (btn) btn.disabled = false;
    updateNavStatus('error', '失败');
    logMessage('compressLog', '失败: ' + (status.error || '未知错误'), 'error');
    showNotification('任务失败', 'error');
}
function initResults() {
    $('downloadResultBtn').addEventListener('click', downloadResult);
    $('decompressBtn').addEventListener('click', startDecompress);
    $('newTaskBtn').addEventListener('click', resetUpload);
}

function updateResults(output, isDecompress) {
    if (isDecompress) {
        $('compressionRatio').textContent = output.decompressed_size ? formatFileSize(output.decompressed_size) : '-';
        $('bitsPerVoxel').textContent = '-';
        $('totalVoxels').textContent = '-';
        $('compressTime').textContent = '-';
        $('originalSizeText').textContent = output.bitstream_size ? formatFileSize(output.bitstream_size) : '-';
        $('compressedSizeText').textContent = output.decompressed_size ? formatFileSize(output.decompressed_size) : '-';
        $('originalSizeBar').style.width = '100%';
        $('compressedSizeBar').style.width = output.bitstream_size ? Math.min(100, (output.decompressed_size / Math.max(1, output.bitstream_size)) * 100) + '%' : '0%';
        logMessage('resultLog', '解压完成!', 'success');
        if (output.has_aux) {
            logMessage('resultLog', '已使用 sign/mant 辅助文件恢复完整 float32 数据');
        } else {
            logMessage('resultLog', '未提供辅助文件，仅解压指数部分');
        }
    } else {
        $('compressionRatio').textContent = output.compression_ratio ? output.compression_ratio.toFixed(2) + 'x' : '-';
        $('bitsPerVoxel').textContent = output.bits_per_voxel ? output.bits_per_voxel.toFixed(2) : '-';
        $('totalVoxels').textContent = output.original_size ? (output.original_size / 4).toLocaleString() : '-';
        
        var timing = output.timing || {};
        $('compressTime').textContent = timing.total_seconds ? timing.total_seconds.toFixed(1) + 's' : '-';
        $('originalSizeText').textContent = formatFileSize(output.original_size);
        $('compressedSizeText').textContent = formatFileSize(output.total_compressed_bytes || output.compressed_size);
        
        var details = [];
        if (output.exponent_bytes) details.push('指数: ' + formatFileSize(output.exponent_bytes));
        if (output.sign_bytes) details.push('符号: ' + formatFileSize(output.sign_bytes));
        if (output.mant_bytes) details.push('尾数: ' + formatFileSize(output.mant_bytes));
        if (details.length > 0) logMessage('resultLog', '压缩分解: ' + details.join(', '));
        if (output.sign_mant_ratio) logMessage('resultLog', output.sign_mant_ratio);
        
        var ratio = output.total_compression_ratio || output.compression_ratio;
        logMessage('resultLog', '总压缩比: ' + (ratio ? ratio.toFixed(2) : '-') + 'x', 'success');
        
        // 详细耗时
        if (timing.patch_build_s !== undefined) {
            logMessage('resultLog',
                '耗时分解: 特征构建' + timing.patch_build_s.toFixed(1) + 's | ' +
                '模型推理' + (timing.model_inference_s || 0).toFixed(1) + 's | ' +
                '范围编码' + (timing.range_coder_s || 0).toFixed(1) + 's | ' +
                '其他' + (timing.other_overhead_s || 0).toFixed(1) + 's');
        }
    }

    $('originalSizeBar').style.width = '100%';
    var totalCompressed = output.total_compressed_bytes || output.compressed_size;
    var ratio = output.original_size > 0 ? (totalCompressed / output.original_size * 100) : 0;
    $('compressedSizeBar').style.width = ratio + '%';
}

async function downloadResult() {
    if (!AppState.taskId) return;
    try {
        if (currentMode === 'decompress') {
            var url = API_BASE_URL + '/api/download/' + AppState.taskId + '?file_type=decompressed';
            var link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', '');
            document.body.appendChild(link);
            link.click();
            link.remove();
        } else {
            await api.downloadResult(AppState.taskId);
        }
        showNotification('下载已开始', 'success');
        logMessage('resultLog', '下载已开始', 'success');
    } catch (error) {
        showNotification('下载失败', 'error');
        logMessage('resultLog', '下载失败: ' + error.message, 'error');
    }
}

async function startDecompress() {
    if (!AppState.taskId) return;
    showLoading(true);
    logMessage('resultLog', '开始解压验证...');
    try {
        await axios.post(API_BASE_URL + '/api/decompress/' + AppState.taskId, {
            output_filename: AppState.taskId + '_decompressed.bin'
        });
        showLoading(false);
        logMessage('resultLog', '解压任务已启动');
        pollUntilDone(AppState.taskId, 'decompress', 2000);
    } catch (error) {
        showLoading(false);
        logMessage('resultLog', '解压失败: ' + error.message, 'error');
    }
}
