/**
 * General Compression App - 通用压缩功能
 */

let selectedAlgo = 'lz77';
let currentFile = null;

const $ = (id) => document.getElementById(id);

document.addEventListener('DOMContentLoaded', () => {
    initAlgoSelect();
    initModeSwitch();
    initTextDemo();
    initFileCompress();
});

/* ===== Algorithm Cards ===== */
function initAlgoSelect() {
    document.querySelectorAll('.gc-algo-card').forEach(card => {
        card.addEventListener('click', () => {
            document.querySelectorAll('.gc-algo-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            selectedAlgo = card.dataset.algo;
        });
    });
}

/* ===== Mode Switch ===== */
function initModeSwitch() {
    const btns = document.querySelectorAll('.gc-mode-btn');
    const panels = document.querySelectorAll('.mode-panel');

    btns.forEach(btn => {
        btn.addEventListener('click', () => {
            btns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            panels.forEach(p => p.classList.add('hidden'));
            const mode = btn.dataset.mode;
            if (mode === 'text') $('textMode').classList.remove('hidden');
            else $('fileMode').classList.remove('hidden');
        });
    });
}

/* ===== Text Demo ===== */
function initTextDemo() {
    $('runDemoBtn').addEventListener('click', runTextDemo);
    $('clearDemoBtn').addEventListener('click', () => {
        $('demoInput').value = '';
        $('demoResultsPanel').classList.add('hidden');
    });
}

async function runTextDemo() {
    const text = $('demoInput').value.trim();
    if (!text) {
        alert('请输入要测试的文本');
        return;
    }

    const btn = $('runDemoBtn');
    btn.disabled = true;
    btn.textContent = '处理中...';

    try {
        const results = await CompressionModule.compareAll(text);
        showDemoResults(results, text.length);
    } catch (e) {
        alert('测试失败: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '▶ 执行对比测试';
    }
}

function showDemoResults(results, origLen) {
    $('origBarVal').textContent = origLen + ' bytes';

    let best = '';
    let bestRatio = 0;
    for (const [k, v] of Object.entries(results)) {
        if (v.success && v.compression_ratio > bestRatio) {
            bestRatio = v.compression_ratio;
            best = k;
        }
    }

    ['lz77', 'rle', 'deflate', 'neural'].forEach(k => {
        if (results[k] && results[k].success) {
            const r = results[k].compressed_size / origLen;
            const bar = $(k + 'Bar');
            const barVal = $(k + 'BarVal');
            if (bar) bar.style.width = (r * 100) + '%';
            if (r < 0.8) bar.textContent = (r * 100).toFixed(0) + '%';
            if (barVal) barVal.textContent = results[k].compressed_size + ' bytes';
        }
    });

    const grid = $('comparisonGrid');
    if (grid) {
        grid.innerHTML = Object.entries(results).filter(([,v]) => v.success).map(([k, v]) => {
            const isBest = k === best;
            const names = { lz77: 'LZ77+Huffman', rle: 'RLE', deflate: 'Deflate', neural: 'Neural Predictor' };
            return `<div class="gc-result-card ${isBest ? 'best' : ''}">
                <div class="gc-result-header">
                    <span class="gc-result-algo">${names[k]}</span>
                    <span class="gc-result-badge ${isBest ? 'best' : 'normal'}">${isBest ? '🏆 最佳' : v.compression_ratio.toFixed(2) + ':1'}</span>
                </div>
                <div class="stat-row"><span class="stat-label">原始</span><span class="stat-value">${v.original_size} B</span></div>
                <div class="stat-row"><span class="stat-label">压缩后</span><span class="stat-value">${v.compressed_size} B</span></div>
                <div class="stat-row"><span class="stat-label">压缩比</span><span class="stat-value" style="color:${isBest ? 'var(--success)' : 'var(--primary-color)'}">${v.compression_ratio.toFixed(2)}:1</span></div>
                <div class="stat-row"><span class="stat-label">节省</span><span class="stat-value" style="color:${isBest ? 'var(--success)' : 'var(--text-primary)'}">${v.space_saving.toFixed(1)}%</span></div>
                <div class="stat-row"><span class="stat-label">验证</span><span class="stat-value" style="color:${v.verification_passed ? 'var(--success)' : 'var(--error)'}">${v.verification_passed ? '✓ 通过' : '✗ 失败'}</span></div>
                <div class="stat-row"><span class="stat-label">耗时</span><span class="stat-value">${v.compression_time_ms.toFixed(0)} ms</span></div>
            </div>`;
        }).join('');
    }

    $('demoResultsPanel').classList.remove('hidden');
}

/* ===== File Compression ===== */
function initFileCompress() {
    const zone = $('uploadZone');
    const input = $('fileInput');
    const compressBtn = $('compressBtn');
    const clearBtn = $('clearFilesBtn');

    zone.addEventListener('click', () => input.click());
    input.addEventListener('change', e => handleFile(e.target.files[0]));

    zone.addEventListener('dragover', e => { e.preventDefault(); zone.style.borderColor = 'var(--primary-color)'; });
    zone.addEventListener('dragleave', () => { zone.style.borderColor = ''; });
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.style.borderColor = '';
        if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    });

    compressBtn.addEventListener('click', compressFile);
    clearBtn.addEventListener('click', clearFile);
}

function handleFile(file) {
    if (!file) return;
    currentFile = file;
    $('fileList').innerHTML = `
        <div class="gc-file-item">
            <div>
                <div class="gc-file-name">${file.name}</div>
                <div class="gc-file-size">${fmtSize(file.size)}</div>
            </div>
            <button class="gc-file-close" onclick="clearFile()">&times;</button>
        </div>`;
    $('compressBtn').disabled = false;
    $('compressResults').classList.add('hidden');
}

function clearFile() {
    currentFile = null;
    $('fileList').innerHTML = '';
    $('compressBtn').disabled = true;
    $('compressResults').classList.add('hidden');
    $('fileInput').value = '';
}

async function compressFile() {
    if (!currentFile) return;

    const btn = $('compressBtn');
    btn.disabled = true;
    btn.textContent = '压缩中...';

    try {
        const buffer = await currentFile.arrayBuffer();
        const data = new Uint8Array(buffer);

        const t0 = performance.now();
        const result = await CompressionModule.compress(selectedAlgo, data);
        const elapsed = performance.now() - t0;

        const cSize = result.data.length;
        const oSize = data.length;

        $('compressResults').classList.remove('hidden');
        $('singleResult').innerHTML = `
            <div style="background:rgba(0,0,0,0.02);padding:var(--spacing-md);border-radius:var(--radius-md);">
                <h3 style="color:var(--primary-color);margin-bottom:var(--spacing-sm);">${getAlgoName(selectedAlgo)}</h3>
                <div class="stat-row"><span class="stat-label">原始大小</span><span class="stat-value">${fmtSize(oSize)}</span></div>
                <div class="stat-row"><span class="stat-label">压缩后</span><span class="stat-value">${fmtSize(cSize)}</span></div>
                <div class="stat-row"><span class="stat-label">压缩比</span><span class="stat-value" style="color:var(--success)">${(oSize / cSize).toFixed(2)}:1</span></div>
                <div class="stat-row"><span class="stat-label">节省空间</span><span class="stat-value" style="color:var(--success)">${((1 - cSize / oSize) * 100).toFixed(1)}%</span></div>
                <div class="stat-row"><span class="stat-label">耗时</span><span class="stat-value">${elapsed.toFixed(0)} ms</span></div>
            </div>`;

        // Download link
        const blob = new Blob([result.data], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        const link = $('downloadLink');
        link.href = url;
        link.download = currentFile.name + '.compressed';
        link.classList.remove('hidden');
    } catch (e) {
        alert('压缩失败: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '🚀 开始压缩';
    }
}

function getAlgoName(algo) {
    const names = { lz77: 'LZ77 + Huffman', rle: 'RLE 行程编码', deflate: 'Deflate', neural: 'Neural Predictor' };
    return names[algo] || algo;
}

function fmtSize(b) {
    if (b === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(b) / Math.log(k));
    return parseFloat((b / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Expose for inline handlers
window.clearFile = clearFile;
