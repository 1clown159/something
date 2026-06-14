/**
 * Shared Utilities - Stage4 Visualizer
 */

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function interpolateColor(color1, color2, factor) {
    const hex2rgb = (hex) => ({
        r: parseInt(hex.slice(1, 3), 16),
        g: parseInt(hex.slice(3, 5), 16),
        b: parseInt(hex.slice(5, 7), 16)
    });
    const c1 = hex2rgb(color1);
    const c2 = hex2rgb(color2);
    const r = Math.round(c1.r + (c2.r - c1.r) * factor);
    const g = Math.round(c1.g + (c2.g - c1.g) * factor);
    const b = Math.round(c1.b + (c2.b - c1.b) * factor);
    return `rgb(${r}, ${g}, ${b})`;
}

function flatten2D(data2D) {
    const flat = [];
    data2D.forEach((row, i) => {
        row.forEach((val, j) => {
            flat.push({ value: val, row: i, col: j });
        });
    });
    return flat;
}

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.style.display = show ? 'flex' : 'none';
}

function showNotification(message, type = 'info') {
    const notif = document.createElement('div');
    const colors = { success: 'var(--success)', error: 'var(--error)', warning: 'var(--warning)', info: 'var(--info)' };
    notif.style.cssText = `
        position: fixed; top: 80px; right: 20px;
        padding: 1rem 1.5rem;
        background: ${colors[type] || colors.info};
        color: white; border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        z-index: 1001;
        animation: slideIn 0.3s ease;
        font-size: 0.9375rem;
    `;
    notif.textContent = message;
    document.body.appendChild(notif);
    setTimeout(() => {
        notif.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notif.remove(), 300);
    }, 3000);
}

function updateNavStatus(type, message) {
    const dot = document.querySelector('.status-dot');
    const text = document.querySelector('.status-text');
    if (!dot || !text) return;
    text.textContent = message;
    const colors = { ready: 'var(--success)', error: 'var(--error)', uploading: 'var(--warning)', compressing: 'var(--warning)', processing: 'var(--info)' };
    dot.style.background = colors[type] || 'var(--success)';
}

function logMessage(containerId, message, type = 'info') {
    const container = document.getElementById(containerId);
    if (!container) return;
    const line = document.createElement('div');
    line.className = `log-line log-${type}`;
    const time = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    line.textContent = `[${time}] ${message}`;
    container.appendChild(line);
    container.scrollTop = container.scrollHeight;
}

function updateProgressRing(percent) {
    const ring = document.getElementById('progressRingFill');
    if (!ring) return;
    const circumference = 2 * Math.PI * 54;
    const offset = circumference - (percent / 100) * circumference;
    ring.style.strokeDashoffset = offset;
}
