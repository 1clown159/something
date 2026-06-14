/**
 * Feature Builder - 6通道因果特征图 (SVG热力图矩阵)
 */

class FeatureBuilder {
    constructor() {
        this.apiBase = API_BASE_URL;
    }

    async build(taskId, coord, patchShape, featureMode, targetMode, filePath) {
        const grid = document.getElementById('featuresGrid');
        if (!grid) return;

        grid.innerHTML = '<div class="loading-spinner" style="margin:auto;grid-column:1/-1;"></div>';

        try {
            var url = this.apiBase + '/api/features/' + taskId;
            if (filePath) url += '?file_path=' + encodeURIComponent(filePath);
            const resp = await axios.post(url, {
                coord: coord,
                patch_shape: patchShape,
                feature_mode: featureMode,
                target_mode: targetMode
            });
            const data = resp.data.features;
            if (data.error) {
                grid.innerHTML = `<div class="chart-placeholder" style="grid-column:1/-1;">错误: ${data.error}</div>`;
                return;
            }
            this._renderChannels(data, grid);
            this._updateMetrics(data);
        } catch (error) {
            grid.innerHTML = `<div class="chart-placeholder" style="grid-column:1/-1;">请求失败: ${error.message}</div>`;
        }
    }

    _renderChannels(data, grid) {
        grid.innerHTML = '';
        const channels = data.channels || [];
        const names = ['像素值 (Values)', '可用掩码 (Valid)', '因果掩码 (Causal)', '映射掩码 (Mapped)', '预测值 (Predicted)', '残差值 (Residual)'];

        channels.forEach((channel, i) => {
            const card = document.createElement('div');
            card.className = 'feature-heat-card';

            const raw2D = channel.data;
            if (!raw2D || raw2D.length === 0) return;
            const rowCount = raw2D.length;
            const colCount = raw2D[0] ? raw2D[0].length : 0;
            if (rowCount === 0 || colCount === 0) return;

            // 通道4：预测值是常量网格，渲染为大号数字
            if (i === 4) {
                const flat = raw2D.flat();
                const val = flat[0] !== undefined ? flat[0] : 0;
                const predInt = Math.round(val * 255);
                card.innerHTML = `
                    <div class="channel-header"><span class="channel-name">${names[i]}</span><span class="channel-index">${i}</span></div>
                    <div class="pred-val-display">
                        <div class="pred-val-big">${predInt}</div>
                        <div class="pred-val-label">LOCO-I 预测值（整网格统一）</div>
                        <div class="pred-val-formula">pred = clamp(left + up − up_left, 0, 255)</div>
                    </div>
                `;
                grid.appendChild(card);
                return;
            }

            // 其余通道：SVG 热力图
            const cellW = Math.max(5, Math.min(14, Math.floor(240 / colCount)));
            const cellH = Math.max(5, Math.min(14, Math.floor(180 / rowCount)));
            const svgW = colCount * cellW + 8;
            const svgH = rowCount * cellH + 8;

            // 确定色阶
            const isMask = (i >= 1 && i <= 3);
            const colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#667eea', '#e74c3c'];
            const themeColor = colors[i];

            let svgInner = '';
            for (let r = 0; r < rowCount; r++) {
                const row = raw2D[r] || [];
                for (let c = 0; c < row.length; c++) {
                    const v = row[c];
                    let fill;
                    if (isMask) {
                        fill = v > 0.5 ? themeColor : '#f0f1f5';
                    } else if (i === 5) {
                        // 残差通道：发散色阶（中心=0→灰, 边缘→红/蓝）
                        const t = Math.abs(v - 0.5) * 2;
                        fill = v > 0.5
                            ? interpolateColor('#555', '#e74c3c', t)
                            : interpolateColor('#555', '#1abc9c', t);
                    } else {
                        // 像素值通道：深→亮
                        fill = interpolateColor('#fafafa', themeColor, v);
                    }
                    const label = isMask ? (v > 0.5 ? '1' : '0') : Math.round(v * 255);
                    svgInner += `<rect class="fh-cell" x="${c * cellW + 1}" y="${r * cellH + 1}" width="${cellW - 1}" height="${cellH - 1}" fill="${fill}" data-row="${r}" data-col="${c}" data-val="${label}"/>`;
                }
            }

            const fmtMin = isMask ? (channel.min ?? 0).toFixed(0) : Math.round((channel.min ?? 0) * 255);
            const fmtMax = isMask ? (channel.max ?? 0).toFixed(0) : Math.round((channel.max ?? 0) * 255);
            const fmtMean = isMask ? (channel.mean ?? 0).toFixed(2) : Math.round((channel.mean ?? 0) * 255);

            card.innerHTML = `
                <div class="channel-header"><span class="channel-name">${names[i]}</span><span class="channel-index">${i}</span></div>
                <div class="fh-svg-wrap">
                    <svg width="${svgW}" height="${svgH}" xmlns="http://www.w3.org/2000/svg">${svgInner}</svg>
                </div>
                <div class="channel-stats">
                    <div class="channel-stat"><span class="channel-stat-value">${fmtMin}</span><span class="channel-stat-label">Min</span></div>
                    <div class="channel-stat"><span class="channel-stat-value">${fmtMax}</span><span class="channel-stat-label">Max</span></div>
                    <div class="channel-stat"><span class="channel-stat-value">${fmtMean}</span><span class="channel-stat-label">Mean</span></div>
                </div>
            `;

            // 绑定 hover 事件
            setTimeout(() => {
                card.querySelectorAll('.fh-cell').forEach(cell => {
                    cell.addEventListener('mouseenter', (e) => {
                        e.target.setAttribute('stroke', 'white');
                        e.target.setAttribute('stroke-width', '1.5');
                        let tip = document.querySelector('.fh-tooltip');
                        if (!tip) {
                            tip = document.createElement('div');
                            tip.className = 'fh-tooltip';
                            tip.style.cssText = 'position:fixed;background:#fff;color:#1a1a2e;padding:5px 9px;border-radius:5px;font-size:11px;font-family:JetBrains Mono,monospace;pointer-events:none;z-index:999;border:1px solid rgba(0,0,0,0.12);box-shadow:0 2px 12px rgba(0,0,0,0.1);white-space:nowrap;';
                            document.body.appendChild(tip);
                        }
                        tip.textContent = `[${e.target.dataset.row},${e.target.dataset.col}] = ${e.target.dataset.val}`;
                        tip.style.left = (e.clientX + 12) + 'px';
                        tip.style.top = (e.clientY - 24) + 'px';
                    });
                    cell.addEventListener('mouseleave', (e) => {
                        e.target.removeAttribute('stroke');
                        e.target.removeAttribute('stroke-width');
                        const t = document.querySelector('.fh-tooltip');
                        if (t) t.remove();
                    });
                });
            }, 50);

            grid.appendChild(card);
        });
    }

    _updateMetrics(data) {
        const $ = (id) => document.getElementById(id);
        if ($('predictedValue')) $('predictedValue').textContent = data.predicted_value ?? '-';
        if ($('actualValue')) $('actualValue').textContent = data.actual_value ?? '-';
        if ($('targetSymbol')) $('targetSymbol').textContent = data.target_symbol ?? data.actual_value ?? '-';
        if ($('coordDisplay')) $('coordDisplay').textContent = (data.coord || []).join(', ');
    }
}
