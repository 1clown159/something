/**
 * Exponent Viewer - 指数数据热力图浏览模块
 */

class ExponentViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.expData = null;
        this.shape = null;
        this.apiBase = API_BASE_URL;
    }

    async loadFromFile(demoTaskId, axis, idx) {
        this.container.innerHTML = '<div class="loading-spinner" style="margin:auto;"></div>';

        try {
            const resp = await axios.get(`${this.apiBase}/api/demo/exponents/${demoTaskId}`, {
                params: { axis: axis, index: idx }
            });
            this.expData = resp.data.data;
            this.shape = resp.data.shape;
            this.renderHeatmap(axis, idx);
        } catch (error) {
            console.error('Exponent load error:', error);
            this.container.innerHTML = '<div class="chart-placeholder">加载失败，请确认文件已上传<br><small>' + (error.response?.data?.detail || error.message) + '</small></div>';
        }
    }

    renderHeatmap(axis, idx) {
        this.container.innerHTML = '';
        if (!this.expData || !Array.isArray(this.expData) || this.expData.length === 0) return;

        const data = this.expData;
        const rowCount = data.length;
        const colCount = data[0] ? data[0].length : 0;

        if (rowCount === 0 || colCount === 0) return;

        const flat = data.flat();
        const dataMin = Math.min(...flat);
        const dataMax = Math.max(...flat);
        const dataRange = dataMax - dataMin || 1;

        const cellW = Math.max(3, Math.min(10, Math.floor(500 / colCount)));
        const cellH = Math.max(3, Math.min(10, Math.floor(380 / rowCount)));
        const svgW = colCount * cellW + 90;
        const svgH = rowCount * cellH + 60;

        const svg = d3.select(this.container)
            .append('svg')
            .attr('width', svgW)
            .attr('height', svgH)
            .append('g')
            .attr('transform', 'translate(55, 30)');

        const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([dataMin, dataMax]);

        for (let i = 0; i < rowCount; i++) {
            const row = data[i];
            if (!row) continue;
            for (let j = 0; j < row.length; j++) {
                const val = row[j];
                svg.append('rect')
                    .attr('class', 'heatmap-cell')
                    .attr('x', j * cellW)
                    .attr('y', i * cellH)
                    .attr('width', cellW)
                    .attr('height', cellH)
                    .attr('fill', colorScale(val))
                    .on('mouseover', function() {
                        d3.select(this).attr('stroke', 'white').attr('stroke-width', 2);
                    })
                    .on('mousemove', function(event) {
                        let tip = document.querySelector('.heatmap-tooltip');
                        if (!tip) {
                            tip = document.createElement('div');
                            tip.className = 'heatmap-tooltip';
                            tip.style.cssText = 'position:fixed;background:#fff;color:#1a1a2e;padding:6px 10px;border-radius:6px;font-size:12px;font-family:JetBrains Mono,monospace;pointer-events:none;z-index:999;border:1px solid rgba(0,0,0,0.12);box-shadow:0 2px 12px rgba(0,0,0,0.1);';
                            document.body.appendChild(tip);
                        }
                        const floatVal = (2 ** (val - 127)).toFixed(4);
                        tip.innerHTML = `exp: <b>${val}</b> (0x${val.toString(16).toUpperCase().padStart(2, '0')})<br>≈ float: ${floatVal}`;
                        tip.style.left = (event.clientX + 14) + 'px';
                        tip.style.top = (event.clientY - 30) + 'px';
                    })
                    .on('mouseout', function() {
                        d3.select(this).attr('stroke', null).attr('stroke-width', null);
                        const tip = document.querySelector('.heatmap-tooltip');
                        if (tip) tip.remove();
                    });
            }
        }

        const barW = 14;
        const barH = rowCount * cellH;
        const legendG = svg.append('g').attr('transform', `translate(${colCount * cellW + 12}, 0)`);
        const steps = 50;
        for (let i = 0; i < steps; i++) {
            const t = 1 - i / (steps - 1);
            legendG.append('rect')
                .attr('x', 0)
                .attr('y', i * barH / steps)
                .attr('width', barW)
                .attr('height', barH / steps + 1)
                .attr('fill', colorScale(dataMin + t * dataRange));
        }
        legendG.append('text')
            .attr('x', barW + 5).attr('y', 10)
            .style('font-size', '9px').style('fill', '#dddddd')
            .text(dataMax);
        legendG.append('text')
            .attr('x', barW + 5).attr('y', barH - 2)
            .style('font-size', '9px').style('fill', '#dddddd')
            .text(dataMin);

        // 中间标注 float ≈ 1.0 (exp=127) 的位置
        if (dataMin <= 127 && dataMax >= 127) {
            const frac = (127 - dataMin) / dataRange;
            const y127 = (1 - frac) * barH;
            legendG.append('line')
                .attr('x1', -4).attr('y1', y127).attr('x2', barW + 4).attr('y2', y127)
                .attr('stroke', 'white').attr('stroke-width', 0.8).attr('stroke-dasharray', '3,3');
            legendG.append('text')
                .attr('x', barW + 5).attr('y', y127 + 4)
                .style('font-size', '8px').style('fill', 'white')
                .text('127 (≈1.0)');
        }

        const axisNames = ['Profile', 'Trace', 'Sample'];
        svg.append('text')
            .attr('x', colCount * cellW / 2)
            .attr('y', rowCount * cellH + 42)
            .attr('text-anchor', 'middle')
            .style('font-size', '12px')
            .style('fill', 'var(--text-muted)')
            .text(`${axisNames[axis] || 'Axis'} 指数切片 #${idx} | exp∈[${dataMin},${dataMax}] | 平均≈${Math.round(flat.reduce((a,b) => a+b, 0)/flat.length)}`);
    }
}
