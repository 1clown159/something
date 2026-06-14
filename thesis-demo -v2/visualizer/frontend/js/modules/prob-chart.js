/**
 * Prob Chart - CNN概率预测 (CDF阶梯图 + Top-10排名表)
 */

class ProbChart {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.apiBase = API_BASE_URL;
        this.lastData = null;
    }

    async predict(taskId, coord, featureMode, targetMode, filePath) {
        this.container.innerHTML = '<div class="loading-spinner" style="margin:auto;"></div>';

        try {
            var body = {
                task_id: taskId,
                coord: coord,
                feature_mode: featureMode,
                target_mode: targetMode
            };
            if (filePath) body.file_path = filePath;
            const resp = await axios.post(this.apiBase + '/api/demo/predict', body);
            this.lastData = resp.data;
            this.render(resp.data);
            return resp.data;
        } catch (error) {
            console.error('[ProbChart] predict error:', error);
            this.container.innerHTML = `<div class="chart-placeholder">预测失败: ${error.response?.data?.detail || error.message}<br><small>请确保后端模型已加载</small></div>`;
            return null;
        }
    }

    render(data) {
        this.container.innerHTML = '';
        const probs = data.probabilities || [];
        if (probs.length === 0) {
            this.container.innerHTML = '<div class="chart-placeholder">无概率数据</div>';
            return;
        }

        const actual = data.actual_symbol;
        const predicted = data.predicted_symbol;
        const entropy = data.entropy;
        const top5 = data.top5 || [];

        const wrapper = document.createElement('div');
        wrapper.className = 'prob-wrapper';
        this.container.appendChild(wrapper);

        this._renderCDF(wrapper, probs, actual, predicted, entropy);
        this._renderTopTable(wrapper, top5, actual, predicted);

        const entBox = document.getElementById('entropyBox');
        if (entBox && entropy !== undefined) {
            entBox.innerHTML = `熵 H = ${entropy.toFixed(4)} bits | 理想码长 ≈ ${entropy.toFixed(2)} bits/symbol`;
        }
    }

    _renderCDF(container, probs, actual, predicted, entropy) {
        const cdfDiv = document.createElement('div');
        cdfDiv.className = 'cdf-panel';

        const w = 580, h = 220;
        const margin = { top: 15, right: 30, bottom: 35, left: 50 };

        const svg = d3.select(cdfDiv)
            .append('svg')
            .attr('width', w + margin.left + margin.right)
            .attr('height', h + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // 计算 CDF
        const cdf = [];
        let cum = 0;
        for (let i = 0; i < probs.length; i++) {
            cum += probs[i];
            cdf.push(cum);
        }

        const x = d3.scaleLinear().domain([0, 255]).range([0, w]);
        const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);

        // 网格
        svg.append('g')
            .attr('class', 'grid')
            .call(d3.axisLeft(y).ticks(5).tickSize(-w).tickFormat(''))
            .style('stroke-dasharray', '3,3')
            .style('stroke-opacity', 0.1);

        // CDF 阶梯线
        const line = d3.line()
            .x((d, i) => x(i))
            .y(d => y(d))
            .curve(d3.curveStepAfter);

        svg.append('path')
            .datum(cdf)
            .attr('fill', 'none')
            .attr('stroke', '#667eea')
            .attr('stroke-width', 2)
            .attr('d', line);

        // 实际符号竖线
        if (actual !== undefined && actual >= 0) {
            const ay = actual > 0 ? y(cdf[actual - 1]) : h;
            svg.append('line')
                .attr('x1', x(actual)).attr('y1', h)
                .attr('x2', x(actual)).attr('y2', ay)
                .attr('stroke', '#ef4444').attr('stroke-width', 2).attr('stroke-dasharray', '6,3');
            svg.append('text')
                .attr('x', x(actual) + 3).attr('y', ay + 12)
                .style('fill', '#ef4444').style('font-size', '10px').style('font-weight', '700')
                .text(`实际 #${actual}`);
        }

        // 预测符号竖线
        if (predicted !== undefined && predicted >= 0) {
            const py = predicted > 0 ? y(cdf[predicted - 1]) : h;
            svg.append('line')
                .attr('x1', x(predicted)).attr('y1', h)
                .attr('x2', x(predicted)).attr('y2', py)
                .attr('stroke', '#3b82f6').attr('stroke-width', 2).attr('stroke-dasharray', '4,4');
            svg.append('text')
                .attr('x', x(predicted) + 3).attr('y', py + (actual === predicted ? 24 : 12))
                .style('fill', '#3b82f6').style('font-size', '10px').style('font-weight', '700')
                .text(`预测 #${predicted}`);
        }

        // 轴
        svg.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(8));
        svg.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.0%')));

        // 熵值标注
        svg.append('text')
            .attr('x', w - 5).attr('y', 12)
            .attr('text-anchor', 'end')
            .style('fill', 'var(--text-accent)').style('font-size', '11px').style('font-family', 'JetBrains Mono,monospace')
            .text(`熵 = ${entropy?.toFixed(3) || '?'} bits`);

        // 标题
        svg.append('text')
            .attr('x', w / 2).attr('y', h + 32)
            .attr('text-anchor', 'middle')
            .style('fill', 'var(--text-muted)').style('font-size', '11px')
            .text('符号值 (0–255) — CDF 累积分布函数');

        container.appendChild(cdfDiv);
    }

    _renderTopTable(container, top5, actual, predicted) {
        const tblDiv = document.createElement('div');
        tblDiv.className = 'top5-panel';

        let rows = '<tr><th>#</th><th>符号</th><th>概率</th><th></th></tr>';
        top5.forEach((item, i) => {
            let mark = '';
            if (item.symbol === actual) mark = '<span style="color:#ef4444;font-weight:700;">← 实际</span>';
            if (item.symbol === predicted) mark = (mark ? mark + ' <span style="color:#3b82f6;">← 预测</span>' : '<span style="color:#3b82f6;font-weight:700;">← 预测</span>');
            const cls = (item.symbol === actual || item.symbol === predicted) ? ' class="t5-highlight"' : '';
            rows += `<tr${cls}><td>${i + 1}</td><td>${item.symbol}</td><td>${(item.prob * 100).toFixed(2)}%</td><td>${mark}</td></tr>`;
        });

        tblDiv.innerHTML = `<h4 class="t5-title">Top-5 概率排名</h4><table class="t5-table">${rows}</table>`;
        container.appendChild(tblDiv);
    }
}
