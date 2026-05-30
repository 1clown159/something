/**
 * Range Coder - 范围编码交互演示模块
 */

class RangeCoder {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.apiBase = API_BASE_URL;
        this.range = { low: 0, high: 32767, total: 32768 };
        this.steps = [];
    }

    reset() {
        this.range = { low: 0, high: 32767, total: 32768 };
        this.steps = [];
        this.container.innerHTML = '<div class="chart-placeholder">编码已重置，运行CNN预测后开始编码</div>';
        const box = document.getElementById('encodingStepBox');
        if (box) box.innerHTML = '等待编码...';
    }

    renderCDF(cdf) {
        this.container.innerHTML = '';
        const width = 600, height = 200;
        const margin = { top: 15, right: 20, bottom: 35, left: 50 };

        const svg = d3.select(this.container)
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const x = d3.scaleLinear().domain([0, 256]).range([0, width]);
        const y = d3.scaleLinear().domain([0, 1]).range([height, 0]);

        const line = d3.line()
            .x((d, i) => x(i))
            .y(d => y(d))
            .curve(d3.curveStepAfter);

        svg.append('path').datum(cdf).attr('fill', 'none').attr('stroke', '#667eea').attr('stroke-width', 2).attr('d', line);

        svg.append('g').attr('transform', `translate(0,${height})`).call(d3.axisBottom(x));
        svg.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.0%')));

        svg.append('text').attr('x', width / 2).attr('y', height + 30)
            .attr('text-anchor', 'middle').style('fill', 'var(--text-muted)').style('font-size', '12px')
            .text('累积分布函数 CDF (符号 0-255)');
    }

    encodeStep(cdf, symbol) {
        if (!cdf || symbol === undefined) return;

        const cdfScaled = cdf.map(v => Math.floor(v * this.range.total));

        const rangeWidth = this.range.high - this.range.low + 1;
        const cumLow = symbol > 0 ? cdfScaled[symbol - 1] : 0;
        const cumHigh = cdfScaled[symbol];

        const newLow = this.range.low + Math.floor(rangeWidth * cumLow / this.range.total);
        const newHigh = this.range.low + Math.floor(rangeWidth * cumHigh / this.range.total) - 1;

        const step = {
            symbol: symbol,
            low: newLow,
            high: newHigh,
            range: newHigh - newLow + 1,
            bits: Math.ceil(-Math.log2((newHigh - newLow + 1) / this.range.total))
        };

        this.range.low = newLow;
        this.range.high = newHigh;
        this.steps.push(step);

        const box = document.getElementById('encodingStepBox');
        if (box) {
            box.innerHTML = `
                符号 ${symbol} → 区间 [${newLow}, ${newHigh}]<br>
                当前区间大小: ${step.range} / ${this.range.total}<br>
                ≈ ${step.bits} bits 信息量 | 累计 ${this.steps.length} 步
            `;
        }

        const log = document.getElementById('encodeLog');
        if (log) {
            const line = document.createElement('div');
            line.className = 'log-line log-info';
            line.textContent = `#${this.steps.length}: 符号=${symbol} 区间=[${newLow},${newHigh}] 区间大小=${step.range} ≈${step.bits}bits`;
            log.appendChild(line);
            log.scrollTop = log.scrollHeight;
        }

        return step;
    }

    getFinalBits() {
        if (this.steps.length === 0) return 0;
        const finalRange = this.range.high - this.range.low + 1;
        return Math.ceil(-Math.log2(finalRange / this.range.total));
    }
}
