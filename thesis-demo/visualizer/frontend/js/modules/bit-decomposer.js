/**
 * Bit Decomposer - Float32 位拆解可视化模块
 */

class BitDecomposer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.apiBase = API_BASE_URL;
    }

    async decompose(values) {
        this.container.innerHTML = '<div class="loading-spinner" style="margin:auto;"></div>';

        try {
            const resp = await axios.post(`${this.apiBase}/api/demo/decompose`, { values  });
            const results = resp.data.decomposed || [];

            this.container.innerHTML = '';
            results.forEach((item, idx) => {
                const wrapper = document.createElement('div');
                wrapper.style.marginBottom = '24px';
                wrapper.innerHTML = `
                    <div style="margin-bottom:8px;font-weight:600;color:var(--text-primary);">
                        浮点值: <span style="color:var(--text-accent);">${item.original}</span>
                    </div>
                `;
                wrapper.appendChild(this._renderBitRow(item));
                wrapper.appendChild(this._renderSections(item));
                this.container.appendChild(wrapper);
            });
        } catch (error) {
            // Fallback: local decompose
            if (values && values.length > 0) {
                this.container.innerHTML = '';
                const v = values[0];
                const buf = new ArrayBuffer(4);
                const f32 = new Float32Array(buf);
                const u32 = new Uint32Array(buf);
                f32[0] = v;
                const bits = u32[0].toString(2).padStart(32, '0');
                const sign = bits[0];
                const exp = bits.slice(1, 9);
                const mant = bits.slice(9);

                const item = {
                    original: v, sign: parseInt(sign, 2), exp_raw: parseInt(exp, 2),
                    exp_value: parseInt(exp, 2) - 127, mant: parseInt(mant, 2).toString(16).toUpperCase().padStart(6, '0'),
                    binary: `${sign}|${exp}|${mant}`
                };
                const wrapper = document.createElement('div');
                wrapper.innerHTML = `<div style="margin-bottom:8px;font-weight:600;color:var(--text-primary);">浮点值: <span style="color:var(--text-accent);">${item.original}</span></div>`;
                wrapper.appendChild(this._renderBitRow(item));
                wrapper.appendChild(this._renderSections(item));
                this.container.innerHTML = '';
                this.container.appendChild(wrapper);
            }
        }
    }

    _renderBitRow(item) {
        const row = document.createElement('div');
        row.className = 'bit-row';
        row.style.display = 'flex';
        row.style.justifyContent = 'center';

        const parts = item.binary.split('|');
        const signBits = parts[0].split('');
        const expBits = parts[1].split('');
        const mantBits = parts[2].split('');

        const allBits = [
            ...signBits.map(b => ({ bit: b, type: 'sign' })),
            ...expBits.map(b => ({ bit: b, type: 'exp' })),
            ...mantBits.map(b => ({ bit: b, type: 'mant' }))
        ];

        allBits.forEach((b, i) => {
            const cell = document.createElement('div');
            cell.className = `bit-cell ${b.type}`;
            cell.textContent = b.bit;
            cell.title = `Bit ${31-i}: ${b.type === 'sign' ? '符号' : b.type === 'exp' ? '指数' : '尾数'}`;
            row.appendChild(cell);
        });

        return row;
    }

    _renderSections(item) {
        const div = document.createElement('div');
        div.style.display = 'flex';
        div.style.gap = '12px';
        div.style.marginTop = '12px';

        const signVal = item.sign === 1 ? '负数 (-)' : '正数 (+)';
        const expVal = `指数值 = ${item.exp_value} (原始 0x${item.exp_raw.toString(16).toUpperCase()})`;
        const mantVal = `尾数 = 0x${item.mant} (${parseInt(item.mant, 16)} / ${1 << 23})`;

        div.innerHTML = `
            <div class="bit-section sign-sec" style="flex:1;">
                <div style="font-size:0.75rem;color:var(--text-muted);">符号位 (1bit)</div>
                <div style="font-family:monospace;color:#fca5a5;margin-top:4px;">${signVal}</div>
            </div>
            <div class="bit-section exp-sec" style="flex:1;">
                <div style="font-size:0.75rem;color:var(--text-muted);">指数位 (8bits)</div>
                <div style="font-family:monospace;color:#93c5fd;margin-top:4px;">${expVal}</div>
            </div>
            <div class="bit-section mant-sec" style="flex:1;">
                <div style="font-size:0.75rem;color:var(--text-muted);">尾数位 (23bits)</div>
                <div style="font-family:monospace;color:#6ee7b7;margin-top:4px;">${mantVal}</div>
            </div>
        `;
        return div;
    }
}
