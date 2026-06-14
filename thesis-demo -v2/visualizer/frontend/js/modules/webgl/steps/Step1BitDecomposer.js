/**
 * Step 1: Float32 Bit Decomposer - 3D Crystal Array (Enhanced)
 */
class Step1BitDecomposer {
    constructor() {
        this.group = new THREE.Group();
        this.cubes = [];
        this.connectors = [];
        this.ripples = [];
        this.data = null;
        this.pulseT = 0;
    }

    async enter(sm, data) {
        sm.scene.add(this.group);
        // Prefer API data over mock
        if (data.apiData && data.apiData.sign !== undefined) {
            this.data = data.apiData;
            // Ensure numeric fields
            this.data.sign = data.apiData.sign;
            this.data.exp = data.apiData.exp_raw || data.apiData.exp;
            this.data.mant = data.apiData.mant;
            this.data.value = data.apiData.original || data.apiData.value;
        } else {
            this.data = WebGLUtils.generateMockFloat32(data.sampleIndex || 0);
        }
        this._build(sm);
        sm.cameraController.setPosition(0, 6, 16, new THREE.Vector3(0, 0, 0), 1400);
        sm.setBloom(0.5, 0.4, 0.82);
        this._updateInfo(this.data);
    }

    _build(sm) {
        const bits = [];
        bits.push({ val: this.data.sign, type: 'sign', idx: 31 });
        for (let i = 7; i >= 0; i--) bits.push({ val: (this.data.exp >> i) & 1, type: 'exp', idx: 23 + i });
        for (let i = 22; i >= 0; i--) bits.push({ val: (this.data.mant >> i) & 1, type: 'mant', idx: i });

        const colorMap = { sign: 0xef4444, exp: 0xf59e0b, mant: 0x06b6d4 };
        const boxSize = 0.6;
        const gap = 0.06;
        const totalW = 32 * (boxSize + gap);
        const startX = -totalW / 2 + boxSize / 2;

        // Build bit cubes with staggered entry animation
        bits.forEach((b, i) => {
            const geo = new THREE.BoxGeometry(boxSize, boxSize * 1.25, boxSize * 0.5);
            const col = new THREE.Color(colorMap[b.type]);
            const mat = new THREE.MeshStandardMaterial({
                color: col,
                transparent: true,
                opacity: 0,
                emissive: col,
                emissiveIntensity: 0,
                roughness: 0.25,
                metalness: 0.75
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(startX + i * (boxSize + gap), 0, 0);
            mesh.userData = {
                bit: b, baseIdx: i,
                baseOpacity: b.val ? 0.95 : 0.45,
                baseEmissive: b.val ? 0.85 : 0.15,
                onHoverEnter: (obj) => {
                    obj.material.emissiveIntensity = 1.8;
                    obj.scale.setScalar(1.35);
                    this._spawnRipple(obj.position, colorMap[b.type]);
                    if (this.formulaSprite) WebGLUtils.animateValue(this.formulaSprite.material, 'opacity', this.formulaSprite.material.opacity, 1, 300);
                },
                onHoverLeave: (obj) => {
                    obj.material.emissiveIntensity = obj.userData.baseEmissive;
                    obj.scale.setScalar(1);
                    if (this.formulaSprite) WebGLUtils.animateValue(this.formulaSprite.material, 'opacity', this.formulaSprite.material.opacity, 0, 300);
                },
                onClick: (obj) => {
                    this._showBitDetail(obj.userData.bit);
                }
            };
            this.group.add(mesh);
            this.cubes.push(mesh);

            // Staggered entry: fly in from far z + random y offset
            const fromZ = -30 - Math.random() * 15;
            const fromY = (Math.random() - 0.5) * 10;
            mesh.position.z = fromZ;
            mesh.position.y = fromY;
            // Animate position
            const targetPos = new THREE.Vector3(startX + i * (boxSize + gap), 0, 0);
            setTimeout(() => {
                WebGLUtils.animateVector3(mesh.position, targetPos, 900, WebGLUtils.Ease.outBack);
                WebGLUtils.animateValue(mesh.material, 'opacity', 0, mesh.userData.baseOpacity, 700);
                if (b.val) {
                    WebGLUtils.animateValue(mesh.material, 'emissiveIntensity', 0, mesh.userData.baseEmissive, 900);
                }
            }, i * 40);
        });

        // Group bracket lines (connecting groups)
        this._buildBrackets(startX, boxSize, gap);

        // Group labels
        this._buildLabels(startX, boxSize, gap);

        // Formula floating text
        this._buildFormulaFloating();

        // Value display
        const valSp = WebGLUtils.createTextSprite(this.data.value.toExponential(6), { fontSize: 32, color: '#f1f5f9' });
        valSp.position.set(0, -2.8, 0);
        valSp.material.opacity = 0;
        this.group.add(valSp);
        setTimeout(() => WebGLUtils.animateValue(valSp.material, 'opacity', 0, 1, 1000), 1800);

        // IEEE754 structure line
        const lineGeo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(startX - 0.5, 1.1, 0),
            new THREE.Vector3(startX + totalW + 0.1, 1.1, 0)
        ]);
        const lineMat = new THREE.LineBasicMaterial({ color: 0x1e293b, transparent: true, opacity: 0 });
        const line = new THREE.Line(lineGeo, lineMat);
        this.group.add(line);
        setTimeout(() => WebGLUtils.animateValue(lineMat, 'opacity', 0, 0.85, 800), 2000);

        // Bit index markers (small ticks)
        for (let i = 0; i <= 32; i += 4) {
            const tickGeo = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(startX + i * (boxSize + gap) - boxSize/2 - gap/2, 1.05, 0),
                new THREE.Vector3(startX + i * (boxSize + gap) - boxSize/2 - gap/2, 1.25, 0)
            ]);
            const tickMat = new THREE.LineBasicMaterial({ color: 0x475569, transparent: true, opacity: 0 });
            const tick = new THREE.Line(tickGeo, tickMat);
            this.group.add(tick);
            setTimeout(() => WebGLUtils.animateValue(tickMat, 'opacity', 0, 0.9, 600), 2200 + i * 30);
        }
    }

    _buildBrackets(startX, boxSize, gap) {
        // Bracket geometries for each group
        const groups = [
            { start: 0, count: 1, color: 0xef4444, y: 1.4 },
            { start: 1, count: 8, color: 0xf59e0b, y: 1.4 },
            { start: 9, count: 23, color: 0x06b6d4, y: 1.4 }
        ];
        groups.forEach((g, gi) => {
            const x1 = startX + g.start * (boxSize + gap) - gap/2 - 0.1;
            const x2 = startX + (g.start + g.count) * (boxSize + gap) - gap/2 + 0.1;
            const y = g.y;
            const h = 0.35;
            const pts = [
                new THREE.Vector3(x1, y, 0),
                new THREE.Vector3(x1, y + h, 0),
                new THREE.Vector3(x2, y + h, 0),
                new THREE.Vector3(x2, y, 0)
            ];
            const geo = new THREE.BufferGeometry().setFromPoints(pts);
            const mat = new THREE.LineBasicMaterial({ color: g.color, transparent: true, opacity: 0 });
            const line = new THREE.Line(geo, mat);
            this.group.add(line);
            this.connectors.push(line);
            setTimeout(() => WebGLUtils.animateValue(mat, 'opacity', 0, 0.95, 800), 1500 + gi * 200);
        });
    }

    _buildLabels(startX, boxSize, gap) {
        this.groupLabels = [];
        const labelData = [
            { text: 'Sign', start: 0, count: 1, color: '#ef4444', y: 2.0 },
            { text: 'Exponent', start: 1, count: 8, color: '#f59e0b', y: 2.0 },
            { text: 'Mantissa', start: 9, count: 23, color: '#06b6d4', y: 2.0 }
        ];
        labelData.forEach((l, i) => {
            const cx = startX + (l.start + l.count / 2) * (boxSize + gap) - boxSize / 2;
            const sp = WebGLUtils.createTextSprite(l.text, { fontSize: 22, color: l.color });
            sp.position.set(cx, l.y, 0);
            sp.material.opacity = 0;
            this.group.add(sp);
            this.groupLabels.push(sp);
            setTimeout(() => WebGLUtils.animateValue(sp.material, 'opacity', 0, 1, 700), 1600 + i * 200);
        });
    }

    _buildFormulaFloating() {
        this.formulaSprite = WebGLUtils.createTextSprite(
            `(-1)^${this.data.sign} × 2^(${this.data.exp}-127)`,
            { fontSize: 22, color: '#e2e8f0' }
        );
        this.formulaSprite.position.set(0, -1.6, 0);
        this.formulaSprite.material.opacity = 0;
        this.group.add(this.formulaSprite);
    }

    _spawnRipple(pos, colorHex) {
        const geo = new THREE.RingGeometry(0.3, 0.35, 32);
        const mat = new THREE.MeshBasicMaterial({
            color: colorHex, transparent: true, opacity: 0.6,
            side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.copy(pos);
        mesh.position.z += 0.1;
        mesh.lookAt(pos.clone().add(new THREE.Vector3(0, 0, 1)));
        this.group.add(mesh);
        // Animate expand + fade
        const start = performance.now();
        const anim = () => {
            const t = (performance.now() - start) / 800;
            if (t >= 1) { this.group.remove(mesh); geo.dispose(); mat.dispose(); return; }
            const s = 1 + t * 4;
            mesh.scale.set(s, s, 1);
            mat.opacity = 0.6 * (1 - t);
            requestAnimationFrame(anim);
        };
        requestAnimationFrame(anim);
    }

    _showBitDetail(bit) {
        const typeNames = { sign: '符号位', exp: '指数位', mant: '尾数位' };
        const detail = `Bit ${bit.idx}: ${typeNames[bit.type]}\nValue: ${bit.val}`;
        console.log(detail);
    }

    update(sm, dt, t) {
        this.pulseT += dt;
        this.cubes.forEach((mesh, i) => {
            const b = mesh.userData.bit;
            if (b.val) {
                mesh.material.emissiveIntensity = mesh.userData.baseEmissive + Math.sin(this.pulseT * 3 + i * 0.5) * 0.25;
            }
            mesh.position.y = Math.sin(this.pulseT * 1.2 + i * 0.3) * 0.06;
        });
        this.group.rotation.y = Math.sin(t * 0.12) * 0.06;
    }

    exit(sm, done) {
        this.cubes.forEach((mesh, i) => {
            WebGLUtils.animateVector3(mesh.position, new THREE.Vector3(mesh.position.x, mesh.position.y, -25), 600, WebGLUtils.Ease.inOutCubic);
            WebGLUtils.animateValue(mesh.material, 'opacity', mesh.material.opacity, 0, 600);
        });
        this.connectors.forEach(l => {
            WebGLUtils.animateValue(l.material, 'opacity', l.material.opacity, 0, 400);
        });
        setTimeout(() => {
            sm.scene.remove(this.group);
            done && done();
        }, 700);
    }

    getInteractables() { return this.cubes; }

    _updateInfo(data) {
        const fb = document.getElementById('formulaBox');
        const it = document.getElementById('infoText');
        if (fb) fb.innerHTML = `<span class="comment"># IEEE 754 Float32 拆解</span><br>
<span class="keyword">def</span> extract_float_components(data):<br>
&nbsp;&nbsp;u32 = data.view(np.uint32)<br>
&nbsp;&nbsp;signs = (u32 &gt;&gt; 31) &amp; 0x1<br>
&nbsp;&nbsp;exps = (u32 &gt;&gt; 23) &amp; 0xFF<br>
&nbsp;&nbsp;mants = u32 &amp; 0x7FFFFF<br>
&nbsp;&nbsp;<span class="keyword">return</span> signs, exps, mants`;
        if (it) it.textContent = `Float32 由 32 个 Bit 组成：1 Bit 符号位、8 Bits 指数位（带 127 偏移）、23 Bits 尾数位。当前值: ${data.value.toExponential(6)}`;
        const s = document.getElementById('metricSign');
        const e = document.getElementById('metricExp');
        const m = document.getElementById('metricMant');
        if (s) s.innerHTML = `${data.sign}<span class="metric-unit">bit</span>`;
        if (e) e.innerHTML = `${data.exp}<span class="metric-unit">(biased)</span>`;
        if (m) m.innerHTML = `${data.mant.toString(16).toUpperCase().padStart(6,'0')}<span class="metric-unit">hex</span>`;
    }
}

if (typeof window !== 'undefined') window.Step1BitDecomposer = Step1BitDecomposer;
