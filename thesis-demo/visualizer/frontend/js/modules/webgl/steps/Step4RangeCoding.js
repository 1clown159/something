/**
 * Step 4: Range Coding - Probability Tunnel (Enhanced)
 */
class Step4RangeCoding {
    constructor() {
        this.group = new THREE.Group();
        this.segments = [];
        this.tunnelGroup = null;
        this.bits = [];
        this.symbol = 0;
        this.probs = [];
        this.cdf = [];
        this.phaseTimer = 0;
        this.tunnelDepth = 0;
    }

    async enter(sm, data) {
        sm.scene.add(this.group);
        if (data.apiData && data.apiData.probabilities) {
            this.probs = data.apiData.probabilities;
            this.symbol = data.apiData.actual_symbol !== undefined ? data.apiData.actual_symbol : (data.symbol || 128);
        } else {
            this.symbol = data.symbol || WebGLUtils.generateMockFloat32(data.sampleIndex || 0).exp;
            this.probs = WebGLUtils.generateProbs(this.symbol);
        }
        // Read encode data if available
        if(data.apiData && data.apiData.encode){
            this.encodeData = data.apiData.encode;
        }else{
            this.encodeData = null;
        }
        let cum = 0;
        this.cdf = this.probs.map(p => { cum += p; return cum; });

        this._buildMainBar(sm);
        this._buildTunnel(sm);
        this._buildBitStream(sm);
        this._buildFormulas(sm);
        if(this.encodeData) this._buildEncodeInfo(sm);
        sm.cameraController.setPosition(0, 4, 14, new THREE.Vector3(0, 0.5, 0), 1400);
        sm.setBloom(0.6, 0.45, 0.78);
    }

    _buildMainBar(sm) {
        const barW = 13;
        const barH = 1.4;
        const startX = -barW / 2;
        const startY = 2;
        let cumX = startX;

        for (let i = 0; i < 256; i++) {
            const w = this.probs[i] * barW;
            const hue = i / 256;
            const color = new THREE.Color().setHSL(hue, 0.8, 0.48);
            const geo = new THREE.BoxGeometry(Math.max(w * 0.98, 0.02), barH, 0.35);
            const mat = new THREE.MeshStandardMaterial({
                color, transparent: true, opacity: 0,
                emissive: color, emissiveIntensity: 0, roughness: 0.35, metalness: 0.55
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(cumX + w / 2, startY, 0);
            mesh.scale.z = 0.01;
            mesh.userData = {
                onHoverEnter: () => this._showInfoSprites(true),
                onHoverLeave: () => this._showInfoSprites(false)
            };
            this.group.add(mesh);
            this.segments.push(mesh);

            const delay = Math.abs(i - this.symbol) * 5;
            setTimeout(() => {
                WebGLUtils.animateValue(mat, 'opacity', 0, i === this.symbol ? 0.98 : 0.6, 500);
                WebGLUtils.animateValue(mesh.scale, 'z', 0.01, 1, 500);
                if (i === this.symbol) WebGLUtils.animateValue(mat, 'emissiveIntensity', 0, 0.9, 700);
            }, delay);
            cumX += w;
        }

        // Frame
        const frameGeo = new THREE.BoxGeometry(barW + 0.15, barH + 0.15, 0.4);
        const frameMat = new THREE.MeshBasicMaterial({ color: 0x334155, wireframe: true, transparent: true, opacity: 0 });
        const frame = new THREE.Mesh(frameGeo, frameMat);
        frame.position.set(0, startY, 0);
        this.group.add(frame);
        setTimeout(() => WebGLUtils.animateValue(frameMat, 'opacity', 0, 0.75, 800), 1200);

        // Arrow marker
        const arrowGeo = new THREE.ConeGeometry(0.18, 0.55, 8);
        const arrowMat = new THREE.MeshBasicMaterial({ color: 0xec4899, transparent: true, opacity: 0 });
        const arrow = new THREE.Mesh(arrowGeo, arrowMat);
        const symStart = (this.cdf[this.symbol] - this.probs[this.symbol]) * barW - barW/2 + this.probs[this.symbol]*barW/2;
        arrow.position.set(symStart, startY + barH/2 + 0.5, 0);
        arrow.rotation.z = Math.PI;
        this.group.add(arrow);
        setTimeout(() => WebGLUtils.animateValue(arrowMat, 'opacity', 0, 1, 600), 1800);

        // Labels
        const lbl0 = WebGLUtils.createTextSprite('0.0', { fontSize: 24, color: '#e2e8f0' });
        lbl0.position.set(-barW/2 - 0.3, startY - 1, 0);
        this.group.add(lbl0);

        const lbl1 = WebGLUtils.createTextSprite('1.0', { fontSize: 24, color: '#e2e8f0' });
        lbl1.position.set(barW/2 + 0.3, startY - 1, 0);
        this.group.add(lbl1);
    }

    _buildTunnel(sm) {
        this.tunnelGroup = new THREE.Group();
        this.tunnelGroup.position.set(0, -2, 0);

        // Build 3 nested tunnel levels for recursive zoom effect
        for (let level = 0; level < 3; level++) {
            const levelGroup = new THREE.Group();
            const scale = Math.pow(0.6, level);
            const tunnelW = 10 * scale;
            const localLow = this.cdf[this.symbol] - this.probs[this.symbol];
            const localRange = this.probs[this.symbol];
            let cumX = -tunnelW / 2;

            for (let i = 0; i < 256; i++) {
                const w = (this.probs[i] / localRange) * tunnelW * 0.06;
                if (w < 0.03) continue;
                const hue = i / 256;
                const color = new THREE.Color().setHSL(hue, 0.65, 0.5);
                const geo = new THREE.BoxGeometry(w * 0.95, 0.6 * scale, 0.2 * scale);
                const mat = new THREE.MeshStandardMaterial({
                    color, transparent: true, opacity: 0,
                    emissive: color, emissiveIntensity: 0, roughness: 0.4
                });
                const mesh = new THREE.Mesh(geo, mat);
                mesh.position.set(cumX + w / 2, 0, -level * 2.5);
                levelGroup.add(mesh);
                cumX += w;
            }

            // Frame
            const fGeo = new THREE.BoxGeometry(tunnelW + 0.1, 0.7 * scale, 0.25 * scale);
            const fMat = new THREE.MeshBasicMaterial({ color: 0x334155, wireframe: true, transparent: true, opacity: 0 });
            levelGroup.add(new THREE.Mesh(fGeo, fMat));

            levelGroup.scale.set(0.01, 0.01, 0.01);
            this.tunnelGroup.add(levelGroup);

            // Staggered entry
            setTimeout(() => {
                WebGLUtils.animateValue(levelGroup.scale, 'x', 0.01, 1, 600, WebGLUtils.Ease.outBack);
                WebGLUtils.animateValue(levelGroup.scale, 'y', 0.01, 1, 600, WebGLUtils.Ease.outBack);
                WebGLUtils.animateValue(levelGroup.scale, 'z', 0.01, 1, 600, WebGLUtils.Ease.outBack);
                levelGroup.children.forEach(c => {
                    if (c.material && !c.material.wireframe) {
                        WebGLUtils.animateValue(c.material, 'opacity', 0, 0.65, 500);
                    }
                });
            }, 2000 + level * 300);
        }

        // Recursive zoom label (small, unobtrusive)
        const zoomLbl = WebGLUtils.createTextSprite('递归缩窄 ×3', { fontSize: 16, color: '#94a3b8' });
        zoomLbl.position.set(0, 1.2, 0);
        zoomLbl.material.opacity = 0;
        this.tunnelGroup.add(zoomLbl);
        setTimeout(() => WebGLUtils.animateValue(zoomLbl.material, 'opacity', 0, 0.7, 500), 2800);

        this.group.add(this.tunnelGroup);
    }

    _buildBitStream(sm) {
        const bitsY = -5;
        const bitSize = 0.32;

        for (let i = 0; i < 20; i++) {
            const val = (this.symbol >> (i % 8)) & 1;
            const geo = new THREE.BoxGeometry(bitSize, bitSize, bitSize);
            const mat = new THREE.MeshStandardMaterial({
                color: val ? 0x10b981 : 0x9ca3af,
                emissive: val ? 0x10b981 : 0x000000,
                emissiveIntensity: 0, transparent: true, opacity: 0,
                roughness: 0.3, metalness: 0.7
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(-3.2 + i * 0.36, bitsY, 0);
            this.group.add(mesh);
            this.bits.push(mesh);

            // Emit one by one
            setTimeout(() => {
                WebGLUtils.animateValue(mat, 'opacity', 0, 1, 300);
                if (val) WebGLUtils.animateValue(mat, 'emissiveIntensity', 0, 0.6, 400);
            }, 3000 + i * 120);
        }

        const lbl = WebGLUtils.createTextSprite('Output Bitstream', { fontSize: 18, color: '#94a3b8' });
        lbl.position.set(0, bitsY - 0.9, 0);
        lbl.material.opacity = 0;
        this.group.add(lbl);
        setTimeout(() => WebGLUtils.animateValue(lbl.material, 'opacity', 0, 1, 500), 3200);
    }

    _buildFormulas(sm) {
        this.formulaSprites = [];
        const formulas = [
            `range = high - low + 1`,
            `low += range × cdf[${this.symbol}]`,
            `range = range × prob[${this.symbol}]`
        ];
        formulas.forEach((f, i) => {
            const sp = WebGLUtils.createTextSprite(f, { fontSize: 18, color: '#e2e8f0' });
            sp.position.set(5, 1 - i * 0.55, 0);
            sp.material.opacity = 0;
            this.group.add(sp);
            this.formulaSprites.push(sp);
        });
    }

    _buildEncodeInfo(sm) {
        const e = this.encodeData;
        if(!e) return;
        this.encodeSprites = [];
        const infos = [
            `Range: [${e.range_low.toFixed(4)}, ${e.range_high.toFixed(4)})`,
            `Prob: ${(e.prob*100).toFixed(2)}%`,
            `Bits: ${e.bits_output}  (${e.encoded_count}/${e.total_coords})`
        ];
        infos.forEach((text, i) => {
            const sp = WebGLUtils.createTextSprite(text, { fontSize: 18, color: '#10b981' });
            sp.position.set(-6, -1.5 - i * 0.55, 0);
            sp.material.opacity = 0;
            this.group.add(sp);
            this.encodeSprites.push(sp);
        });
    }

    _showInfoSprites(show) {
        const target = show ? 1 : 0;
        (this.formulaSprites || []).forEach(sp => {
            WebGLUtils.animateValue(sp.material, 'opacity', sp.material.opacity, target, 250);
        });
        (this.encodeSprites || []).forEach(sp => {
            WebGLUtils.animateValue(sp.material, 'opacity', sp.material.opacity, target, 250);
        });
    }

    update(sm, dt, t) {
        this.phaseTimer += dt;
        // Tunnel breathing
        if (this.tunnelGroup) {
            this.tunnelGroup.children.forEach((child, i) => {
                if (child.type === 'Group') {
                    child.scale.x = 1 + Math.sin(this.phaseTimer * 0.5 + i) * 0.02;
                }
            });
        }
        // Bits shimmer
        this.bits.forEach((b, i) => {
            if (b.material.emissiveIntensity > 0) {
                b.material.emissiveIntensity = 0.6 + Math.sin(t * 5 + i) * 0.2;
            }
            b.position.y = -5 + Math.sin(t * 2 + i * 0.5) * 0.03;
        });
        // Segments subtle pulse
        this.segments.forEach((seg, i) => {
            if (i === this.symbol) {
                seg.material.emissiveIntensity = 0.9 + Math.sin(t * 4) * 0.2;
            }
        });
    }

    exit(sm, done) {
        this.segments.forEach(s => {
            WebGLUtils.animateValue(s.scale, 'z', s.scale.z, 0.01, 400);
            WebGLUtils.animateValue(s.material, 'opacity', s.material.opacity, 0, 400);
        });
        this.bits.forEach(b => {
            WebGLUtils.animateValue(b.scale, 'x', b.scale.x, 0.01, 400);
            WebGLUtils.animateValue(b.scale, 'y', b.scale.y, 0.01, 400);
        });
        setTimeout(() => { sm.scene.remove(this.group); done && done(); }, 500);
    }

    getInteractables() { return this.segments; }
}

if (typeof window !== 'undefined') window.Step4RangeCoding = Step4RangeCoding;
