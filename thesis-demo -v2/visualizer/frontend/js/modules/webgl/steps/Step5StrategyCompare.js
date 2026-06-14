/**
 * Step 5: Sign & Mantissa Strategy Comparison - Strategy Arena (Enhanced)
 */
class Step5StrategyCompare {
    constructor() {
        this.group = new THREE.Group();
        this.pillars = [];
        this.connectors = [];
        this.halos = [];
        this.strategies = [
            { name: 'Bitpack', sign: 0.125, mant: 2.875, total: 3.0, color: 0x06b6d4 },
            { name: 'RLE', sign: 0.05, mant: 2.1, total: 2.15, color: 0xf59e0b },
            { name: 'Deflate', sign: 0.08, mant: 1.8, total: 1.88, color: 0x10b981 },
            { name: 'lzma', sign: 0.07, mant: 1.6, total: 1.67, color: 0x8b5cf6 },
            { name: 'PathL-lite', sign: 0.06, mant: 1.4, total: 1.46, color: 0xec4899, best: true }
        ];
    }

    async enter(sm, data) {
        sm.scene.add(this.group);
        this._build(sm);
        sm.cameraController.setPosition(0, 6, 16, new THREE.Vector3(0, 2, 0), 1400);
        sm.setBloom(0.55, 0.4, 0.8);
    }

    _build(sm) {
        const maxTotal = 3.0;
        const spacing = 3.2;
        const startX = -((this.strategies.length - 1) * spacing) / 2;

        this.strategies.forEach((s, i) => {
            const h = (s.total / maxTotal) * 5.5;
            const x = startX + i * spacing;

            // Main pillar
            const geo = new THREE.CylinderGeometry(0.75, 0.95, 1, 28);
            const mat = new THREE.MeshStandardMaterial({
                color: s.color, roughness: 0.25, metalness: 0.75,
                emissive: s.color, emissiveIntensity: 0, transparent: true, opacity: 0
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(x, 0, 0);
            mesh.scale.y = 0.01;
            this.group.add(mesh);
            this.pillars.push(mesh);

            // Grow animation
            setTimeout(() => {
                WebGLUtils.animateValue(mesh.scale, 'y', 0.01, h, 900 + i * 120, WebGLUtils.Ease.outBack);
                WebGLUtils.animateValue(mat, 'opacity', 0, 0.92, 700);
                WebGLUtils.animateValue(mat, 'emissiveIntensity', 0, s.best ? 0.5 : 0.12, 800);
            }, 400 + i * 180);

            // Sign bar (inner red section)
            const signRatio = s.sign / s.total;
            const signH = signRatio * h;
            const signGeo = new THREE.CylinderGeometry(0.6, 0.6, 1, 24);
            const signMat = new THREE.MeshStandardMaterial({
                color: 0xef4444, emissive: 0xef4444, emissiveIntensity: 0,
                transparent: true, opacity: 0, roughness: 0.3, metalness: 0.6
            });
            const signMesh = new THREE.Mesh(signGeo, signMat);
            signMesh.position.set(x, signH / 2, 0);
            signMesh.scale.y = 0.01;
            this.group.add(signMesh);
            setTimeout(() => {
                WebGLUtils.animateValue(signMesh.scale, 'y', 0.01, signH, 1000 + i * 120);
                WebGLUtils.animateValue(signMat, 'opacity', 0, 0.75, 800);
                WebGLUtils.animateValue(signMat, 'emissiveIntensity', 0, 0.25, 700);
            }, 800 + i * 180);

            // Sign label
            const signLbl = WebGLUtils.createTextSprite(`Sign ${s.sign.toFixed(2)}B`, { fontSize: 16, color: '#fca5a5' });
            signLbl.position.set(x + 1.1, signH / 2, 0);
            signLbl.scale.setScalar(0.65);
            signLbl.material.opacity = 0;
            this.group.add(signLbl);
            setTimeout(() => WebGLUtils.animateValue(signLbl.material, 'opacity', 0, 1, 500), 1400 + i * 120);

            // Name label
            const nameLbl = WebGLUtils.createTextSprite(s.name, { fontSize: 28, color: '#0f172a' });
            nameLbl.position.set(x, -0.7, 0);
            nameLbl.scale.setScalar(1.1);
            nameLbl.material.opacity = 0;
            this.group.add(nameLbl);
            setTimeout(() => WebGLUtils.animateValue(nameLbl.material, 'opacity', 0, 1, 500), 1200 + i * 120);

            // Total value
            const valLbl = WebGLUtils.createTextSprite(`${s.total.toFixed(2)}B`, { fontSize: 26, color: s.best ? '#ec4899' : '#334155' });
            valLbl.position.set(x, h + 0.6, 0);
            valLbl.scale.setScalar(1.0);
            valLbl.material.opacity = 0;
            this.group.add(valLbl);
            setTimeout(() => WebGLUtils.animateValue(valLbl.material, 'opacity', 0, 1, 500), 1400 + i * 120);

            // Best halo
            if (s.best) {
                const haloGeo = new THREE.TorusGeometry(1.3, 0.035, 8, 40);
                const haloMat = new THREE.MeshBasicMaterial({
                    color: 0xec4899, transparent: true, opacity: 0,
                    blending: THREE.AdditiveBlending
                });
                const halo = new THREE.Mesh(haloGeo, haloMat);
                halo.position.set(x, h + 1.3, 0);
                halo.rotation.x = Math.PI / 2;
                this.group.add(halo);
                this.halos.push(halo);
                setTimeout(() => WebGLUtils.animateValue(haloMat, 'opacity', 0, 0.7, 800), 1600);

                // Star icon
                const star = WebGLUtils.createTextSprite('★ 推荐', { fontSize: 20, color: '#ec4899' });
                star.position.set(x + 1.6, h + 1.3, 0);
                star.scale.setScalar(0.8);
                star.material.opacity = 0;
                this.group.add(star);
                setTimeout(() => WebGLUtils.animateValue(star.material, 'opacity', 0, 1, 500), 1700);
            }
        });

        // Comparison connectors (lines between pillars)
        for (let i = 0; i < this.strategies.length - 1; i++) {
            const s1 = this.strategies[i];
            const s2 = this.strategies[i+1];
            const x1 = startX + i * spacing;
            const x2 = startX + (i+1) * spacing;
            const y1 = (s1.total / maxTotal) * 5.5;
            const y2 = (s2.total / maxTotal) * 5.5;
            const pts = [new THREE.Vector3(x1 + 0.8, y1, 0), new THREE.Vector3(x2 - 0.8, y2, 0)];
            const geo = new THREE.BufferGeometry().setFromPoints(pts);
            const mat = new THREE.LineBasicMaterial({ color: 0x475569, transparent: true, opacity: 0 });
            const line = new THREE.Line(geo, mat);
            this.group.add(line);
            this.connectors.push(line);
            setTimeout(() => WebGLUtils.animateValue(mat, 'opacity', 0, 0.3, 600), 1800 + i * 100);
        }

        // Title
        const title = WebGLUtils.createTextSprite('Sign & Mantissa 压缩策略对比', { fontSize: 40, color: '#0f172a' });
        title.position.set(0, 7, 0);
        title.scale.setScalar(2.0);
        title.material.opacity = 0;
        this.group.add(title);
        setTimeout(() => WebGLUtils.animateValue(title.material, 'opacity', 0, 1, 800), 300);

        // Ground grid
        const gridHelper = new THREE.GridHelper(22, 22, 0x9ca3af, 0xdde2e8);
        gridHelper.position.y = -1;
        gridHelper.material.transparent = true;
        gridHelper.material.opacity = 0;
        this.group.add(gridHelper);
        setTimeout(() => WebGLUtils.animateValue(gridHelper.material, 'opacity', 0, 0.3, 1000), 500);
    }

    update(sm, dt, t) {
        this.pillars.forEach((p, i) => {
            if (this.strategies[i].best) {
                p.material.emissiveIntensity = 0.5 + Math.sin(t * 3) * 0.2;
            }
        });
        this.halos.forEach((h, i) => {
            h.rotation.z += dt * 1.8;
            h.position.y += Math.sin(t * 2.5 + i) * 0.002;
        });
    }

    exit(sm, done) {
        this.pillars.forEach(p => {
            WebGLUtils.animateValue(p.scale, 'y', p.scale.y, 0.01, 400);
            WebGLUtils.animateValue(p.material, 'opacity', p.material.opacity, 0, 400);
        });
        this.connectors.forEach(l => {
            WebGLUtils.animateValue(l.material, 'opacity', l.material.opacity, 0, 300);
        });
        setTimeout(() => { sm.scene.remove(this.group); done && done(); }, 500);
    }

    getInteractables() { return this.pillars; }
}

if (typeof window !== 'undefined') window.Step5StrategyCompare = Step5StrategyCompare;
