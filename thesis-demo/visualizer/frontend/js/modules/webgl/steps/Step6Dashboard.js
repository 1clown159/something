/**
 * Step 6: Compression Dashboard - Reactor Core (Enhanced)
 */
class Step6Dashboard {
    constructor() {
        this.group = new THREE.Group();
        this.donutGroup = new THREE.Group();
        this.torusParts = [];
        this.orbitalLabels = [];
        this.particles = null;
        this.flowParticles = null;
        this.pulseRings = [];
    }

    async enter(sm, data) {
        sm.scene.add(this.group);
        // Use real stats from payload, or fallback
        var stats = data.stats || null;
        this._buildDonut(sm, stats);
        this._buildOrbitalMetrics(sm, stats);
        this._buildFlow(sm);
        this._buildPipeline(sm, stats);
        this._buildPulseRings(sm);
        sm.cameraController.setPosition(0, 5, 13, new THREE.Vector3(0, 0.5, 0), 1500);
        sm.setBloom(0.9, 0.65, 0.72);
    }

    _buildDonut(sm, stats) {
        var R = 2.4, r = 0.6;
        var expColor = 0xf59e0b, signColor = 0xef4444, mantColor = 0x06b6d4;
        var hasReal = stats && stats.compressed_size > 0;

        if (!hasReal) {
            // Placeholder — no real compression data yet
            var ph = WebGLUtils.createTextSprite('请先加载 SEG-Y 文件\n并进行压缩', { fontSize: 28, color: '#e2e8f0' });
            ph.position.set(0, 0, 0);
            ph.material.opacity = 0;
            this.group.add(ph);
            setTimeout(function() { WebGLUtils.animateValue(ph.material, 'opacity', 0, 1, 800); }, 400);
            return;
        }

        var expVal  = (stats.exponent_bytes || 0) / 1024;
        var signVal = (stats.sign_bytes || 0) / 1024;
        var mantVal = (stats.mant_bytes || 0) / 1024;
        var total   = expVal + signVal + mantVal || 1;
        var ratioText = (stats.compression_ratio || (stats.original_size / (stats.compressed_size || 1))).toFixed(2);

        const segs = [
            { color: expColor, frac: expVal / total, name: 'Exponent' },
            { color: signColor, frac: signVal / total, name: 'Sign' },
            { color: mantColor, frac: mantVal / total, name: 'Mantissa' }
        ];
        let startAngle = -Math.PI / 2;
        segs.forEach((s, si) => {
            const angleLen = s.frac * Math.PI * 2;
            const geo = new THREE.TorusGeometry(R, r, 20, Math.max(16, Math.floor(angleLen * 40)), angleLen);
            const mat = new THREE.MeshStandardMaterial({
                color: s.color, emissive: s.color, emissiveIntensity: 0,
                roughness: 0.15, metalness: 0.85, transparent: true, opacity: 0
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.rotation.z = startAngle;
            mesh.scale.set(0.01, 0.01, 0.01);
            mesh.userData = {
                segIndex: si,
                onHoverEnter: () => this._showSegLabel(si),
                onHoverLeave: () => this._hideSegLabel(si)
            };
            this.donutGroup.add(mesh);
            this.torusParts.push(mesh);

            setTimeout(() => {
                WebGLUtils.animateValue(mesh.scale, 'x', 0.01, 1, 900, WebGLUtils.Ease.outBack);
                WebGLUtils.animateValue(mesh.scale, 'y', 0.01, 1, 900, WebGLUtils.Ease.outBack);
                WebGLUtils.animateValue(mesh.scale, 'z', 0.01, 1, 900, WebGLUtils.Ease.outBack);
                WebGLUtils.animateValue(mat, 'opacity', 0, 0.95, 700);
                WebGLUtils.animateValue(mat, 'emissiveIntensity', 0, 0.5, 800);
            }, si * 250);

            startAngle += angleLen;
        });
        this.group.add(this.donutGroup);

        // Center text (add to group, not donutGroup, so it doesn't rotate)
        var centerLbl = WebGLUtils.createTextSprite(ratioText + ':1', { fontSize: 40, color: '#f1f5f9' });
        centerLbl.position.set(0, 0.2, 0);
        centerLbl.material.opacity = 0;
        this.group.add(centerLbl);
        setTimeout(() => WebGLUtils.animateValue(centerLbl.material, 'opacity', 0, 1, 800), 1000);

        const subLbl = WebGLUtils.createTextSprite('压缩比', { fontSize: 22, color: '#e2e8f0' });
        subLbl.position.set(0, -1.0, 0);
        subLbl.material.opacity = 0;
        this.group.add(subLbl);
        setTimeout(() => WebGLUtils.animateValue(subLbl.material, 'opacity', 0, 1, 600), 1200);

        // Section labels (outside the donut, hidden by default, shown on hover)
        this.segLabels = [];
        segs.forEach((s, si) => {
            const midAngle = (-Math.PI / 2) + segs.slice(0, si).reduce((sum, seg) => sum + seg.frac, 0) * Math.PI * 2 + s.frac * Math.PI;
            const lx = Math.cos(midAngle) * (R + 1.1);
            const ly = Math.sin(midAngle) * (R + 1.1);
            const lbl = WebGLUtils.createTextSprite(`${s.name}\n${(s.frac*total).toFixed(2)}KB`, { fontSize: 18, color: '#' + new THREE.Color(s.color).getHexString() });
            lbl.position.set(lx, ly, 0);
            lbl.material.opacity = 0;
            this.group.add(lbl);
            this.segLabels.push(lbl);
        });
    }

    _showSegLabel(idx) {
        if (!this.segLabels || !this.segLabels[idx]) return;
        this.segLabels.forEach((lbl, i) => {
            WebGLUtils.animateValue(lbl.material, 'opacity', lbl.material.opacity, i === idx ? 1 : 0, 250);
        });
    }

    _hideSegLabel(idx) {
        if (!this.segLabels || !this.segLabels[idx]) return;
        WebGLUtils.animateValue(this.segLabels[idx].material, 'opacity', this.segLabels[idx].material.opacity, 0, 250);
    }

    _buildOrbitalMetrics(sm, stats) {
        // Removed — orbital text was too large and cluttered the view.
        // Segment details now shown on hover via _showSegLabel / _hideSegLabel.
    }

    _buildFlow(sm) {
        // Rising particles from center bottom
        this.particles = new WebGLUtils.ParticleSystem(400, { size: 0.1, opacity: 0.85 });
        const pos = this.particles.positions;
        const vel = this.particles.velocities;
        for (let i = 0; i < 400; i++) {
            const angle = Math.random() * Math.PI * 2;
            const rad = Math.random() * 2;
            pos[i*3] = Math.cos(angle) * rad;
            pos[i*3+1] = -4 + Math.random() * 2;
            pos[i*3+2] = Math.sin(angle) * rad;
            vel[i*3] = (Math.random()-0.5)*0.008;
            vel[i*3+1] = 0.015 + Math.random()*0.025;
            vel[i*3+2] = (Math.random()-0.5)*0.008;
        }
        this.particles.geometry.attributes.position.needsUpdate = true;
        this.group.add(this.particles.mesh);
    }

    _buildPipeline(sm, stats) {
        var hasReal = stats && stats.compressed_size > 0;
        if (!hasReal) return;  // skip mock pipeline
        var origKB = ((stats.original_size || 0) / 1024).toFixed(1);
        var expKB  = ((stats.exponent_bytes || 0) / 1024).toFixed(1);
        var signKB = ((stats.sign_bytes || 0) / 1024).toFixed(1);
        var mantKB = ((stats.mant_bytes || 0) / 1024).toFixed(1);
        var totalKB = ((stats.compressed_size || 0) / 1024).toFixed(1);
        const stages = [
            { text: 'Raw\n' + origKB + 'KB', x: -4.0, color: 0x64748b },
            { text: 'Exp\n' + expKB + 'KB', x: -1.8, color: 0xf59e0b },
            { text: 'Sign\n' + signKB + 'KB', x: 0.0, color: 0xef4444 },
            { text: 'Mant\n' + mantKB + 'KB', x: 1.8, color: 0x06b6d4 },
            { text: 'Total\n' + totalKB + 'KB', x: 3.8, color: 0x10b981 }
        ];

        stages.forEach((s, i) => {
            const geo = new THREE.BoxGeometry(1.1, 0.65, 0.35);
            const mat = new THREE.MeshStandardMaterial({
                color: s.color, roughness: 0.3, metalness: 0.7,
                emissive: s.color, emissiveIntensity: 0.15, transparent: true, opacity: 0
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(s.x, -3.8, 0);
            this.group.add(mesh);
            setTimeout(() => {
                WebGLUtils.animateValue(mat, 'opacity', 0, 0.9, 500);
                WebGLUtils.animateValue(mat, 'emissiveIntensity', 0, 0.25, 500);
            }, 1800 + i * 100);

            const sp = WebGLUtils.createTextSprite(s.text, { fontSize: 16, color: '#f1f5f9' });
            sp.position.set(s.x, -4.5, 0.2);
            sp.material.opacity = 0;
            this.group.add(sp);
            setTimeout(() => WebGLUtils.animateValue(sp.material, 'opacity', 0, 1, 400), 1900 + i * 100);
        });

        // Flow particles along pipeline
        this.flowParticles = new WebGLUtils.ParticleSystem(100, { size: 0.07, opacity: 0.9 });
        const fpos = this.flowParticles.positions;
        for (let i = 0; i < 100; i++) {
            fpos[i*3] = -5.5 + Math.random()*10;
            fpos[i*3+1] = -3.8 + (Math.random()-0.5)*0.15;
            fpos[i*3+2] = (Math.random()-0.5)*0.25;
        }
        this.flowParticles.geometry.attributes.position.needsUpdate = true;
        this.group.add(this.flowParticles.mesh);
    }

    _buildPulseRings(sm) {
        for (let i = 0; i < 3; i++) {
            const geo = new THREE.RingGeometry(3.2 + i * 0.6, 3.25 + i * 0.6, 64);
            const mat = new THREE.MeshBasicMaterial({
                color: 0x667eea, transparent: true, opacity: 0,
                side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(0, 0, 0);
            mesh.rotation.x = Math.PI / 2;
            this.group.add(mesh);
            this.pulseRings.push({ mesh, mat, offset: i * 2.1, speed: 0.8 });
        }
    }

    update(sm, dt, t) {
        // Slowly rotate the whole donut
        this.donutGroup.rotation.z += dt * 0.05;

        // Orbital labels (disabled to reduce clutter)
        (this.orbitalLabels || []).forEach(ol => {
            ol.container.rotation.z = ol.baseAngle + t * ol.speed;
            ol.sp.position.y = Math.sin(t * 1.5 + ol.baseAngle) * 0.15;
        });

        // Rising particles
        if (this.particles) {
            const pos = this.particles.positions;
            const vel = this.particles.velocities;
            for (let i = 0; i < 400; i++) {
                pos[i*3] += vel[i*3];
                pos[i*3+1] += vel[i*3+1];
                pos[i*3+2] += vel[i*3+2];
                if (pos[i*3+1] > 6) {
                    const angle = Math.random() * Math.PI * 2;
                    const rad = Math.random() * 2;
                    pos[i*3] = Math.cos(angle) * rad;
                    pos[i*3+1] = -4;
                    pos[i*3+2] = Math.sin(angle) * rad;
                }
            }
            this.particles.geometry.attributes.position.needsUpdate = true;
        }

        // Flow particles
        if (this.flowParticles) {
            const fpos = this.flowParticles.positions;
            for (let i = 0; i < 100; i++) {
                fpos[i*3] += 0.025;
                if (fpos[i*3] > 5.5) fpos[i*3] = -5.5;
            }
            this.flowParticles.geometry.attributes.position.needsUpdate = true;
        }

        // Pulse rings
        this.pulseRings.forEach((pr, i) => {
            const phase = (t * pr.speed + pr.offset) % 3;
            const tNorm = phase / 3;
            pr.mat.opacity = 0.3 * Math.sin(tNorm * Math.PI);
            const s = 1 + tNorm * 0.3;
            pr.mesh.scale.set(s, s, 1);
        });
    }

    exit(sm, done) {
        this.torusParts.forEach(tp => {
            WebGLUtils.animateValue(tp.scale, 'x', 1, 0.01, 500);
            WebGLUtils.animateValue(tp.scale, 'y', 1, 0.01, 500);
        });
        (this.orbitalLabels || []).forEach(ol => {
            WebGLUtils.animateValue(ol.sp.material, 'opacity', ol.sp.material.opacity, 0, 400);
        });
        setTimeout(() => { sm.scene.remove(this.group); done && done(); }, 600);
    }

    getInteractables() { return this.torusParts; }
}

if (typeof window !== 'undefined') window.Step6Dashboard = Step6Dashboard;
