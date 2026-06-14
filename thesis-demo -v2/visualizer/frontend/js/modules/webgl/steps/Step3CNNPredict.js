/**
 * Step 3: CNN Probability Prediction - Probability Theater (Enhanced)
 */
class Step3CNNPredict {
    constructor() {
        this.group = new THREE.Group();
        this.bars = [];
        this.surface = null;
        this.cnnGroup = null;
        this.particleSystem = null;
        this.probs = [];
        this.sampleIndex = 0;
        this.beacon = null;
    }

    async enter(sm, data) {
        sm.scene.add(this.group);
        this.sampleIndex = data.sampleIndex || 0;
        let actualSymbol;
        if (data.apiData && data.apiData.probabilities) {
            this.probs = data.apiData.probabilities;
            actualSymbol = data.apiData.actual_symbol !== undefined ? data.apiData.actual_symbol : (data.symbol || 128);
        } else {
            const mock = WebGLUtils.generateMockFloat32(this.sampleIndex);
            this.probs = WebGLUtils.generateProbs(mock.exp);
            actualSymbol = mock.exp;
        }
        this._buildBars(sm, actualSymbol);
        this._buildSurface(sm, actualSymbol);
        this._buildCNN(sm);
        this._buildParticles(sm);
        sm.cameraController.setPosition(0, 7, 16, new THREE.Vector3(0, 1.5, 0), 1500);
        sm.setBloom(0.7, 0.5, 0.75);
    }

    _buildBars(sm, actualSymbol) {
        const radius = 6;
        const maxH = 5;
        const maxProb = Math.max(...this.probs);
        const barGeo = new THREE.BoxGeometry(0.16, 1, 0.16);
        barGeo.translate(0, 0.5, 0);

        for (let i = 0; i < 256; i++) {
            const angle = (i / 256) * Math.PI * 1.4 - Math.PI * 0.7;
            const h = (this.probs[i] / maxProb) * maxH;
            const hue = i / 256;
            const color = new THREE.Color().setHSL(hue, 0.85, 0.5);
            const mat = new THREE.MeshStandardMaterial({
                color, transparent: true, opacity: 0,
                emissive: color, emissiveIntensity: 0, roughness: 0.25, metalness: 0.7
            });
            const mesh = new THREE.Mesh(barGeo, mat);
            const x = Math.sin(angle) * radius;
            const z = Math.cos(angle) * radius;
            mesh.position.set(x, 0, z);
            mesh.scale.y = 0.01;
            mesh.lookAt(0, 0, 0);
            mesh.userData = { symbol: i, targetH: Math.max(h, 0.03), angle, radius };
            this.group.add(mesh);
            this.bars.push(mesh);

            // Staggered grow
            const delay = Math.abs(i - actualSymbol) * 6;
            setTimeout(() => {
                WebGLUtils.animateValue(mesh.scale, 'y', 0.01, mesh.userData.targetH, 700, WebGLUtils.Ease.outBack);
                WebGLUtils.animateValue(mesh.material, 'opacity', 0, 0.9, 500);
                if (i === actualSymbol) {
                    WebGLUtils.animateValue(mesh.material, 'emissiveIntensity', 0, 1.2, 800);
                } else if (this.probs[i] > maxProb * 0.3) {
                    WebGLUtils.animateValue(mesh.material, 'emissiveIntensity', 0, 0.4, 600);
                }
            }, delay);
        }

        // Beacon for actual symbol
        const actualAngle = (actualSymbol / 256) * Math.PI * 1.4 - Math.PI * 0.7;
        const beaconGeo = new THREE.CylinderGeometry(0.04, 0.04, 10, 8);
        const beaconMat = new THREE.MeshBasicMaterial({
            color: 0xec4899, transparent: true, opacity: 0, blending: THREE.AdditiveBlending
        });
        this.beacon = new THREE.Mesh(beaconGeo, beaconMat);
        this.beacon.position.set(Math.sin(actualAngle) * radius, 5, Math.cos(actualAngle) * radius);
        this.group.add(this.beacon);
        setTimeout(() => WebGLUtils.animateValue(beaconMat, 'opacity', 0, 0.75, 800), 1000);

        // Center value label
        const valSp = WebGLUtils.createTextSprite(`Symbol ${actualSymbol}`, { fontSize: 28, color: '#ec4899' });
        valSp.position.set(0, -1.8, 0);
        valSp.material.opacity = 0;
        this.group.add(valSp);
        setTimeout(() => WebGLUtils.animateValue(valSp.material, 'opacity', 0, 1, 800), 1200);
    }

    _buildSurface(sm, actualSymbol) {
        // Create a smooth surface connecting bar tops using TubeGeometry along the arc
        const radius = 6;
        const maxProb = Math.max(...this.probs);
        const points = [];
        for (let i = 0; i <= 256; i++) {
            const idx = Math.min(i, 255);
            const angle = (idx / 256) * Math.PI * 1.4 - Math.PI * 0.7;
            const h = (this.probs[idx] / maxProb) * 5 + 0.1;
            points.push(new THREE.Vector3(
                Math.sin(angle) * radius * 0.85,
                h,
                Math.cos(angle) * radius * 0.85
            ));
        }
        const curve = new THREE.CatmullRomCurve3(points);
        const tubeGeo = new THREE.TubeGeometry(curve, 128, 0.15, 8, false);
        const tubeMat = new THREE.MeshStandardMaterial({
            color: 0x667eea, transparent: true, opacity: 0,
            emissive: 0x667eea, emissiveIntensity: 0.3, roughness: 0.4, metalness: 0.6
        });
        this.surface = new THREE.Mesh(tubeGeo, tubeMat);
        this.group.add(this.surface);
        setTimeout(() => WebGLUtils.animateValue(tubeMat, 'opacity', 0, 0.65, 1200), 1500);
    }

    _buildCNN(sm) {
        this.cnnGroup = new THREE.Group();
        this.cnnGroup.position.set(-8, 2.5, 0);
        const layers = [
            { w: 2.2, h: 0.35, d: 2.2, c: 0x06b6d4, label: 'Conv2D 3×3', em: 0.4 },
            { w: 1.9, h: 0.28, d: 1.9, c: 0x0891b2, label: 'ReLU', em: 0.3 },
            { w: 1.5, h: 0.25, d: 1.5, c: 0x0e7490, label: 'Conv2D 3×3', em: 0.25 },
            { w: 1.1, h: 0.22, d: 1.1, c: 0x155e75, label: 'Conv2D 3×3', em: 0.2 },
            { w: 0.7, h: 0.55, d: 0.7, c: 0x1e3a8a, label: 'GAP → FC 256', em: 0.35 }
        ];
        this.layerMeshes = [];
        let y = 0;
        layers.forEach((layer, i) => {
            const geo = new THREE.BoxGeometry(layer.w, layer.h, layer.d);
            const mat = new THREE.MeshStandardMaterial({
                color: layer.c, roughness: 0.3, metalness: 0.75,
                emissive: layer.c, emissiveIntensity: 0, transparent: true, opacity: 0
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.y = y;
            this.cnnGroup.add(mesh);

            const lbl = WebGLUtils.createTextSprite(layer.label, { fontSize: 18, color: '#fff' });
            lbl.position.set(0, y, layer.d / 2 + 0.35);
            lbl.material.opacity = 0;
            this.cnnGroup.add(lbl);

            mesh.userData = {
                labelSprite: lbl,
                onHoverEnter: () => { WebGLUtils.animateValue(lbl.material, 'opacity', lbl.material.opacity, 1, 250); },
                onHoverLeave: () => { WebGLUtils.animateValue(lbl.material, 'opacity', lbl.material.opacity, 0, 250); }
            };
            this.layerMeshes.push(mesh);

            // Entry animation (label stays hidden until hover)
            setTimeout(() => {
                WebGLUtils.animateValue(mat, 'opacity', 0, 0.9, 500);
                WebGLUtils.animateValue(mat, 'emissiveIntensity', 0, layer.em, 600);
            }, 1000 + i * 150);

            y += layer.h + 0.35;
        });

        const title = WebGLUtils.createTextSprite('Small2DCNN', { fontSize: 24, color: '#e2e8f0' });
        title.position.set(0, y + 0.5, 0);
        title.material.opacity = 0;
        this.cnnGroup.add(title);
        setTimeout(() => WebGLUtils.animateValue(title.material, 'opacity', 0, 1, 500), 1800);

        this.group.add(this.cnnGroup);

        // Data flow tube
        const flowPts = [
            new THREE.Vector3(-8, 2.5, 0),
            new THREE.Vector3(-4, 2, 0),
            new THREE.Vector3(-1, 1.5, 0),
            new THREE.Vector3(0, 1, 0)
        ];
        const flowCurve = new THREE.CatmullRomCurve3(flowPts);
        const flowGeo = new THREE.TubeGeometry(flowCurve, 32, 0.025, 8, false);
        const flowMat = new THREE.MeshBasicMaterial({
            color: 0x66748b, transparent: true, opacity: 0,
            blending: THREE.AdditiveBlending
        });
        this.group.add(new THREE.Mesh(flowGeo, flowMat));
        setTimeout(() => WebGLUtils.animateValue(flowMat, 'opacity', 0, 0.7, 1000), 1200);
    }

    _buildParticles(sm) {
        this.particleSystem = new WebGLUtils.ParticleSystem(300, { size: 0.1, opacity: 0.9 });
        const pos = this.particleSystem.positions;
        const vel = this.particleSystem.velocities;
        for (let i = 0; i < 300; i++) {
            pos[i*3] = -8 + (Math.random()-0.5)*2;
            pos[i*3+1] = 2.5 + Math.random()*3;
            pos[i*3+2] = (Math.random()-0.5)*2;
            vel[i*3] = (Math.random()-0.5)*0.015;
            vel[i*3+1] = (Math.random()-0.5)*0.015;
            vel[i*3+2] = (Math.random()-0.5)*0.015;
        }
        this.particleSystem.geometry.attributes.position.needsUpdate = true;
        this.group.add(this.particleSystem.mesh);
    }

    update(sm, dt, t) {
        this.bars.forEach((bar, i) => {
            if (bar.userData.targetH > 2.5) {
                bar.material.emissiveIntensity = 0.4 + Math.sin(t * 3 + i * 0.2) * 0.15;
            }
        });

        // Surface pulse
        if (this.surface) {
            this.surface.material.emissiveIntensity = 0.3 + Math.sin(t * 2) * 0.1;
        }

        // Beacon pulse
        if (this.beacon) {
            this.beacon.material.opacity = 0.35 + Math.sin(t * 4) * 0.15;
        }

        // Particles flow from CNN to center then to bars
        if (this.particleSystem) {
            const pos = this.particleSystem.positions;
            const vel = this.particleSystem.velocities;
            for (let i = 0; i < 300; i++) {
                const px = pos[i*3], py = pos[i*3+1], pz = pos[i*3+2];
                const distToCenter = Math.sqrt(px*px + pz*pz);
                if (distToCenter > 2) {
                    vel[i*3] += (-px) * 0.0008;
                    vel[i*3+1] += (2 - py) * 0.0005;
                    vel[i*3+2] += (-pz) * 0.0008;
                } else {
                    // Push outward to bars
                    const angle = Math.atan2(px, pz);
                    vel[i*3] += Math.sin(angle) * 0.003;
                    vel[i*3+1] += (Math.random()-0.5)*0.002;
                    vel[i*3+2] += Math.cos(angle) * 0.003;
                }
                vel[i*3] *= 0.96;
                vel[i*3+1] *= 0.96;
                vel[i*3+2] *= 0.96;
                pos[i*3] += vel[i*3];
                pos[i*3+1] += vel[i*3+1];
                pos[i*3+2] += vel[i*3+2];

                if (distToCenter > 8 || py > 8 || py < -2) {
                    pos[i*3] = -8 + (Math.random()-0.5)*2;
                    pos[i*3+1] = 2.5 + Math.random()*3;
                    pos[i*3+2] = (Math.random()-0.5)*2;
                    vel[i*3] = 0; vel[i*3+1] = 0; vel[i*3+2] = 0;
                }
            }
            this.particleSystem.geometry.attributes.position.needsUpdate = true;
        }

        if (this.cnnGroup) {
            this.cnnGroup.position.y = 2.5 + Math.sin(t * 0.6) * 0.08;
        }
    }

    exit(sm, done) {
        this.bars.forEach(b => {
            WebGLUtils.animateValue(b.scale, 'y', b.scale.y, 0.01, 400);
            WebGLUtils.animateValue(b.material, 'opacity', b.material.opacity, 0, 400);
        });
        if (this.surface) {
            WebGLUtils.animateValue(this.surface.material, 'opacity', this.surface.material.opacity, 0, 400);
        }
        setTimeout(() => { sm.scene.remove(this.group); done && done(); }, 500);
    }

    getInteractables() { return this.bars.concat(this.layerMeshes || []); }
}

if (typeof window !== 'undefined') window.Step3CNNPredict = Step3CNNPredict;
