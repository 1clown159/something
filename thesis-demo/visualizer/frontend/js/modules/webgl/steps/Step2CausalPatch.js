/**
 * Step 2: Causal Patch Voxel Scan Field (Enhanced)
 */
class Step2CausalPatch {
    constructor() {
        this.group = new THREE.Group();
        this.voxels = [];
        this.scanLine = null;
        this.currentDot = null;
        this.patchGroup = null;
        this.gridSize = 10;
        this.cellSize = 0.75;
        this.diagIdx = 0;
        this.lastUpdate = 0;
        this.scanSpeed = 0.4;
    }

    async enter(sm, data) {
        sm.scene.add(this.group);
        this._buildVoxels(sm);
        this._buildPatch(sm);
        this._buildScanLine(sm);
        // If real coord available, highlight it
        if(data.apiData && data.apiData.coord){
            this.realCoord = data.apiData.coord;
            this._highlightRealCoord(sm);
        }
        sm.cameraController.setPosition(10, 8, 12, new THREE.Vector3(0, 1, 0), 1400);
        sm.setBloom(0.4, 0.35, 0.82);
    }

    _buildVoxels(sm) {
        const gs = this.gridSize;
        const cs = this.cellSize;
        const offset = (gs * cs) / 2 - cs / 2;
        const geo = new THREE.BoxGeometry(cs * 0.82, cs * 0.82, cs * 0.82);

        for (let z = 0; z < gs; z++) {
            for (let y = 0; y < gs; y++) {
                for (let x = 0; x < gs; x++) {
                    const diag = x + y + z;
                    const mat = new THREE.MeshStandardMaterial({
                        color: 0x6b7280,
                        transparent: true,
                        opacity: 0.65,
                        roughness: 0.7,
                        metalness: 0.2
                    });
                    const mesh = new THREE.Mesh(geo, mat);
                    const px = x * cs - offset;
                    const py = y * cs - offset + 2;
                    const pz = z * cs - offset;
                    mesh.position.set(px, py - 10, pz); // start from below
                    mesh.userData = { diag, x, y, z, basePos: new THREE.Vector3(px, py, pz) };
                    this.group.add(mesh);
                    this.voxels.push(mesh);

                    // Staggered rise
                    const delay = (x + y + z) * 50 + Math.random() * 200;
                    setTimeout(() => {
                        WebGLUtils.animateVector3(mesh.position, mesh.userData.basePos, 700, WebGLUtils.Ease.outBack);
                    }, delay);
                }
            }
        }

        // Current point
        const sphereGeo = new THREE.SphereGeometry(cs * 0.45, 24, 24);
        const sphereMat = new THREE.MeshStandardMaterial({
            color: 0xec4899, emissive: 0xec4899, emissiveIntensity: 1.5,
            transparent: true, opacity: 0
        });
        this.currentDot = new THREE.Mesh(sphereGeo, sphereMat);
        this.currentDot.position.set(0, -10, 0);
        this.group.add(this.currentDot);

        // Glow shell
        const glowGeo = new THREE.SphereGeometry(cs * 0.8, 16, 16);
        const glowMat = WebGLUtils.createGlowMaterial(0xec4899, 1.2);
        const glow = new THREE.Mesh(glowGeo, glowMat);
        this.currentDot.add(glow);
    }

    _buildPatch(sm) {
        this.patchGroup = new THREE.Group();
        const cell = 0.32;
        const center = 8;

        for (let r = 0; r < 17; r++) {
            for (let c = 0; c < 17; c++) {
                const dx = c - center;
                const dy = r - center;
                const isCausal = (dx < 0) || (dx === 0 && dy < 0) || (dx > 0 && dy < -dx);
                const isCenter = (r === center && c === center);
                let color = 0x6b7280;
                let opacity = 0.55;
                if (isCenter) { color = 0xec4899; opacity = 0.98; }
                else if (isCausal) {
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    const t = 1 - Math.min(dist / 12, 1);
                    color = new THREE.Color(0x06b6d4).lerp(new THREE.Color(0x0e7490), t).getHex();
                    opacity = 0.8 + t * 0.15;
                }
                const geo = new THREE.PlaneGeometry(cell * 0.9, cell * 0.9);
                const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide });
                const mesh = new THREE.Mesh(geo, mat);
                mesh.position.set((c - center) * cell, (center - r) * cell, 0);
                this.patchGroup.add(mesh);
            }
        }

        // Patch border
        const borderGeo = new THREE.EdgesGeometry(new THREE.PlaneGeometry(17 * cell, 17 * cell));
        const borderMat = new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.85 });
        const border = new THREE.LineSegments(borderGeo, borderMat);
        border.position.z = 0.01;
        this.patchGroup.add(border);

        this.patchGroup.position.set(6, 1, 4);
        this.patchGroup.rotation.x = -0.3;
        this.patchGroup.rotation.y = 0.4;
        this.patchGroup.scale.set(0.01, 0.01, 0.01);
        this.group.add(this.patchGroup);

        // Animate patch entry
        setTimeout(() => {
            WebGLUtils.animateValue(this.patchGroup.scale, 'x', 0.01, 1, 600, WebGLUtils.Ease.outBack);
            WebGLUtils.animateValue(this.patchGroup.scale, 'y', 0.01, 1, 600, WebGLUtils.Ease.outBack);
            WebGLUtils.animateValue(this.patchGroup.scale, 'z', 0.01, 1, 600, WebGLUtils.Ease.outBack);
        }, 1500);

        // Patch label
        const lbl = WebGLUtils.createTextSprite('17×17 Causal Patch', { fontSize: 20, color: '#e2e8f0' });
        lbl.position.set(6, 3.8, 4);
        lbl.material.opacity = 0;
        this.group.add(lbl);
        setTimeout(() => WebGLUtils.animateValue(lbl.material, 'opacity', 0, 1, 600), 1800);
    }

    _buildScanLine(sm) {
        const geo = new THREE.PlaneGeometry(8, 0.05);
        const mat = new THREE.MeshBasicMaterial({
            color: 0xec4899, transparent: true, opacity: 0,
            side: THREE.DoubleSide, blending: THREE.AdditiveBlending
        });
        this.scanLine = new THREE.Mesh(geo, mat);
        this.scanLine.position.set(0, -5, 0);
        this.scanLine.rotation.z = Math.PI / 4;
        this.group.add(this.scanLine);
    }

    _highlightRealCoord(sm) {
        // Add a marker for the real coord from backend
        const cs = this.cellSize;
        const offset = (this.gridSize * cs) / 2 - cs / 2;
        const c = this.realCoord;
        // Map coord to voxel position (scale down to fit in the 10×10 grid)
        const px = (c[1] % this.gridSize) * cs - offset;  // use trace for x
        const py = (c[2] % this.gridSize) * cs - offset + 2;  // use sample for y
        const pz = (c[0] % this.gridSize) * cs - offset;  // use profile for z

        const markerGeo = new THREE.SphereGeometry(cs * 0.5, 16, 16);
        const markerMat = new THREE.MeshStandardMaterial({
            color: 0xec4899, emissive: 0xec4899, emissiveIntensity: 2.0,
            transparent: true, opacity: 0
        });
        const marker = new THREE.Mesh(markerGeo, markerMat);
        marker.position.set(px, py, pz);
        this.group.add(marker);
        setTimeout(() => {
            WebGLUtils.animateValue(markerMat, 'opacity', 0, 0.9, 600);
        }, 1000);

        // Label
        const lbl = WebGLUtils.createTextSprite(
            '('+c[0]+','+c[1]+','+c[2]+')',
            { fontSize: 18, color: '#ec4899' }
        );
        lbl.position.set(px, py + 0.8, pz);
        lbl.material.opacity = 0;
        this.group.add(lbl);
        setTimeout(() => WebGLUtils.animateValue(lbl.material, 'opacity', 0, 1, 600), 1200);
    }

    _updateVoxelColors() {
        const d = this.diagIdx;
        let currentPos = new THREE.Vector3();
        this.voxels.forEach(v => {
            const diag = v.userData.diag;
            const isCurrent = diag === d;
            const isCausal = diag < d;
            let targetColor, targetOpacity, targetEmissive;
            if (isCurrent) {
                targetColor = 0xec4899;
                targetOpacity = 0.98;
                targetEmissive = 1.2;
                currentPos.copy(v.position);
            } else if (isCausal) {
                targetColor = 0x06b6d4;
                targetOpacity = 0.8;
                targetEmissive = 0.5;
            } else {
                targetColor = 0x6b7280;
                targetOpacity = 0.65;
                targetEmissive = 0;
            }
            // Smooth transition
            const dur = 300;
            const startCol = v.material.color.clone();
            const endCol = new THREE.Color(targetColor);
            const startTime = performance.now();
            const startOp = v.material.opacity;
            const startEm = v.material.emissiveIntensity;
            const animate = (now) => {
                const t = Math.min((now - startTime) / dur, 1);
                v.material.color.lerpColors(startCol, endCol, t);
                v.material.opacity = startOp + (targetOpacity - startOp) * t;
                v.material.emissiveIntensity = startEm + (targetEmissive - startEm) * t;
                if (t < 1) requestAnimationFrame(animate);
            };
            requestAnimationFrame(animate);
        });

        // Move current dot
        if (this.currentDot) {
            this.currentDot.material.opacity = 1;
            WebGLUtils.animateVector3(this.currentDot.position, currentPos, 400, WebGLUtils.Ease.outCubic);
        }

        // Update scan line
        if (this.scanLine) {
            const offset = (d / (this.gridSize * 3 - 1)) * 7 - 3.5;
            WebGLUtils.animateVector3(this.scanLine.position, new THREE.Vector3(offset, offset + 2, 0), 400);
            WebGLUtils.animateValue(this.scanLine.material, 'opacity', 0.6, 0, 600);
            setTimeout(() => WebGLUtils.animateValue(this.scanLine.material, 'opacity', 0, 0.6, 200), 100);
        }
    }

    update(sm, dt, t) {
        this.lastUpdate += dt;
        if (this.lastUpdate > this.scanSpeed) {
            this.lastUpdate = 0;
            this.diagIdx = (this.diagIdx + 1) % (this.gridSize * 3);
            this._updateVoxelColors();
        }
        if (this.currentDot) {
            const s = 1 + Math.sin(t * 5) * 0.12;
            this.currentDot.scale.setScalar(s);
            if (this.currentDot.children[0]) {
                this.currentDot.children[0].scale.setScalar(1.3 + Math.sin(t * 3.5) * 0.2);
            }
        }
        if (this.patchGroup) {
            this.patchGroup.position.y = 1 + Math.sin(t * 0.7) * 0.1;
            this.patchGroup.rotation.z = Math.sin(t * 0.3) * 0.02;
        }
        this.group.rotation.y = t * 0.06;
    }

    exit(sm, done) {
        this.voxels.forEach(v => {
            WebGLUtils.animateVector3(v.position, new THREE.Vector3(v.position.x, v.position.y - 15, v.position.z), 500);
            WebGLUtils.animateValue(v.material, 'opacity', v.material.opacity, 0, 500);
        });
        if (this.currentDot) {
            WebGLUtils.animateValue(this.currentDot.material, 'opacity', this.currentDot.material.opacity, 0, 400);
        }
        if (this.patchGroup) {
            WebGLUtils.animateValue(this.patchGroup.scale, 'x', 1, 0.01, 400);
            WebGLUtils.animateValue(this.patchGroup.scale, 'y', 1, 0.01, 400);
        }
        setTimeout(() => {
            sm.scene.remove(this.group);
            done && done();
        }, 600);
    }

    getInteractables() { return this.voxels; }
}

if (typeof window !== 'undefined') window.Step2CausalPatch = Step2CausalPatch;
