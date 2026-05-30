/**
 * WebGL Utils - 3D visualization helpers
 */

// ---- Color Utilities ----
const ColorPalette = {
    sign:   0xef4444,
    exp:    0xf59e0b,
    mant:   0x06b6d4,
    causal: 0x06b6d4,
    current:0xec4899,
    future: 0x6b7280,
    accent: 0x5b6af0,
    grid:   0x9ca3af,
    text:   0x0f172a,
    muted:  0x1e293b,
    bg:     0xdde2e8
};

function hexToThree(hex) { return new THREE.Color(hex); }

function lerpColor(a, b, t) {
    return new THREE.Color(a).lerp(new THREE.Color(b), t);
}

// ---- Easing ----
const Ease = {
    outCubic: t => 1 - Math.pow(1 - t, 3),
    inOutCubic: t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2,
    outElastic: t => {
        const c4 = (2 * Math.PI) / 3;
        return t === 0 ? 0 : t === 1 ? 1 : Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
    },
    outBack: t => {
        const c1 = 1.70158;
        const c3 = c1 + 1;
        return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
    }
};

// ---- Animation helpers ----
function animateValue(obj, prop, from, to, duration, easing = Ease.outCubic, onDone) {
    const start = performance.now();
    function tick(now) {
        let t = Math.min((now - start) / duration, 1);
        t = easing(t);
        obj[prop] = from + (to - from) * t;
        if (t < 1) requestAnimationFrame(tick);
        else if (onDone) onDone();
    }
    requestAnimationFrame(tick);
}

function animateVector3(vec3, target, duration, easing = Ease.outCubic, onDone) {
    const start = performance.now();
    const fromX = vec3.x, fromY = vec3.y, fromZ = vec3.z;
    function tick(now) {
        let t = Math.min((now - start) / duration, 1);
        t = easing(t);
        vec3.set(
            fromX + (target.x - fromX) * t,
            fromY + (target.y - fromY) * t,
            fromZ + (target.z - fromZ) * t
        );
        if (t < 1) requestAnimationFrame(tick);
        else if (onDone) onDone();
    }
    requestAnimationFrame(tick);
}

// ---- 3D Text with Canvas Texture ----
function createTextSprite(text, opts = {}) {
    const fontSize = opts.fontSize || 48;
    const color = opts.color || '#0f172a';
    const bg = opts.background || 'transparent';
    const lineHeight = opts.lineHeight || 1.35;
    const padding = opts.padding || 16;
    const scaleBase = opts.scaleBase || 40; // back to original moderate scale
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Support multiline text
    const lines = String(text).split('\n');
    ctx.font = `bold ${fontSize}px "Inter","Segoe UI",sans-serif`;
    let maxW = 0;
    lines.forEach(function(line) {
        const m = ctx.measureText(line);
        if (m.width > maxW) maxW = m.width;
    });

    const w = Math.ceil(maxW) + padding * 2;
    const h = Math.ceil(fontSize * lineHeight * lines.length) + padding * 2;
    canvas.width = w;
    canvas.height = h;
    if (bg !== 'transparent') {
        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, w, h);
    }
    ctx.font = `bold ${fontSize}px "Inter","Segoe UI",sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = color;

    const startY = h / 2 - (fontSize * lineHeight * (lines.length - 1)) / 2;
    lines.forEach(function(line, i) {
        ctx.fillText(line, w / 2, startY + i * fontSize * lineHeight);
    });

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(w / scaleBase, h / scaleBase, 1);
    return sprite;
}

// ---- Tooltip system ----
const Tooltip3D = {
    el: null,
    lastX: 0, lastY: 0,
    init() {
        if (this.el) return;
        this.el = document.createElement('div');
        this.el.className = 'demo-tooltip';
        this.el.style.cssText = 'position:fixed;top:0;left:0;pointer-events:none;z-index:9999;display:none;';
        document.body.appendChild(this.el);
    },
    show(html, x, y) {
        if (!this.el) this.init();
        this.el.innerHTML = html;
        this.el.style.display = 'block';
        this.el.style.left = ((x || this.lastX) + 12) + 'px';
        this.el.style.top = ((y || this.lastY) - 12) + 'px';
    },
    hide() {
        if (this.el) this.el.style.display = 'none';
    },
    setPos(x, y) { this.lastX = x; this.lastY = y; }
};

// ---- Particle system base ----
class ParticleSystem {
    constructor(count, opts = {}) {
        this.count = count;
        this.geometry = new THREE.BufferGeometry();
        this.positions = new Float32Array(count * 3);
        this.velocities = new Float32Array(count * 3);
        this.colors = new Float32Array(count * 3);
        this.alphas = new Float32Array(count);
        this.sizes = new Float32Array(count);
        this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
        this.geometry.setAttribute('color', new THREE.BufferAttribute(this.colors, 3));
        this.geometry.setAttribute('alpha', new THREE.BufferAttribute(this.alphas, 1));
        this.geometry.setAttribute('size', new THREE.BufferAttribute(this.sizes, 1));
        this.material = new THREE.PointsMaterial({
            size: opts.size || 0.15,
            vertexColors: true,
            transparent: true,
            opacity: opts.opacity || 0.8,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            sizeAttenuation: true
        });
        this.mesh = new THREE.Points(this.geometry, this.material);
        this.reset();
    }
    reset() {
        for (let i = 0; i < this.count; i++) {
            this.positions[i*3] = 0;
            this.positions[i*3+1] = 0;
            this.positions[i*3+2] = 0;
            this.velocities[i*3] = (Math.random()-0.5)*0.02;
            this.velocities[i*3+1] = (Math.random()-0.5)*0.02;
            this.velocities[i*3+2] = (Math.random()-0.5)*0.02;
            this.colors[i*3] = 1;
            this.colors[i*3+1] = 1;
            this.colors[i*3+2] = 1;
            this.alphas[i] = 0;
            this.sizes[i] = 1;
        }
        this.geometry.attributes.position.needsUpdate = true;
    }
    update() {
        const pos = this.positions;
        const vel = this.velocities;
        for (let i = 0; i < this.count; i++) {
            pos[i*3]   += vel[i*3];
            pos[i*3+1] += vel[i*3+1];
            pos[i*3+2] += vel[i*3+2];
            this.alphas[i] *= 0.99;
        }
        this.geometry.attributes.position.needsUpdate = true;
        this.geometry.attributes.alpha.needsUpdate = true;
    }
}

// ---- Glow / Bloom helper ----
function createGlowMaterial(color, intensity = 1.5) {
    return new THREE.MeshBasicMaterial({
        color: color,
        transparent: true,
        opacity: 0.6 * intensity,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        side: THREE.BackSide
    });
}

// ---- Raycaster helper ----
class RaycasterHelper {
    constructor(camera, renderer) {
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        this.camera = camera;
        this.renderer = renderer;
        this.objects = [];
        this.hoverObj = null;
    }
    add(objects) {
        if (Array.isArray(objects)) this.objects.push(...objects);
        else this.objects.push(objects);
    }
    clear() { this.objects = []; this.hoverObj = null; }
    onPointerMove(clientX, clientY, onEnter, onLeave, onMove) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((clientY - rect.top) / rect.height) * 2 + 1;
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.objects, false);
        if (intersects.length > 0) {
            const obj = intersects[0].object;
            if (this.hoverObj !== obj) {
                if (this.hoverObj && onLeave) onLeave(this.hoverObj);
                this.hoverObj = obj;
                if (onEnter) onEnter(obj, intersects[0]);
            } else if (onMove) {
                onMove(obj, intersects[0]);
            }
        } else {
            if (this.hoverObj && onLeave) onLeave(this.hoverObj);
            this.hoverObj = null;
        }
    }
}

// ---- Float32 Decomposition (client-side) ----
function decomposeFloat32(value) {
    const buf = new ArrayBuffer(4);
    const f32 = new Float32Array(buf);
    const u32 = new Uint32Array(buf);
    f32[0] = value;
    const raw = u32[0];
    const sign = (raw >> 31) & 0x1;
    const exp  = (raw >> 23) & 0xFF;
    const mant = raw & 0x7FFFFF;
    return { value, raw, sign, exp, mant };
}

// ---- Mock Data Generator ----
function generateMockFloat32(idx) {
    const seed = idx * 997;
    const pseudo = (n) => {
        let x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
        return x - Math.floor(x);
    };
    const sign = pseudo(seed) > 0.5 ? 1 : 0;
    const exp = Math.floor(pseudo(seed + 1) * 40) + 110;
    const mant = Math.floor(pseudo(seed + 2) * 0x7FFFFF);
    const value = (sign ? -1 : 1) * Math.pow(2, exp - 127) * (1 + mant / 0x800000);
    return { value, sign, exp, mant };
}

function generateProbs(center) {
    const probs = new Array(256).fill(0);
    for (let i = 0; i < 256; i++) {
        const dist = Math.abs(i - center);
        probs[i] = Math.exp(-dist * dist / (2 * 15 * 15)) + Math.random() * 0.02;
    }
    const sum = probs.reduce((a, b) => a + b, 0);
    return probs.map(p => p / sum);
}

// ---- Exports ----
if (typeof window !== 'undefined') {
    window.WebGLUtils = {
        ColorPalette, hexToThree, lerpColor, Ease,
        animateValue, animateVector3,
        createTextSprite, Tooltip3D, ParticleSystem,
        createGlowMaterial, RaycasterHelper,
        decomposeFloat32, generateMockFloat32, generateProbs
    };
}
