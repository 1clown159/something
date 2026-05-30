/**
 * SceneManager - Three.js scene, renderer, post-processing (Bloom)
 */
class SceneManager {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0f172a);
        this.scene.fog = new THREE.FogExp2(0x0f172a, 0.002);

        // Camera
        this.camera = new THREE.PerspectiveCamera(55, this.width / this.height, 0.1, 1000);
        this.camera.position.set(0, 4, 12);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
        this.renderer.setSize(this.width, this.height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.toneMapping = THREE.ReinhardToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        this.container.appendChild(this.renderer.domElement);
        this.renderer.domElement.style.display = 'block';
        this.renderer.domElement.style.width = '100%';
        this.renderer.domElement.style.height = '100%';

        // Post-processing (Bloom)
        this.composer = new THREE.EffectComposer(this.renderer);
        this.composer.addPass(new THREE.RenderPass(this.scene, this.camera));
        this.bloomPass = new THREE.UnrealBloomPass(
            new THREE.Vector2(this.width, this.height),
            0.6,  // strength
            0.4,  // radius
            0.85  // threshold
        );
        this.composer.addPass(this.bloomPass);

        // Lights
        this._setupLights();

        // Resize handler
        window.addEventListener('resize', () => this.onResize());

        // Animation loop state
        this.running = false;
        this._rafId = null;
        this.updaters = [];
        this.clock = new THREE.Clock();
    }

    _setupLights() {
        const ambient = new THREE.AmbientLight(0x707080, 1.2);
        this.scene.add(ambient);

        const dir = new THREE.DirectionalLight(0xffffff, 0.8);
        dir.position.set(5, 10, 7);
        this.scene.add(dir);

        const point = new THREE.PointLight(0x5b6af0, 1.2, 40);
        point.position.set(0, 5, 0);
        this.scene.add(point);
        this.mainLight = point;
    }

    onResize() {
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;
        this.camera.aspect = this.width / this.height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.width, this.height);
        this.composer.setSize(this.width, this.height);
    }

    addUpdater(fn) { this.updaters.push(fn); }
    removeUpdater(fn) { this.updaters = this.updaters.filter(u => u !== fn); }

    start() {
        if (this.running) return;
        this.running = true;
        const loop = () => {
            if (!this.running) return;
            this._rafId = requestAnimationFrame(loop);
            const dt = this.clock.getDelta();
            const t = this.clock.getElapsedTime();
            this.updaters.forEach(fn => fn(dt, t));
            this.composer.render();
        };
        loop();
    }

    stop() {
        this.running = false;
        if (this._rafId) cancelAnimationFrame(this._rafId);
    }

    setBloom(strength, radius, threshold) {
        this.bloomPass.strength = strength;
        this.bloomPass.radius = radius;
        this.bloomPass.threshold = threshold;
    }

    dispose() {
        this.stop();
        this.scene.traverse(o => {
            if (o.geometry) o.geometry.dispose();
            if (o.material) {
                if (Array.isArray(o.material)) o.material.forEach(m => m.dispose());
                else o.material.dispose();
            }
        });
        this.renderer.dispose();
        this.composer.dispose();
        if (this.renderer.domElement.parentNode) {
            this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
        }
    }
}

if (typeof window !== 'undefined') window.SceneManager = SceneManager;
