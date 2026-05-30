/**
 * CameraController - Programmatic camera moves + OrbitControls
 */
class CameraController {
    constructor(camera, rendererDomElement) {
        this.camera = camera;
        this.controls = new THREE.OrbitControls(camera, rendererDomElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.enablePan = false;
        this.controls.minDistance = 2;
        this.controls.maxDistance = 40;
        this.controls.autoRotate = false;
        this.controls.autoRotateSpeed = 0.5;
        this.controls.maxPolarAngle = Math.PI * 0.85;

        this.autoPlaying = false;
        this._targetPos = new THREE.Vector3();
        this._targetLook = new THREE.Vector3();
        this._animating = false;
    }

    lookAt(point) {
        this.controls.target.copy(point);
        this.controls.update();
    }

    setPosition(x, y, z, lookAtPoint, duration = 1200, easing = WebGLUtils.Ease.inOutCubic) {
        if (this._animating) return;
        this._animating = true;
        const fromPos = this.camera.position.clone();
        const toPos = new THREE.Vector3(x, y, z);
        const fromLook = this.controls.target.clone();
        const toLook = lookAtPoint ? new THREE.Vector3(lookAtPoint.x, lookAtPoint.y, lookAtPoint.z) : fromLook;
        const start = performance.now();

        const tick = (now) => {
            let t = Math.min((now - start) / duration, 1);
            t = easing(t);
            this.camera.position.lerpVectors(fromPos, toPos, t);
            this.controls.target.lerpVectors(fromLook, toLook, t);
            this.controls.update();
            if (t < 1) {
                requestAnimationFrame(tick);
            } else {
                this._animating = false;
            }
        };
        requestAnimationFrame(tick);
    }

    orbit(speed = 0.2) {
        this.controls.autoRotate = true;
        this.controls.autoRotateSpeed = speed;
    }

    stopOrbit() {
        this.controls.autoRotate = false;
    }

    reset() {
        this.setPosition(0, 4, 12, new THREE.Vector3(0, 0, 0), 1500);
    }

    update() {
        this.controls.update();
    }
}

if (typeof window !== 'undefined') window.CameraController = CameraController;
