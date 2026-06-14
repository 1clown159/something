/**
 * StepRenderer - 6-step lifecycle manager
 */
class StepRenderer {
    constructor(sceneManager) {
        this.sm = sceneManager;
        this.currentStep = -1;
        this.steps = [];
        this.raycaster = null;
    }

    register(stepIndex, sceneBuilder) {
        this.steps[stepIndex] = sceneBuilder;
    }

    /** Exit current step and clean up (for switching to 2D mode) */
    async cleanup() {
        const prev = this.currentStep;
        if (prev >= 0 && this.steps[prev] && this.steps[prev].exit) {
            await this._runExit(this.steps[prev]);
        }
        this._disposeStepGroup(prev);
        this.currentStep = -1;
        this.raycaster = null;
    }

    async goto(stepIndex, data = {}) {
        const sameStep = stepIndex === this.currentStep;
        const prev = this.currentStep;
        this.currentStep = stepIndex;

        if (sameStep) {
            // Same step: dispose old group entirely, then re-enter
            this._disposeStepGroup(stepIndex);
            if (this.steps[stepIndex] && this.steps[stepIndex].enter) {
                await this.steps[stepIndex].enter(this.sm, data);
            }
            this._rebuildRaycaster(stepIndex);
            return;
        }

        // Exit previous (with animation)
        if (prev >= 0 && this.steps[prev] && this.steps[prev].exit) {
            await this._runExit(this.steps[prev]);
        }
        // Dispose previous step's leftover resources
        this._disposeStepGroup(prev);

        // Enter new
        if (this.steps[stepIndex] && this.steps[stepIndex].enter) {
            await this.steps[stepIndex].enter(this.sm, data);
        }

        this._rebuildRaycaster(stepIndex);
    }

    /** Thoroughly dispose a step's group and all children */
    _disposeStepGroup(stepIndex) {
        if (stepIndex < 0) return;
        const step = this.steps[stepIndex];
        if (!step || !step.group) return;
        // Dispose all geometries/materials recursively
        step.group.traverse(o => {
            if (o.geometry) o.geometry.dispose();
            if (o.material) {
                if (Array.isArray(o.material)) o.material.forEach(m => m.dispose());
                else o.material.dispose();
            }
        });
        // Remove all children from group
        while (step.group.children.length > 0) {
            step.group.remove(step.group.children[0]);
        }
        // Remove group from scene
        if (step.group.parent) step.group.parent.remove(step.group);
    }

    _rebuildRaycaster(stepIndex) {
        if (this.steps[stepIndex] && this.steps[stepIndex].getInteractables) {
            const objs = this.steps[stepIndex].getInteractables();
            this.raycaster = new RaycasterHelper(this.sm.camera, this.sm.renderer);
            this.raycaster.add(objs);
        } else {
            this.raycaster = null;
        }
    }

    _runExit(step) {
        return new Promise(resolve => {
            if (step.exit) step.exit(this.sm, resolve);
            else resolve();
        });
    }

    update(dt, t) {
        const step = this.steps[this.currentStep];
        if (step && step.update) step.update(this.sm, dt, t);
    }

    onPointerMove(e) {
        if (!this.raycaster) return;
        this.raycaster.onPointerMove(
            e.clientX, e.clientY,
            (obj, hit) => {
                if (obj.userData.onHoverEnter) obj.userData.onHoverEnter(obj, hit);
            },
            (obj) => {
                if (obj.userData.onHoverLeave) obj.userData.onHoverLeave(obj);
                WebGLUtils.Tooltip3D.hide();
            },
            (obj, hit) => {
                if (obj.userData.onHoverMove) obj.userData.onHoverMove(obj, hit);
            }
        );
    }

    onClick(e) {
        if (!this.raycaster) return;
        const rect = this.sm.renderer.domElement.getBoundingClientRect();
        const mouse = new THREE.Vector2(
            ((e.clientX - rect.left) / rect.width) * 2 - 1,
            -((e.clientY - rect.top) / rect.height) * 2 + 1
        );
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, this.sm.camera);
        const intersects = raycaster.intersectObjects(this.raycaster.objects, false);
        if (intersects.length > 0 && intersects[0].object.userData.onClick) {
            intersects[0].object.userData.onClick(intersects[0].object, intersects[0]);
        }
    }
}

if (typeof window !== 'undefined') window.StepRenderer = StepRenderer;
