/**
 * Step Navigation - Common step-based navigation for demo/compress pages
 */

class StepNavigation {
    constructor(options = {}) {
        this.steps = options.steps || [];
        this.currentStep = options.startStep || 0;
        this.onStepChange = options.onStepChange || (() => {});
        this.stepBtns = [];
        this.sections = [];
        this._init();
    }

    _init() {
        this.stepBtns = Array.from(document.querySelectorAll('.step-btn'));
        this.sections = Array.from(document.querySelectorAll('.content-section'));

        this.stepBtns.forEach((btn, index) => {
            btn.addEventListener('click', () => {
                if (!btn.classList.contains('disabled')) {
                    this.goToStep(index);
                }
            });
        });
    }

    goToStep(stepIndex) {
        if (stepIndex < 0 || stepIndex >= this.steps.length) return;

        this.currentStep = stepIndex;

        this.stepBtns.forEach((btn, index) => {
            btn.classList.toggle('active', index === stepIndex);
            btn.classList.toggle('done', index < stepIndex);
        });

        this.sections.forEach((section, index) => {
            section.classList.toggle('active', index === stepIndex);
        });

        this.onStepChange(stepIndex, this.steps[stepIndex]);
    }

    next() { if (this.currentStep < this.steps.length - 1) this.goToStep(this.currentStep + 1); }
    prev() { if (this.currentStep > 0) this.goToStep(this.currentStep - 1); }
    reset() { this.goToStep(0); }

    setStepEnabled(stepIndex, enabled) {
        const btn = this.stepBtns[stepIndex];
        if (btn) {
            btn.classList.toggle('disabled', !enabled);
        }
    }

    markStepDone(stepIndex) {
        const btn = this.stepBtns[stepIndex];
        if (btn) btn.classList.add('done');
    }
}
