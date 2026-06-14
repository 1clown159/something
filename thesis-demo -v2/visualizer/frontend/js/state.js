/**
 * State Management - Stage4 Visualizer
 */

const AppState = {
    taskId: null,
    fileName: null,
    fileSize: 0,
    isCompressing: false,
    currentSection: 'upload',
    currentStep: 0,

    reset() {
        this.taskId = null;
        this.fileName = null;
        this.fileSize = 0;
        this.isCompressing = false;
        this.currentSection = 'upload';
        this.currentStep = 0;
    },

    setTask(taskId, fileName, fileSize) {
        this.taskId = taskId;
        this.fileName = fileName;
        this.fileSize = fileSize;
    }
};
