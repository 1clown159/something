/**
 * DemoAPI v2 - Small Volume Pipeline Endpoints
 */
class DemoAPI {
    constructor() {
        this.base = (typeof API_BASE_URL !== 'undefined') ? API_BASE_URL : 'http://localhost:8000';
    }

    /** Upload SGY → extract 2×2×100 small volume → precompute probabilities */
    async uploadSGY(file) {
        const fd = new FormData();
        fd.append('file', file);
        try {
            const resp = await axios.post(`${this.base}/api/demo/upload-sgy`, fd, {
                headers: { 'Content-Type': 'multipart/form-data' },
                timeout: 180000  // 3 min for model loading
            });
            return resp.data;
        } catch (e) { console.warn('[DemoAPI] uploadSGY:', e.message); return null; }
    }

    /** Bit decomposition for a sample index */
    async decompose(taskId, sampleIndex) {
        try {
            const resp = await axios.post(`${this.base}/api/demo/decompose`, {
                task_id: taskId,
                sample_index: sampleIndex
            });
            return resp.data;
        } catch (e) { console.warn('[DemoAPI] decompose:', e.message); return null; }
    }

    /** Feature extraction (6-channel causal patch) */
    async features(taskId, sampleIndex) {
        try {
            const resp = await axios.post(`${this.base}/api/demo/features`, {
                task_id: taskId,
                sample_index: sampleIndex
            });
            return resp.data;
        } catch (e) { console.warn('[DemoAPI] features:', e.message); return null; }
    }

    /** CNN probability prediction */
    async predict(taskId, sampleIndex) {
        try {
            const resp = await axios.post(`${this.base}/api/demo/predict`, {
                task_id: taskId,
                sample_index: sampleIndex
            });
            return resp.data;
        } catch (e) { console.warn('[DemoAPI] predict:', e.message); return null; }
    }

    /** Range coding encode info */
    async encode(taskId, sampleIndex) {
        try {
            const resp = await axios.post(`${this.base}/api/demo/encode`, {
                task_id: taskId,
                sample_index: sampleIndex
            });
            return resp.data;
        } catch (e) { console.warn('[DemoAPI] encode:', e.message); return null; }
    }

    /** Compression stats for the whole small volume */
    async stats(taskId) {
        try {
            const resp = await axios.get(`${this.base}/api/demo/stats/${taskId}`);
            return resp.data;
        } catch (e) { console.warn('[DemoAPI] stats:', e.message); return null; }
    }

    // ----- Legacy methods (kept for compatibility) -----
    async loadSGY(file) {
        return this.uploadSGY(file);
    }
    async getStats(taskId) {
        return this.stats(taskId);
    }
}

if (typeof window !== 'undefined') window.DemoAPI = DemoAPI;
