/**
 * API Module - Backend communication
 */

const API_BASE_URL = (() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('api') || 'http://localhost:8000';
})();

class Stage4API {
    constructor() {
        this.baseURL = API_BASE_URL;
    }

    /**
     * 上传文件
     */
    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post(`${this.baseURL}/api/upload`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            return response.data;
        } catch (error) {
            console.error('Upload error:', error);
            throw error;
        }
    }

    /**
     * 开始压缩任务
     */
    async startCompression(taskId, config) {
        try {
            const response = await axios.post(
                `${this.baseURL}/api/compress/${taskId}`,
                config
            );
            return response.data;
        } catch (error) {
            console.error('Compression error:', error);
            throw error;
        }
    }

    /**
     * 获取任务状态
     */
    async getStatus(taskId) {
        try {
            const response = await axios.get(`${this.baseURL}/api/status/${taskId}`);
            return response.data;
        } catch (error) {
            console.error('Status error:', error);
            throw error;
        }
    }

    /**
     * 提取特征数据
     */
    async extractFeatures(taskId, coord, patchShape, featureMode, targetMode) {
        try {
            const response = await axios.post(
                `${this.baseURL}/api/features/${taskId}`,
                {
                    coord: coord,
                    patch_shape: patchShape,
                    feature_mode: featureMode,
                    target_mode: targetMode
                }
            );
            return response.data;
        } catch (error) {
            console.error('Features error:', error);
            throw error;
        }
    }

    /**
     * 获取可视化数据
     */
    async getVisualizationData(taskId, dataType) {
        try {
            const response = await axios.get(
                `${this.baseURL}/api/visualize/${taskId}?data_type=${dataType}`
            );
            return response.data;
        } catch (error) {
            console.error('Visualization error:', error);
            throw error;
        }
    }

    /**
     * 获取压缩统计
     */
    async getCompressionStats(taskId) {
        try {
            const response = await axios.get(`${this.baseURL}/api/stats/${taskId}`);
            return response.data;
        } catch (error) {
            console.error('Stats error:', error);
            throw error;
        }
    }

    /**
     * 下载压缩结果
     */
    async downloadResult(taskId) {
        try {
            const response = await axios.get(
                `${this.baseURL}/api/download/${taskId}`,
                { responseType: 'blob' }
            );
            
            // Create download link
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', `compressed_${taskId}.s4rc`);
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.URL.revokeObjectURL(url);
            
            return true;
        } catch (error) {
            console.error('Download error:', error);
            throw error;
        }
    }

    /**
     * 轮询任务状态
     */
    async pollTaskStatus(taskId, onUpdate, interval = 2000, maxAttempts = 150) {
        let attempts = 0;
        
        const poll = async () => {
            try {
                const status = await this.getStatus(taskId);
                onUpdate(status);
                
                if (status.status === 'completed' || status.status === 'failed') {
                    return status;
                }
                
                attempts++;
                if (attempts >= maxAttempts) {
                    throw new Error('Polling timeout');
                }
                
                setTimeout(poll, interval);
            } catch (error) {
                onUpdate({ status: 'error', error: error.message });
            }
        };
        
        return poll();
    }
}

// Create global API instance
const api = new Stage4API();
