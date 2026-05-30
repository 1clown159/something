/**
 * Verify Panel - 无损验证面板模块
 */

class VerifyPanel {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.apiBase = API_BASE_URL;
    }

    async verify(taskId) {
        this.container.innerHTML = '<div class="loading-spinner" style="margin:auto;"></div>';

        try {
            // Get both status and verify results
            const statusResp = await axios.get(`${this.apiBase}/api/status/${taskId}`);
            const status = statusResp.data;

            const verifyBox = document.getElementById('verifyStatusBox');
            const isMatch = status.verify_match;

            this.container.innerHTML = `
                <div style="text-align:center;padding:var(--spacing-xl);">
                    <div style="font-size:3rem;margin-bottom:var(--spacing-sm);">
                        ${isMatch ? '✅' : '❌'}
                    </div>
                    <div style="font-size:1.5rem;font-weight:700;color:${isMatch ? 'var(--success)' : 'var(--error)'};margin-bottom:var(--spacing-sm);">
                        ${isMatch ? '无损验证通过！' : '验证未通过'}
                    </div>
                    <p style="color:var(--text-secondary);margin-bottom:var(--spacing-md);">
                        ${isMatch ? '原始数据与解压后数据完全一致' : '数据存在差异'}
                    </p>
                </div>
                <table class="verify-table">
                    <thead>
                        <tr>
                            <th>指标</th>
                            <th>原始</th>
                            <th>压缩后</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>文件大小</td>
                            <td>${formatFileSize(status.output?.original_size || 0)}</td>
                            <td>${formatFileSize(status.output?.compressed_size || 0)}</td>
                        </tr>
                        <tr>
                            <td>压缩比</td>
                            <td colspan="2">${(status.output?.compression_ratio || 0).toFixed(2)}x</td>
                        </tr>
                        <tr>
                            <td>码率</td>
                            <td colspan="2">${(status.output?.bits_per_voxel || 0).toFixed(2)} bits/voxel</td>
                        </tr>
                        <tr>
                            <td>数据一致性</td>
                            <td colspan="2" class="${isMatch ? 'verify-match' : 'verify-mismatch'}">
                                ${isMatch ? '✓ 完全一致' : '✗ 不一致'}
                            </td>
                        </tr>
                    </tbody>
                </table>
            `;

            if (verifyBox) {
                verifyBox.innerHTML = isMatch
                    ? `<span style="color:var(--success);">✓ 无损编码：压缩后的符号序列可完全还原为原始数据</span>`
                    : `<span style="color:var(--error);">✗ 验证失败</span>`;
            }
        } catch (error) {
            this.container.innerHTML = `<div class="chart-placeholder">验证失败: ${error.message}<br>请先完成压缩操作</div>`;
        }
    }
}
