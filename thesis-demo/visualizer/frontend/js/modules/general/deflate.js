/**
 * Deflate Compression - Frontend Implementation
 * Uses browser's CompressionStream API (if available)
 */
const DeflateCompressor = (() => {
    const name = 'Deflate';

    async function deflateCompress(data) {
        if (!data || data.length === 0) {
            return new Uint8Array(0);
        }

        if (typeof CompressionStream !== 'undefined') {
            const cs = new CompressionStream('deflate');
            const writer = cs.writable.getWriter();
            writer.write(data);
            writer.close();
            const reader = cs.readable.getReader();
            const chunks = [];
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
            }
            return new Uint8Array(chunks.reduce((acc, chunk) => {
                const newArr = new Uint8Array(acc.length + chunk.length);
                newArr.set(acc);
                newArr.set(chunk, acc.length);
                return newArr;
            }, new Uint8Array(0)));
        } else {
            return new Uint8Array(data);
        }
    }

    async function deflateDecompress(data) {
        if (!data || data.length === 0) {
            return new Uint8Array(0);
        }

        if (typeof DecompressionStream !== 'undefined') {
            const ds = new DecompressionStream('deflate');
            const writer = ds.writable.getWriter();
            writer.write(data);
            writer.close();
            const reader = ds.readable.getReader();
            const chunks = [];
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
            }
            return new Uint8Array(chunks.reduce((acc, chunk) => {
                const newArr = new Uint8Array(acc.length + chunk.length);
                newArr.set(acc);
                newArr.set(chunk, acc.length);
                return newArr;
            }, new Uint8Array(0)));
        } else {
            return new Uint8Array(0);
        }
    }

    async function compress(data) {
        const startTime = performance.now();
        if (!data || data.length === 0) {
            return { data: new Uint8Array(0), originalSize: 0, compressedSize: 0 };
        }

        const compressed = await deflateCompress(data);
        const elapsed = performance.now() - startTime;

        return {
            data: compressed,
            originalSize: data.length,
            compressedSize: compressed.length,
            elapsed
        };
    }

    async function decompress(data) {
        return deflateDecompress(data);
    }

    function getStats(originalSize, compressedSize, elapsedTime) {
        const ratio = compressedSize > 0 ? originalSize / compressedSize : 0;
        const saving = originalSize > 0 ? (1 - compressedSize / originalSize) * 100 : 0;
        const throughput = elapsedTime > 0 ? originalSize / (elapsedTime * 1024 * 1024) : 0;

        return {
            original_size: originalSize,
            compressed_size: compressedSize,
            compression_ratio: Math.round(ratio * 100) / 100,
            space_saving: Math.round(saving * 10) / 10,
            compression_time: Math.round(elapsedTime * 1000 * 100) / 100,
            throughput: Math.round(throughput * 100) / 100,
            entropy: 0,
            algorithm: name
        };
    }

    return { name, compress, decompress, getStats, isAsync: true };
})();

if (typeof module !== 'undefined' && module.exports) {
    module.exports = DeflateCompressor;
}