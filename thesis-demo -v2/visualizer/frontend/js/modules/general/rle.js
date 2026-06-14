/**
 * RLE Compression - Frontend Implementation
 * Run-Length Encoding
 */
const RLECompressor = (() => {
    const name = 'RLE';

    function compress(data) {
        if (!data || data.length === 0) {
            return { data: new Uint8Array(0), originalSize: 0, compressedSize: 0 };
        }

        const result = [];
        let pos = 0;

        while (pos < data.length) {
            let run = 1;
            const currentByte = data[pos];

            while (pos + run < data.length &&
                   data[pos + run] === currentByte &&
                   run < 255) {
                run++;
            }

            if (run >= 3) {
                result.push(255, currentByte, run);
                pos += run;
            } else {
                result.push(currentByte);
                pos++;
            }
        }

        return {
            data: new Uint8Array(result),
            originalSize: data.length,
            compressedSize: result.length
        };
    }

    function decompress(data) {
        if (!data || data.length === 0) {
            return new Uint8Array(0);
        }

        const result = [];
        let pos = 0;

        while (pos < data.length) {
            if (data[pos] === 255 && pos + 2 < data.length) {
                const count = data[pos + 2];
                for (let i = 0; i < count; i++) {
                    result.push(data[pos + 1]);
                }
                pos += 3;
            } else {
                result.push(data[pos]);
                pos++;
            }
        }

        return new Uint8Array(result);
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

    return { name, compress, decompress, getStats };
})();

if (typeof module !== 'undefined' && module.exports) {
    module.exports = RLECompressor;
}