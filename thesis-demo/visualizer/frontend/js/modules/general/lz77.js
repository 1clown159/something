/**
 * LZ77 Compression - Frontend Implementation
 * Simplified LZ77 with adaptive prediction
 */
const LZ77Compressor = (() => {
    const name = 'LZ77+Huffman';
    const WINDOW_SIZE = 32768;
    const MIN_MATCH = 3;
    const MAX_MATCH = 258;

    function predict(data) {
        if (!data || data.length === 0) return new Uint8Array(0);
        const result = new Uint8Array(data.length);
        let prev = 0;
        for (let i = 0; i < data.length; i++) {
            if (i < 3) {
                result[i] = data[i];
            } else {
                result[i] = (data[i] - prev) & 255;
            }
            prev = data[i];
        }
        return result;
    }

    function depredict(data) {
        if (!data || data.length === 0) return new Uint8Array(0);
        const result = new Uint8Array(data.length);
        for (let i = 0; i < data.length; i++) {
            if (i < 3) {
                result[i] = data[i];
            } else {
                const prev = result[i - 1];
                result[i] = (prev + data[i]) & 255;
            }
        }
        return result;
    }

    function lz77Encode(data) {
        if (!data || data.length === 0) return new Uint8Array(0);

        const result = [];
        let pos = 0;
        const hashTable = {};

        while (pos < data.length) {
            if (pos + MIN_MATCH <= data.length) {
                const key = Array.from(data.slice(pos, pos + 3)).map(b => b.toString(16).padStart(2, '0')).join('');
                let bestOffset = null;
                let bestLen = MIN_MATCH - 1;

                if (hashTable[key]) {
                    for (const cp of hashTable[key]) {
                        if (cp >= pos || pos - cp > WINDOW_SIZE) continue;
                        let len = 0;
                        while (len < MAX_MATCH &&
                               pos + len < data.length &&
                               data[cp + len] === data[pos + len]) {
                            len++;
                        }
                        if (len > bestLen) {
                            bestLen = len;
                            bestOffset = pos - cp;
                        }
                    }
                }

                if (bestOffset !== null && bestLen >= MIN_MATCH) {
                    result.push(1, (bestOffset >> 8) & 255, bestOffset & 255, Math.min(bestLen, 255));
                    for (let i = 0; i < Math.min(bestLen, 255); i++) {
                        if (pos + i + 3 <= data.length) {
                            const h = Array.from(data.slice(pos + i, pos + i + 3)).map(b => b.toString(16).padStart(2, '0')).join('');
                            if (!hashTable[h]) hashTable[h] = [];
                            hashTable[h].push(pos + i);
                            if (hashTable[h].length > 100) hashTable[h].shift();
                        }
                    }
                    pos += Math.min(bestLen, 255);
                    continue;
                }
            }

            result.push(0, data[pos]);
            if (pos + 3 <= data.length) {
                const key = Array.from(data.slice(pos, pos + 3)).map(b => b.toString(16).padStart(2, '0')).join('');
                if (!hashTable[key]) hashTable[key] = [];
                hashTable[key].push(pos);
                if (hashTable[key].length > 100) hashTable[key].shift();
            }
            pos++;
        }

        return new Uint8Array(result);
    }

    function lz77Decode(data) {
        if (!data || data.length === 0) return new Uint8Array(0);

        const result = [];
        let pos = 0;

        while (pos < data.length) {
            if (data[pos] === 0) {
                if (pos + 1 < data.length) result.push(data[pos + 1]);
                pos += 2;
            } else {
                if (pos + 3 >= data.length) break;
                const offset = (data[pos + 1] << 8) | data[pos + 2];
                const length = data[pos + 3];
                const start = result.length - offset;
                for (let i = 0; i < length; i++) {
                    if (start + i >= 0 && start + i < result.length) {
                        result.push(result[start + i]);
                    }
                }
                pos += 4;
            }
        }

        return new Uint8Array(result);
    }

    async function compress(data) {
        const startTime = performance.now();
        if (!data || data.length === 0) {
            return { data: new Uint8Array(0), originalSize: 0, compressedSize: 0 };
        }

        const predicted = predict(data);
        const lz77Data = lz77Encode(predicted);
        const compressed = await DeflateCompressor.compress(lz77Data);

        const elapsed = performance.now() - startTime;
        return {
            data: compressed.data,
            originalSize: data.length,
            compressedSize: compressed.data.length,
            elapsed: elapsed + (compressed.elapsed || 0)
        };
    }

    async function decompress(data) {
        if (!data || data.length === 0) return new Uint8Array(0);
        try {
            const inflated = await DeflateCompressor.decompress(data);
            const lz77Decoded = lz77Decode(inflated);
            return depredict(lz77Decoded);
        } catch (e) {
            console.error('Decompression error:', e);
            return new Uint8Array(0);
        }
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
    module.exports = LZ77Compressor;
}