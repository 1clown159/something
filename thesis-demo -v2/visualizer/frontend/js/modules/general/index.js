/**
 * Compression Module - Frontend Compression Algorithms
 * Exported for use in browser
 */
const CompressionModule = (() => {
    const engines = {
        lz77: LZ77Compressor,
        rle: RLECompressor,
        deflate: DeflateCompressor,
        neural: NeuralCompressor
    };

    const algorithms = Object.keys(engines);

    function getEngine(name) {
        return engines[name] || engines.lz77;
    }

    function listAlgorithms() {
        return [...algorithms];
    }

    async function compress(name, data) {
        const engine = getEngine(name);
        if (engine.isAsync) {
            return engine.compress(data);
        } else {
            return engine.compress(data);
        }
    }

    async function decompress(name, data) {
        const engine = getEngine(name);
        if (engine.isAsync) {
            return engine.decompress(data);
        } else {
            return engine.decompress(data);
        }
    }

    function getStats(name, originalSize, compressedSize, elapsedTime) {
        const engine = getEngine(name);
        return engine.getStats(originalSize, compressedSize, elapsedTime);
    }

    async function compareAll(text) {
        const data = new TextEncoder().encode(text);
        const results = {};

        for (const algo of algorithms) {
            try {
                const result = await compress(algo, data);
                const decompressed = await decompress(algo, result.data);
                const verified = Array.from(decompressed).every((v, i) => v === data[i]);

                results[algo] = {
                    success: true,
                    algorithm_name: getEngine(algo).name,
                    original_size: data.length,
                    compressed_size: result.data.length,
                    compression_ratio: result.data.length > 0 ? data.length / result.data.length : 0,
                    space_saving: result.data.length > 0 ? (1 - result.data.length / data.length) * 100 : 0,
                    compression_time_ms: result.elapsed,
                    verification_passed: verified
                };
            } catch (e) {
                results[algo] = { success: false, error: e.message };
            }
        }

        return results;
    }

    return {
        engines,
        algorithms,
        getEngine,
        listAlgorithms,
        compress,
        decompress,
        getStats,
        compareAll
    };
})();

if (typeof module !== 'undefined' && module.exports) {
    module.exports = CompressionModule;
}