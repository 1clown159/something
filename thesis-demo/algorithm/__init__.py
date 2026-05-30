# Stage4 Algorithm Module
# 核心压缩算法：基于深度学习的无损浮点数据压缩

from .common import (
    ExperimentConfig, VolumeData, VolumeShape, SplitConfig,
    extract_float_exponents, extract_float_components,
    infer_shape_from_file
)
from .stage4 import (
    Small2DCNN,
    build_single_stage4_feature_causal_edge,
    build_stage4_features_causal_edge,
    load_stage4_model,
    resolve_feature_mode,
    resolve_target_mode,
    feature_mode_to_in_channels,
    predictor_for_coord,
    target_symbol_for_coord
)
from .codec import (
    Stage4GlobalDiagonalRangeCodec,
    Stage4RangeCodec,
    Stage4TileRangeCodec,
    write_bitstream,
    read_bitstream,
    file_sha256
)

__all__ = [
    "ExperimentConfig", "VolumeData", "VolumeShape", "SplitConfig",
    "extract_float_exponents", "extract_float_components", "infer_shape_from_file",
    "Small2DCNN",
    "build_single_stage4_feature_causal_edge", "build_stage4_features_causal_edge",
    "load_stage4_model",
    "resolve_feature_mode", "resolve_target_mode", "feature_mode_to_in_channels",
    "predictor_for_coord", "target_symbol_for_coord",
    "Stage4GlobalDiagonalRangeCodec", "Stage4RangeCodec", "Stage4TileRangeCodec",
    "write_bitstream", "read_bitstream", "file_sha256"
]
