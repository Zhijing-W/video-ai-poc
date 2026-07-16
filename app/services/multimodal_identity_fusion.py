"""兼容旧导入路径：实际实现已迁到 app.identity.identity_confidence。"""
from ..identity.identity_confidence import fuse_multimodal_identity, score_identity_confidence

__all__ = ["fuse_multimodal_identity", "score_identity_confidence"]
