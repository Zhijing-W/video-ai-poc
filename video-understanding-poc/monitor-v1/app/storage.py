"""Blob 存储封装（可选）。

设计原则：**Storage 没配置时自动降级**，不上传、不报错，整条管线
仍可在本地纯文件模式下跑通。等真正建了 Storage Account 再启用。

启用方式（二选一，填进 .env）：
  1. AZURE_STORAGE_CONNECTION_STRING=...（最简单，PoC 推荐）
  2. AZURE_STORAGE_ACCOUNT_NAME=...  + 当前身份有 RBAC（用 DefaultAzureCredential）
"""
from __future__ import annotations

from pathlib import Path

from .core.config import settings

# 延迟导入 azure-storage-blob，未配置时连导入都省了
_service = None
_init_done = False


def is_enabled() -> bool:
    """是否配置了 Blob 存储。"""
    return bool(settings.storage_connection_string or settings.storage_account_name)


def _get_service():
    """惰性创建 BlobServiceClient，失败则返回 None（降级）。"""
    global _service, _init_done
    if _init_done:
        return _service
    _init_done = True

    if not is_enabled():
        _service = None
        return None

    try:
        from azure.storage.blob import BlobServiceClient

        if settings.storage_connection_string:
            _service = BlobServiceClient.from_connection_string(
                settings.storage_connection_string
            )
        else:
            from azure.identity import DefaultAzureCredential

            account_url = (
                f"https://{settings.storage_account_name}.blob.core.windows.net"
            )
            _service = BlobServiceClient(
                account_url, credential=DefaultAzureCredential()
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[storage] 初始化失败，降级为纯本地模式：{exc}")
        _service = None
    return _service


def _ensure_container(service) -> None:
    try:
        service.create_container(settings.storage_container_name)
    except Exception:  # noqa: BLE001
        # 已存在或无权限创建（容器可能已手动建好），忽略
        pass


def upload_file(local_path: str | Path, blob_name: str) -> str | None:
    """上传单个文件到容器，返回 blob URL；未配置/失败则返回 None。"""
    service = _get_service()
    if service is None:
        return None
    try:
        _ensure_container(service)
        client = service.get_blob_client(
            container=settings.storage_container_name, blob=blob_name
        )
        with open(local_path, "rb") as fh:
            client.upload_blob(fh, overwrite=True)
        return client.url
    except Exception as exc:  # noqa: BLE001
        print(f"[storage] 上传 {blob_name} 失败：{exc}")
        return None


def upload_text(text: str, blob_name: str, content_type: str = "application/json") -> str | None:
    """上传一段文本（如 result.json）到容器，返回 URL 或 None。"""
    service = _get_service()
    if service is None:
        return None
    try:
        from azure.storage.blob import ContentSettings

        _ensure_container(service)
        client = service.get_blob_client(
            container=settings.storage_container_name, blob=blob_name
        )
        client.upload_blob(
            text.encode("utf-8"),
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
        return client.url
    except Exception as exc:  # noqa: BLE001
        print(f"[storage] 上传文本 {blob_name} 失败：{exc}")
        return None
