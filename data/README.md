# 数据目录

此目录只存放本地运行数据，不提交大型数据集或客户视频。

建议布局：

```text
data/
├── samples/       # 可公开、许可允许的小型演示视频
├── external/      # 手动下载的数据集，例如 MEVID
└── generated/     # 预处理缓存
```

Event Monitor 也支持直接在页面上传视频，因此仓库不强制附带样片。

数据集应在实验文档中记录名称、版本、许可和下载方式，不要将原始数据复制进 Git。

样片可配一个同名 gallery sidecar，例如：

```text
samples/
├── chokepoint_real.mp4
├── chokepoint_real.gallery.json
└── chokepoint_real_gallery/
    └── alice/
        ├── body_001.jpg
        └── face_001.jpg
```

`chokepoint_real.gallery.json` 使用版本 1：

```json
{
  "version": 1,
  "dataset": {
    "name": "ChokePoint",
    "sequence": "P1E_S1",
    "analysis_camera": "C1",
    "gallery_camera": "C2"
  },
  "subjects": [
    {
      "label": "Alice",
      "source_id": "0003",
      "body_images": ["chokepoint_real_gallery/alice/body_001.jpg"],
      "face_images": ["chokepoint_real_gallery/alice/face_001.jpg"]
    }
  ]
}
```

视频必须来自一个机位的一段连续原始帧；不要把 gallery 图片或多个片段拼成视频。
参考图应来自同一数据集的另一机位，并且只把库内身份写进 sidecar。引用路径必须是
sidecar 所在目录下的相对路径。
