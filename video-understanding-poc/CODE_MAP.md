# 仓库代码地图

## 默认开发范围

所有新功能默认只修改：

```text
event-monitor/
```

开始前阅读 `event-monitor/CODE_MAP.md`。

## 冻结旧版

```text
monitor-v1/
```

只有任务明确要求修改旧实时监控功能时，才读取这个目录。

## 非源码目录

常规代码搜索不要扫描：

- `.venv/`
- `data/`
- `out/`
- `gfpgan/`
- `*.pt`、`*.onnx`、`*.pth`
