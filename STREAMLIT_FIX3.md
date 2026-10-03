# Streamlit 修复 v3

- 避免 `src.xxx` 包导入，直接把 `src/` 加入 Python 路径。
- `modeling.py` 内部同步改为 `from features import FEATURES`。
- 添加 `.python-version = 3.11`。
- requirements 保持无 lxml / lightgbm 强制依赖。

建议覆盖：`app.py`、`src/modeling.py`、`requirements.txt`，并新增 `.python-version`。
