# 日本IPO相似案例 + 概率预测网页

这是一个面向**日本新股 / 小盘高波动股**的研究工具。它不是“用一句话猜明天最高点”，而是把当前股票状态与历史IPO状态做相似度匹配，再用分位数机器学习估计未来 1 / 3 / 5 个交易日的最高/最低价格分布，并计算你指定价位被触及的历史加权概率。

## 你平时怎么用

部署完成后，日常只需要打开网页：

1. 输入股票代码（默认 `627A`）。
2. 可上传券商/行情截图；OCR 只负责预填数字，网页会要求你确认开盘/高/低/当前价/成交量。
3. 输入持仓、成本、想测试的卖价和回补价。
4. 点 **开始预测**。
5. 看：
   - 未来 1/3/5 日最高价 P10/P50/P90；
   - 最低价 P10/P50/P90；
   - 1950、1750、1700 等任意价位的触及概率；
   - 历史上最像当前状态的 IPO 案例；
   - 模型动作与“卖飞风险代理”；
   - 时间顺序留出集的误差指标。

> 重要：仅凭日线不能知道同一天是“先到1950再到1750”，还是反过来。网页不会伪造这种顺序概率。要判断盘中顺序，需要分钟线或你继续上传盘中截图。

## 数据是怎么来的

### IPO 母表：JPX 官方

程序从日本交易所集团（JPX）“新規上場会社情報 / 新規上場銘柄一覧”及其归档页读取：

- 上市日期
- 公司名
- 证券代码
- 市场区分
- 公募/卖出价格（有数据时）
- 公募、卖出数量（有数据时）

入口：`https://www.jpx.co.jp/listing/stocks/new/`

### 上市后价格：Yahoo Finance（日线）

为了让个人版无需付费 API，历史 OHLCV 使用 `yfinance` 读取 Yahoo Finance 的日线；每只股票在进入训练库前必须通过：

- OHLC 包络关系（Low <= Open/Close <= High）
- 正价格
- 非负成交量
- 首个交易日与 JPX 上市日差距合理
- 最少历史天数

失败的股票会写入 `quality_report.csv` 并排除，不会强行塞给模型。

### 627A akippa 特别核验

`data/verified_627A.csv` 内置截至 **2026-10-02** 的 8 根日线，已用株探和みんかぶ逐行交叉检查；10/2 还与 Yahoo Finance Japan 核对。上市日期、市场、570 日元公募价来自 JPX。

## 模型不是怎么做的

它**不是**拿 627A 自己 8 根K线去训练一个 LSTM。那样样本太少，基本等于过拟合。

## 模型怎么做的

### 1. 横向历史案例库

对过去多年的 IPO，每个上市后的第 4～35 个交易日都生成一个“当时状态”。状态只使用当日及之前的数据，绝不偷看未来。

主要特征包括：

- 上市第几天
- 当前价 / 首日开盘、首日收盘、公募价
- 上市以来最大涨幅
- 从上市后最高点回撤多少
- 距离峰值几天
- 1 / 3 / 5 日收益率
- 跳空、日内振幅、上下影线
- 当前价在当天振幅中的位置
- 成交量 / 峰值量、3日均量、5日均量
- 3 / 5 日波动率
- 最近3日涨跌天数
- 距离最大成交量日多久

### 2. 相似案例（Nearest Neighbors）

标准化上述特征后，找距离当前状态最近的 15～80 个历史状态。越相似的案例权重越高。

自定义价位的“触及概率”主要来自这套**相似案例加权经验分布**，因此你可以输入 1950、1700、2300 等任意价格，不需要为每个价位重新训练模型。

### 3. 分位数模型

用 LightGBM Quantile Regression 分别学习：

- 未来 N 日最高收益率分布
- 未来 N 日最低收益率分布
- 未来 N 日末收盘收益率中位数

输出 P10 / P50 / P90，而不是假装能准确到“最高1963、最低1712”。

如果部署环境无法加载 LightGBM，会自动退回 sklearn 的 Quantile Gradient Boosting。

### 4. 时间顺序回测

验证集按时间切，不随机打乱，避免把未来 IPO 市场环境泄漏给过去。网页显示未来最高、最低、收盘收益率中位预测的 MAE。

## 第一次部署后的历史库构建

源码包**不直接附带批量 Yahoo 历史数据**，避免把第三方行情数据当成我们自己的数据重新分发。部署到有互联网的环境后：

1. 打开网页；
2. 左侧点 **建立/更新公开历史库**；
3. 程序自动抓 JPX 母表、下载上市后日线、质量检查并生成训练样本；
4. 之后即可预测。

生成文件：

- `data/ipo_master.csv`
- `data/ipo_daily.parquet`
- `data/quality_report.csv`
- `data/training_samples.parquet`
- `data/manifest.json`

也可以命令行运行：

```bash
python scripts/build_history.py --start-year 2018
```

## 部署成“只打开网址就用”

推荐 Streamlit Community Cloud：

1. 把整个项目放到 GitHub 仓库。
2. 登录 `https://share.streamlit.io/`。
3. Create app → 选仓库 → 入口文件填 `app.py`。
4. Python 建议 3.12。
5. Deploy。
6. 第一次打开，点左侧“建立/更新公开历史库”。

`requirements.txt` 和 `packages.txt` 已经写好；后者会安装 Tesseract 日语 OCR。

## 盘中截图的限制

模型的历史训练基础是**完整收盘日线**。如果你上午 10:00 上传截图，当天成交量、最高/最低都还没走完，所以网页会把它标成“盘中模式”，并提醒把置信度打折。

如果后续要把盘中预测做得更严谨，下一阶段应加入大量历史 5 分钟 / 1 分钟数据，而不是把未完成日K当成完整日K。

## 文件结构

```text
ipo_stock_predictor/
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
├── .streamlit/config.toml
├── data/
│   ├── verified_627A.csv
│   ├── akippa_metadata.json
│   └── README.md
├── scripts/
│   ├── build_history.py
│   └── verify_627a.py
└── src/
    ├── data_sources.py
    ├── features.py
    ├── modeling.py
    └── ocr_utils.py
```

## 这套工具能回答什么 / 不能回答什么

可以：

- “像现在这种 IPO 冲顶后回撤+缩量，过去类似案例未来5天通常怎么走？”
- “未来3日触及1950的历史加权概率多大？”
- “2300属于中位路径还是尾部路径？”
- “如果我要做100股T，1950这个卖价是不是过于贪？”

不能保证：

- 精确预测明日最高/最低；
- 用日线恢复盘中先后顺序；
- 提前知道突发公告、监管、停牌、市场黑天鹅；
- 保证盈利。

模型是统计决策辅助，不会自动向券商下单。

## Strict out-of-sample backtest (added)
Open **模型体检：严格样本外回测 / 分位数校准** in the Streamlit app. The test uses year-based walk-forward folds, keeps each test IPO entirely out of the training set, and forbids training dates from reaching into the test year. It reports P10/P50/P90 empirical coverage and calibration of nearest-neighbor touch probabilities.
