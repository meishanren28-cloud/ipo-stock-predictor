# 数据查验与设计记录（2026-10-03）

## 1. 627A akippa 上市元数据

JPX 新规上场页面显示：

- 上市日：2026-09-18
- 代码：627A
- 市场：スタンダード
- 仮条件：540～570円
- 公募・売出価格：570円
- 公募：378千股
- 売出：1,833.1千股（OA 331.6千股）

来源：

- https://www.jpx.co.jp/listing/stocks/new/
- https://www.jpx.co.jp/listing/stocks/new/t13vrt000001udd2-att/09akippa-Outline.pdf

JPX 公司概要还确认：上场时发行股数 6,262,140 股，业务为停车位 marketplace「アキッパ」。

## 2. 627A 日线交叉核验

截至 2026-10-02，株探与みんかぶ的 8 根日线逐行一致：

| 日期 | 开 | 高 | 低 | 收 | 成交量 |
|---|---:|---:|---:|---:|---:|
| 2026-09-18 | 1244 | 1315 | 1019 | 1037 | 12,872,700 |
| 2026-09-24 | 995 | 1337 | 989 | 1337 | 10,680,200 |
| 2026-09-25 | 1367 | 1637 | 1350 | 1637 | 27,485,000 |
| 2026-09-28 | 1837 | 2037 | 1800 | 2037 | 8,642,700 |
| 2026-09-29 | 2238 | 2507 | 2024 | 2105 | 40,989,800 |
| 2026-09-30 | 2250 | 2379 | 2016 | 2041 | 35,229,300 |
| 2026-10-01 | 1911 | 1953 | 1682 | 1745 | 10,505,600 |
| 2026-10-02 | 1750 | 1967 | 1750 | 1778 | 11,148,400 |

2026-10-02 又与 Yahoo Finance Japan 核验一致。

来源：

- https://s.kabutan.jp/stocks/627A/historical_prices/daily/
- https://minkabu.jp/stock/627A/daily_bar
- https://finance.yahoo.co.jp/quote/627A.T

这些数据已写入 `data/verified_627A.csv`。

## 3. 为什么历史母表用 JPX

JPX 的新规上场归档保留历年 IPO 的上市日期、代码、市场、发行/卖出价格等，是比新闻站或IPO博客更适合作为“谁属于IPO历史样本”的权威母表。

已确认归档页至少覆盖：

- 2025: https://www.jpx.co.jp/listing/stocks/new/00-archives-01.html
- 2024: https://www.jpx.co.jp/listing/stocks/new/00-archives-02.html
- 2023: https://www.jpx.co.jp/listing/stocks/new/00-archives-03.html
- 2022: https://www.jpx.co.jp/listing/stocks/new/00-archives-04.html

程序会从当前页自动发现归档链接，并另有编号页 fallback，而不是把年份写死。

## 4. 为什么第一版没有要求 J-Quants 付费

JPX 官方 2026 年资料确认：J-Quants V2 使用 API Key；分钟/Tick 数据是 Light 以上的 5,500円/月附加项。免费方案可用于部分历史日线，但有数据期间和延迟限制。

来源：

- https://www.jpx.co.jp/english/corporate/news/news-releases/6020/20260119.html
- https://elb.test-dlv.jpx-jquants.com/

因此第一版采用“JPX 元数据 + 公开 Yahoo Finance 日线 + 用户当前截图”，避免为历史相似案例先付费。

## 5. 为什么不输出“先1950、再1750”的日线概率

一根日线只有 O/H/L/C，无法知道最高价和最低价的发生顺序。任何只用日线却声称能统计“先高后低/先低后高”的模型都会凭空补信息。

所以网页只给：

- 未来窗口内上触某价概率
- 未来窗口内下触某价概率
- 高/低分位区间

盘中路径要靠分钟数据或连续截图更新。

## 6. 为什么用相似案例 + Quantile Model

对刚上市的新股，单股自身历史很短。把它作为横截面状态（上市第N天、从峰值回撤、量能衰减、最近收益等）去匹配过去其他IPO，比硬训练单股 LSTM 更合理。

模型分两路：

1. KNN / Nearest Neighbors：直接展示最像的历史状态和后续实际分布；
2. LightGBM Quantile Regression：学习跨案例的非线性关系并输出 P10/P50/P90。

两者结果可以互相检查；如果机器学习预测和相似案例经验分布相差很大，用户能直接看到，而不是只得到一个黑箱数字。

## 7. 部署方式已核实

Streamlit Community Cloud 官方支持：

- 从 GitHub 仓库部署
- `requirements.txt` 安装 Python 包
- `packages.txt` 安装 Debian apt 系统依赖
- 免费个人/教育/非商业部署入口

来源：

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

项目因此包含 `requirements.txt` + `packages.txt`，OCR 的 Tesseract 也能随部署安装。
