# 部署成网址：最短步骤

这个项目已经按 Streamlit Community Cloud 的目录规则准备好。

## 你要准备的

- 一个 GitHub 账号
- 一个 Streamlit Community Cloud 账号（可直接用 GitHub 登录）

不需要：

- Python 环境
- Excel
- LSEG
- Alpha Vantage
- 付费行情 API

## 部署

1. 在 GitHub 新建一个仓库，例如 `ipo-stock-predictor`。
2. 把这个项目文件夹里的所有文件上传到仓库根目录。
3. 打开 `https://share.streamlit.io/`，用 GitHub 登录。
4. 点 **Create app**。
5. Repository 选择刚才的仓库。
6. Branch 选 `main`。
7. Main file path 填：`app.py`。
8. Advanced settings 里 Python 选 **3.12**。
9. 点 Deploy。

完成后会得到类似：

`https://你的名字-ipo-stock-predictor.streamlit.app/`

以后你就只打开这个网址。

## 第一次打开网页

左侧会显示“历史训练库尚未建立”。

点：

**建立/更新公开历史库**

程序会自动：

JPX IPO 上市档案 → Yahoo Finance 上市后日线 → 数据质量检查 → 生成训练样本。

之后再输入/上传 627A 的最新状态并点预测。

## 为什么没有直接把几百只股票历史CSV塞进压缩包

两个原因：

1. 避免把第三方行情数据重新打包分发；
2. 让你部署时按公开数据源自动生成，数据来源和更新时间可追踪。

627A 本身截至 2026-10-02 的 8 根日线因为是当前开发对象，项目中放了一份交叉核验的小型种子文件用于校验。

## 如果 Yahoo Finance 在云端临时限流

网页不会拿缺失数据硬训练；失败股票会被写入质量报告并排除。

如果可用股票少到不足以训练，网页会明确报“样本不足”，不会给假预测。后续可以很容易加第二个历史行情源作为备用。
