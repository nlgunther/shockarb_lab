# 📊 Market Close Report

**2026-06-03 16:56**

> ⚠️ **Stale data** — snapshot is more than 6 hours old. Run `python utils/market_data.py` for a fresh fetch.

---

## 🇺🇸 Broad Market

| Index              | Close  | Change   |
| ------------------ | ------:| --------:|
| S&P 500 (SPY)      | 754.24 | ▼ -0.70% |
| Nasdaq 100 (QQQ)   | 744.21 | ▼ -0.26% |
| Russell 2000 (IWM) | 287.67 | ▼ -1.37% |
| Dow Jones (DIA)    | 508.26 | ▼ -1.13% |

---

## 🏭 Sectors *(best → worst)*

| Sector                     | Close      | Change       |
| -------------------------- | ----------:| ------------:|
| **Energy (XLE)**           | **58.71**  | **▲ +1.29%** |
| **Health Care (XLV)**      | **147.55** | **▲ +0.79%** |
| **Consumer Staples (XLP)** | **82.16**  | **▲ +0.40%** |
| Materials (XLB)            | 51.63      | ▲ +0.21%     |
| Real Estate (XLRE)         | 43.51      | ▲ +0.05%     |
| Industrials (XLI)          | 174.05     | ▼ -0.08%     |
| Utilities (XLU)            | 43.71      | ▼ -0.43%     |
| Consumer Disc. (XLY)       | 116.73     | ▼ -0.73%     |
| *Tech (XLK)*               | *196.23*   | *▼ -1.00%*   |
| *Financials (XLF)*         | *50.87*    | *▼ -1.15%*   |
| *Comm. Services (XLC)*     | *112.08*   | *▼ -1.31%*   |

---

## 💵 Bonds & Rates

| Instrument             | Close  | Change   |
| ---------------------- | ------:| --------:|
| 20yr Treasury (TLT)    | 85.31  | ▼ -0.40% |
| 7-10yr Treasury (IEF)  | 94.00  | ▼ -0.25% |
| High Yield (HYG)       | 79.68  | ▼ -0.28% |
| Inv. Grade Corp. (LQD) | 108.62 | ▼ -0.28% |

---

## 🌍 Overseas Markets

*Snapshot fetched at 2026-06-03 16:56 ET.*

### Europe

| Market            | Close     | Change   | Status            |
| ----------------- | ---------:| --------:| ----------------- |
| FTSE 100 (London) | 10,373.50 | ▲ +0.33% | (closed at fetch) |
| DAX (Frankfurt)   | 25,124.17 | ▲ +0.48% | (closed at fetch) |
| CAC 40 (Paris)    | 8,209.09  | ▲ +0.77% | (closed at fetch) |
| Euro Stoxx 50     | 6,107.85  | ▲ +1.21% | (closed at fetch) |

### Asia-Pacific

| Market              | Close     | Change   | Status            |
| ------------------- | ---------:| --------:| ----------------- |
| Nikkei 225 (Tokyo)  | 66,734.24 | ▼ -0.30% | (closed at fetch) |
| Hang Seng (HK)      | 26,038.32 | ▲ +2.52% | (closed at fetch) |
| Shanghai Composite  | nan       | ▼ +nan%  | (closed at fetch) |
| BSE Sensex (Mumbai) | 74,649.84 | ▲ +0.52% | (closed at fetch) |
| ASX 200 (Sydney)    | 8,724.40  | ▼ -0.06% | (closed at fetch) |

### Americas (ex-US)

| Market              | Close      | Change   | Status                 |
| ------------------- | ----------:| --------:| ---------------------- |
| Bovespa (São Paulo) | 170,330.62 | ▼ -2.22% | (likely open at fetch) |

---

## 📉 Risk Gauges

| Gauge      | Close  | Change   |
| ---------- | ------:| --------:|
| VIX        | 16.06  | ▲ +1.84% |
| Gold (GLD) | 407.87 | ▼ -0.99% |
| Oil (USO)  | 140.86 | ▲ +2.62% |

---

## 🎯 ShockArb Fit Analysis

### Condition Checks

| Condition              | Status   | Notes                                           |
| ---------------------- | -------- | ----------------------------------------------- |
| **Breadth**            | MIXED    | balanced (-0.09)                                |
| **Volatility (VIX)**   | MODERATE | VIX 16.1, +1.8% today                           |
| **Sector dispersion**  | MODERATE | 2.6 pp spread                                   |
| **Market trend**       | CHOPPY   |                                                 |
| **Tech concentration** | MODERATE | tech vs SPY: -0.3 pp                            |
| **Bond signal**        | RISK-ON  | TLT -0.40% — bonds selling, risk-on environment |

### Overall Fit: 🟡 CAUTION

### Analysis

The market is choppy with no clear directional bias. VIX is moderate (VIX 16.1, +1.8% today). Breadth is mixed with moderate sector dispersion (2.6 pp spread). Bonds are selling off with equities, suggesting the move is driven by rate/inflation concerns rather than pure panic.

### Recommendation

> Mixed conditions — run the scanner but apply elevated thresholds (r² > 0.55, confidence_delta > 0.003) and avoid clusters of correlated names.

---

*Snapshot: 2026-06-03 16:56 | Source: rules*