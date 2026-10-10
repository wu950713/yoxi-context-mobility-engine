Markdown# yoxi 智慧出行預測系統：模組底層運作原理與資料管線規格書
> **2026 和泰 AI 黑客松｜初賽提案：泰懂你｜yoxi 城市漫遊夥伴**  
> *Technical Implementation & Algorithmic Pipeline Specification (PoC Level)*

> 💡 **Note**: 本 Repository 目前為 PoC 概念驗證與架構規格規範階段，展示完整資料清洗管線、空間特徵工程、演算法選型、RAG 批次快取機制與推播規則引擎邏輯。

---

## 系統核心架構總覽（Architecture Overview）

本架構採「三層解耦設計」，兼顧近即時預測效能、資料隱私與極低運算成本：
1. **數據獲取與空間特徵層**：歷史出行脫敏數據清洗、Uber H3 Resolution 8 格網投影、政府開放資料（氣象署/路況/藝文活動）介接。
2. **預測決策與分群引擎**：
   - **K-means 聚類**：5 項連續型生活特徵向量聚合，建立生活型態人設 ($K=3$)。
   - **LightGBM 二元分類**：時序隔離時間切分盲測（前 8 週訓練 / 後 2 週測試），預測未來 15–120 分鐘潛在出行機率。
3. **生成展示與推播觸發層**：
   - **LLM + RAG**：H3 格網 30 分鐘 Redis 批次快取，節省 90% 以上 API 調用成本。
   - **Guardrails AI**：Regex 事實一致性與金額審查器，杜絕 AI 幻覺。
   - **Rule Engine**：每日 2 則上限、22:00 靜音、連續 3 次未開降頻 72 小時、商圈預算池上限。

---

## 一、 歷史出行數據清理與特徵篩選管線

針對三個月去識別化出行訂單資料集，實施以下過濾規則：
- **地理座標限制**：經度限制 `119.8°E ~ 122.2°E`、緯度限制 `21.8°N ~ 25.4°N`，排除 GPS 漂移導致起訖點相同且時間大於 0 之無效樣本。
- **距離與時間極值**：排除行駛距離 $\le 0.3$ 公里（誤觸取消）與 $> 150$ 公里之離群值；設定行駛平均時速介於 $3 \sim 110\text{ km/h}$。
- **Uber H3 空間網格投影**：使用 `h3-py` 將連續 GPS 座標轉換為 Resolution 8（單格跨度約 461 公尺，面積約 0.737 平方公里）六角格網：
```python
import h3

def assign_h3_indices(df, resolution=8):
    df['pickup_h3'] = df.apply(lambda r: h3.geo_to_h3(r['PickUp_Latitude'], r['PickUp_Longitude'], resolution), axis=1)
    df['dropoff_h3'] = df.apply(lambda r: h3.geo_to_h3(r['DropOff_Latitude'], r['DropOff_Longitude'], resolution), axis=1)
    return df
二、 K-means 用戶生活微輪廓分群運算1. 特徵聚合向量對每位用戶聚合萃取 5 項連續型特徵：尖峰通勤指數：週間早晚尖峰乘車佔比夜間活躍指數：22:00–04:00 乘車佔比週末休閒指數：週末乘車佔比商圈漫遊廣度：用戶歷史起點相異 H3 格網數車型敏感度：尊榮/商務車型選用比例2. 標準化與分群求解Pythonfrom sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
X_scaled = scaler.fit_transform(user_feature_matrix)

# 依據輪廓係數收斂為 K=3：[高規律通勤族]、[夜貓社交達人]、[週末休閒探索者]
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
user_clusters = kmeans.fit_predict(X_scaled)
三、 LightGBM 出行時機預測引擎1. 樣本構造與時間切分驗證（Time-Split）樣本標籤：以 1 小時滑動窗口為基準，起算 2 小時內有訂單標記為 $y=1$，其餘為 $y=0$。類別不平衡：設定 scale_pos_weight = 8.5 平衡稀疏叫車樣本。時序隔離驗證：前 8 週資料為訓練集（Train Set），後 2 週時序隔離數據為測試集（Test Set），嚴禁使用隨機 K-Fold 交叉驗證以避免 Data Leakage。2. 核心參數與判定門檻Pythonparams = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'scale_pos_weight': 8.5,
    'verbose': -1
}
# 預測閾值：機率 >= 0.65 觸發情境卡片；機率 < 0.40 降級為通用生活情報
四、 RAG 批次快取與事實防護網管線1. H3 空間網格快取（Redis Cache）Key 規範：briefing:{h3_id}:{time_block_30min}機制：同格網同半小時內僅呼叫 1 次 LLM 生成生活情報並存入 Redis（TTL 1,800 秒），該區所有開啟 App 的用戶共用快取，節省 90% 以上 API 成本，延遲 < 15ms。2. Regex 事實一致性防護網（Guardrails）Pythonimport re

def validate_llm_briefing(text, ground_truth_rain_prob, allowed_discount=80):
    # 1. 檢驗降雨機率是否捏造
    rain_matches = re.findall(r'(\d{1,3})%', text)
    if rain_matches and abs(int(rain_matches[0]) - ground_truth_rain_prob) > 5:
        return False, "Rain probability mismatch"
            
    # 2. 檢驗折價券金額是否幻覺生成
    discount_matches = re.findall(r'(\d+)\s*元', text)
    if discount_matches:
        for val in discount_matches:
            if int(val) > allowed_discount:
                return False, "Unauthorized discount injection"
                
    return True, "Valid"
五、 規則引擎、狀態機與防打擾機制每日上限：每位用戶每日最多 2 則推播。夜間靜音：22:00 至翌日 07:00 關閉主動推播。防疲勞降頻：連續 3 則推播未開啟，進入 COOLDOWN_TIRED 狀態（冷卻 72 小時停止自動推播）。商圈預算池：單日補貼達上限 95% 時，自動切換為發放和泰 Points 微任務；同帳號每 7 天限領 1 次大額情境券。專案目錄結構規劃（Project Layout）Plaintextyoxi-context-mobility-engine/
│
├── README.md                          # 專案總覽與規格說明書
├── requirements.txt                   # 核心相依套件清單
│
├── src/                               # 核心演算法與管線模組
│   ├── data_pipeline/                 # H3 空間格網投影與異常清洗
│   ├── models/                        # K-means 分群與 LightGBM 預測
│   ├── rag_engine/                    # Redis 批次快取與 Regex 防護網
│   └── rules/                         # 推播狀態機與預算冷卻規則
│
└── notebooks/
    └── poc_pipeline_walkthrough.ipynb # 演算法與時間切分驗證展示筆記本
