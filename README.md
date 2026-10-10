# yoxi 智慧出行預測系統：模組底層運作原理與資料管線規格書
> **2026 和泰 AI 黑客松｜初賽提案：泰懂你｜yoxi 城市漫遊夥伴**  
> *Technical Implementation & Algorithmic Pipeline Specification (PoC Level)*

>  **Note**: 本 Repository 目前為 PoC 概念驗證與架構規格規範階段，展示完整資料清洗管線、空間特徵工程、演算法選型、RAG 批次快取機制與推播規則引擎邏輯。

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
