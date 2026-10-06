# 測試流程設計與檢查結果

本次重構從 `dev` 建立 `refactor/test-code-optimize`。範圍是測試與量測架構；既有演算法的計分、kernel 與 traceback 保持原樣。

## 原流程的問題與處理

| 問題 | 改進 |
| --- | --- |
| 正確性、計時、最大 5000 筆 batch 放在同一個 Catch case | CTest 正確性測試與 `sw-benchmark` 分開，預設不跑效能測試 |
| 所有測試都依賴 CUDA 工具鏈與 GPU | `SW_ENABLE_CUDA=OFF` 可獨立建置 CPU；GPU 測試獨立程序，沒有裝置時回傳 77，CTest 顯示 skipped |
| CTest 僅在子目錄啟用 | 根目錄 `include(CTest)`；從 build 根目錄可發現所有測試 |
| CMake glob 收入所有 cpp，強制 Release/O3/NDEBUG/native | 明列 target/source；使用者控制 build type；native 指令集需明確開啟 |
| scalar / SIMD 語意不同，分數差 20 也算過 | 分別對 overlap / local 獨立參考實作做精確分數比對 |
| CUDA batch 只檢查輸出數量；CPU batch 只抽查第一筆 | 檢查每筆順序、分數、offset、CIGAR；CPU 跨執行緒數與重複呼叫；CUDA single/batch 分開驗證 |
| batch=1 偷換為 single 呼叫 | `--mode single` 與 `--mode batch` 固定選擇，不因 batch 大小改變 |
| 每次只有一次計時，重複同一筆資料 | 固定 seed、不同序列、warmup、多次樣本、median/min/p95、吞吐量、checksum |
| `test_accuracy.sh` 跑 100 次且覆寫 tracked FASTA | 改呼叫 CTest repeat；資料在記憶體生成；FASTA 匯出要求新目錄與 seed |
| Docker 假設 build 已存在，掛載路徑沒引號 | 建立容器內獨立 build 目錄，正確處理路徑，透過 CTest 執行 |

## 共用介面契約

`support/backend.hpp` 是測試／benchmark 的 adapter 邊界，不是取代既有 library 的正式 API。

- `Pair{ref, query}` 讓每筆輸入成對，消除兩個 vector 數量不同卻被 `min()` 靜默截斷的問題。
- `Scoring` 明定符號，gap 長度 k 的代價為 `open + (k - 1) * extend`。
- `Descriptor` 宣告對齊模型與分數位元數。scalar/CUDA 是零邊界、最後一列／行挑終點的 overlap；SIMD 是取全域最高 cell 的 local。不能假設三者相同。
- `prepare` 擁有輸入，驗證 A/C/G/T、非空序列、分數範圍、整數索引、執行緒及 cell 上限。空 batch 合法；單筆空字串一致拒絕。這些限制屬於測試 adapter，不代表原始 library 已修復輸入驗證。
- `Options` 明確選擇 single/batch、CPU threads（1..256）與 max_cells。single 和 CUDA 要求 threads=1；CPU batch 實際 worker 數為 min(threads, batch)。
- `PreparedWork::run()` 回傳與輸入等長且順序一致的 host results，返回前所有 GPU 工作必須完成。同一 work 不可並行呼叫，重複循序呼叫必須可靠。
- CPU batch 使用共用 executor 呼叫單筆函式，thread 建立與 join 都計入延遲；它不量測舊 SIMD `batch_align()` 的自動執行緒策略。CUDA batch 直接呼叫真正 batch API。
- 16-bit SIMD 採保守算術範圍驗證；超過限制明確拒絕，避免溢位污染效能數據。預設總 cell 預算 1600 萬，為工作量上限而非精確記憶體用量或 OOM 保證。

新增 CUDA 實作時可實作另一個 `PreparedWork`，沿用同一組 corpus、結果與計時契約。未來 device buffer/profile 可以在 prepare 配置並在 run 重用；比較時必須標示 preparation 不計時以及冷啟動／重用模式。非同步 API 應另設 submit/wait 介面；不得讓 `run()` 提早返回來美化量測。

目前 Result 只有 reference offset、CIGAR、score。SIMD local 缺少 query 起點／終點，無法完整重播 local CIGAR；日後正式 API 應回傳 reference/query 的半開區間，並明訂同分解的 tie-breaking。現有 CUDA 回歸測試以 scalar 的 offset/CIGAR 作相容性目標，分數再對獨立 oracle 驗證。

## 正確性與既有缺陷

```sh
ctest --test-dir build --output-on-failure -L correctness
./test/test_accuracy.sh build 5
./build/test/biovoltron-test '[known-defect]'
```

若有 Python 3 且啟用 benchmark，CTest 另註冊 `sw.tools`，檢查 CLI 拒絕條件、CSV 結構、checksum 可重現性和 FASTA 覆寫保護；只使用極小工作量，不做效能門檻。

日常 CPU 測試涵蓋 oracle 手算案例、介面拒絕條件、可重現資料、完全相同序列、異長 scalar DP、CIGAR 重播、SIMD local 精確分數、batch 順序與多執行緒一致性。Oracle 使用獨立的 64-bit、rolling-row affine-gap recurrence，不呼叫既有 backend。

**已確認缺陷：** scalar 的 <=2 mismatch 快速路徑直接回傳 `match * length`，例如 `ACGTACGT` / `ACGAACGT` 回傳 24，正確計分是 20。`[known-defect]` 是明確 opt-in、會失敗的回歸測試，不使用 WILL_FAIL，也不納入預設通過集合。修復後應把它改標為 `[cpu]`。因此預設綠燈不代表整個 library 無正確性問題。

其他待改善項目（程式碼檢查，尚未全面驗證）：

1. CUDA API 失敗目前呼叫 `exit()`，手動配置／釋放 device memory；應改成 RAII 與可回報的錯誤，才能可靠測試失敗路徑。
2. CUDA trace 以 int8 儲存 gap，需補長 gap >127 的語意與溢位回歸；新增 kernel 前應先建立更大的邊界 corpus。
3. scalar/CUDA 的短 mismatch 快速路徑是否永遠符合最佳 overlap alignment，還需要對所有 scoring 預設逐一驗證。
4. SIMD 為 16-bit，且非 ACGT 原先默認映射成 A；adapter 已拒絕不支援輸入，但正式 library 仍需定義策略。
5. SIMD local traceback 的 query 座標不足，分數測試通過不能代表 traceback 完整正確。

GPU corpus 包含 1/7/31/32/33/63/64/65 長度、異長 pair、兩套 scoring、single 與真正 batch（含 batch=1）、每筆輸出和重複呼叫。沒有可用 GPU 時只能確認建置及 skip 行為，不能宣稱 CUDA correctness 已通過。

## 效能量測契約

```sh
./build/test/sw-benchmark --backend scalar --mode batch --batch 16 --length 128 --threads 4 --seed 42 --warmup 2 --repetitions 7 > scalar.csv
./build/test/sw-benchmark --backend simd --mode batch --batch 16 --length 128 --threads 4 --seed 42 --warmup 2 --repetitions 7 > simd.csv
./build_cuda/test/sw-benchmark --backend cuda --mode batch --batch 16 --length 128 --seed 42 --warmup 2 --repetitions 7 > cuda.csv
```

資料生成、host input preparation、checksum 計算不在計時內。每次使用 `steady_clock`，涵蓋 backend 呼叫的記憶體配置、CPU threads、H2D、kernel、D2H、traceback、result 建構與完成同步。回傳 result 的銷毀在計時外。CUDA events 的純 kernel 時間應日後另外報告，不能混用這個欄位。

CSV 包含 backend/model/mode、要求與實際 CPU threads、資料尺寸／identity／seed、warmup／repetitions、scope、min/median/p95、pairs/s、nominal cells/s、checksum。C++ corpus 的 identity 是逐位置機率；Python FASTA 工具則產生指定整數比例的 mismatch，兩種 generator 不保證相同 seed 得到相同序列。nominal cells/s 是 N*M 工作量估計；快速路徑沒有執行完整 DP，不能當實際 kernel GCUPS。

編譯器、host hardware threads、assertions 與 CUDA build 狀態輸出到 stderr。正式結果需一併保存 Git SHA、CMakeCache、CPU/GPU 型號、driver/toolkit、原始命令及原始 CSV；同一次 run 的 p95 只是樣本分位數，不是統計信賴區間。預設沒有以 wall-time 作 CI 門檻，避免共享機器噪音造成假失敗。

優化順序建議：先統一語意與修正分數／traceback，再重用 device allocation 和 packed input，最後評估 kernel/warp 配置、stream pipeline、pinned memory 與 CUDA-event 分階段量測。每一步先通過相同 corpus，再比較相同契約的效能數據。
