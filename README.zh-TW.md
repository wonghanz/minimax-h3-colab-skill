# MiniMax H3 Colab 技能

這是一個完整、可獨立使用的 Codex 技能儲存庫，能將本機參照圖片交給 Google Colab，製作短篇 MiniMax H3 Ref2VA 影片。使用者可以直接 clone、安裝技能、完成 Colab CLI 登入，再使用內附的 runner 或 Shell 啟動器。儲存庫包含：

- `SKILL.md`：Codex 選用此技能時讀取的指示；
- `scripts/runner.py`：管理 session、查詢用量、上傳檔案、批次執行、下載與清理的 runner；
- `assets/MiniMax_H3_Turbo_Colab.ipynb`：在遠端 Colab runtime 執行的推論 Notebook；
- `run_colab_inference.sh`：單支影片的便利啟動器；
- `install.sh`：可攜、可重複執行且預設不覆蓋舊檔的技能安裝程式；
- `tests/`：使用假的 Colab CLI 的離線測試，另含 Notebook 與 CLI 介面檢查。

本機電腦只負責準備與上傳輸入檔案；模型推論會在 Colab GPU 執行，完成的 MP4 再下載回指定位置。

## 需求

- 已安裝 Codex。設定 `CODEX_HOME` 時，技能會安裝到 `$CODEX_HOME/skills`；未設定時會安裝到 `~/.codex/skills`。
- Python 3.11 以上版本，用於執行內附 runner。runner 只使用 Python 標準函式庫。
- [`uv`](https://docs.astral.sh/uv/)，用來安裝 Colab CLI；也可以用其他方式讓 `colab` 出現在 `PATH`。
- [`google-colab-cli`](https://pypi.org/project/google-colab-cli/)。本 runner 已用 Colab CLI 0.7.4 驗證，使用文件所列的 `version`、`usage`、`new`、`status`、`upload`、`exec`、`download`、`stop` 指令；安裝的 CLI 不在驗證過的 0.6–0.7 系列時，runner 會提出警告。目前的 CLI 版本需要 Python 3.12 以上；`uv` 可以另外管理 CLI 使用的 Python，不影響 runner 的 Python 3.11 以上需求。該 CLI 僅支援 Linux 與 macOS，runner 因此也只在這兩個平台可用。
- 可使用 Colab compute units 且能配置 GPU 的 Google 帳號。A100 或其他高記憶體 runtime 可能需要對應的 Colab 方案與足夠餘額。

系統有安裝 `ffprobe` 時，runner 會檢查下載的 MP4：檔案不存在或無法讀取就判定該工作失敗；只有視訊、沒有音訊的 MP4 會保留並提出警告，因為那是已經花額度換來的結果。需要把無聲 MP4 視為失敗時，请加 `--require-audio`。沒有 `ffprobe` 時，仍會檢查檔案存在且大小不為零。

## Clone 與安裝技能

```bash
git clone <repository-url> minimax-h3-colab-skill
cd minimax-h3-colab-skill
./install.sh
```

安裝程式會把必要檔案複製到：

```text
$CODEX_HOME/skills/minimax-h3-colab
```

如果沒有設定 `CODEX_HOME`，目的地是 `~/.codex/skills/minimax-h3-colab`。

需要指定其他技能目錄時（例如測試或使用不同的 Codex 設定），可以明確傳入：

```bash
./install.sh --dest /absolute/path/to/codex/skills
```

預設行為是可重複執行且不會破壞既有安裝。如果目的地已存在，安裝程式會保留原目錄並正常結束。要明確替換既有版本時才使用 `--force`；舊目錄會先移到帶時間戳記的 `.backup.*` 路徑，方便復原：

```bash
./install.sh --force
```

安裝新的技能時，程式會先寫入暫存目錄，再以原子重新命名完成安裝。安裝到 Codex 的內容只有 `SKILL.md`、`scripts/` 與 `assets/`，不會把 Git 資料、測試、README 或輸出檔複製進去。

安裝完成後，請在 Codex 開啟新的回合，或重新整理技能清單（若 Codex 用戶端會快取可用技能）。技能名稱是 `minimax-h3-colab`。

## 安裝並授權 Colab CLI

以使用者工具安裝 CLI：

```bash
uv python install 3.12
uv tool install --python 3.12 google-colab-cli
```

確認 CLI 可以使用：

```bash
colab version
```

runner 預設使用 OAuth2。第一次授權並查詢帳戶餘額：

```bash
colab --auth=oauth2 usage
```

請依照 CLI 印出的網址與驗證碼說明完成流程。Token 會由 CLI 存放在它自己的本機設定中；不要把憑證、Token 或瀏覽器狀態放進本儲存庫、prompt 或工作 manifest。如果環境已經設定 Google Application Default Credentials，可以明確指定另一種認證方式：

```bash
COLAB_AUTH=adc colab --auth=adc usage
```

runner 會把 `--auth="$COLAB_AUTH"` 傳給每一個 Colab 指令；預設值是 `oauth2`。

## 單支影片推論

先建立 UTF-8 文字檔，例如 `prompt.txt`，再執行：

```bash
./run_colab_inference.sh \
  --image /absolute/path/reference_1.png \
  --image /absolute/path/reference_2.jpg \
  --prompt /absolute/path/prompt.txt \
  --output /absolute/path/intro.mp4
```

第一張圖片會在 Notebook 中對應 `<Picture 1>`，第二張對應 `<Picture 2>`，依此類推。每次傳入 1–9 張非空白圖片。prompt 檔案必須是非空白 UTF-8 文字，並會原封不動上傳。如果省略 `--output`，MP4 會存到第一張參照圖片旁邊，檔名加上 `_minimax_h3.mp4`。

預設影片長度是 12 秒；可用 `H3_DURATION_SECONDS` 設為 4–15 秒：

```bash
H3_DURATION_SECONDS=8 ./run_colab_inference.sh \
  --image /absolute/path/reference.png \
  --prompt /absolute/path/prompt.txt
```

Shell 啟動器預設要求 A100 高記憶體 runtime。以下環境變數可以調整設定：

| 變數 | 預設值 | 說明 |
| --- | --- | --- |
| `COLAB_AUTH` | `oauth2` | Colab CLI 認證策略：`oauth2` 或 `adc` |
| `COLAB_GPU` | `A100` | 傳給 `colab new` 的 GPU 名稱 |
| `COLAB_HIGH_MEM` | `1` | 設為 `0` 以不傳入 `--high-mem` |
| `COLAB_EXEC_TIMEOUT` | `3600` | runtime 已熱身後，每個 Notebook 的執行逾時秒數 |
| `COLAB_SETUP_TIMEOUT` | `10800` | 第一次執行 Notebook 的逾時秒數（要安裝 ComfyUI 並下載模型權重） |
| `COLAB_SESSION_NAME` | 自動產生 | runner 建立的可重用 session 名稱 |
| `H3_DURATION_SECONDS` | `12` | 影片長度，範圍 4–15 秒 |
| `H3_SEED` | 隨機 | 要重用的 seed；啟動器的 `--seed` 優先於此變數 |

啟動器另支援 `--seed`（重現同一個結果）與 `--progress`（狀態檔）。未指定 `--progress` 時，狀態檔會寫在輸出檔旁邊，名稱為 `<輸出檔>.progress.json`。為了避免 prompt 內容外洩，超過 400 字元的 log 與含 prompt 標記的行在寫入時會被截短或取代。

## 批次推論

要製作多支影片，請把所有工作放在同一份 manifest，讓它們共用一個 Colab session 與已載入的模型。`jobs.json` 範例：

```json
{
  "jobs": [
    {
      "id": "intro",
      "title": "Presenter introduction",
      "reference_images": [
        "/absolute/path/girl-front.png",
        "/absolute/path/girl-side.png"
      ],
      "prompt_file": "/absolute/path/intro-prompt.txt",
      "duration_seconds": 8,
      "output_name": "intro"
    },
    {
      "id": "demo",
      "title": "CLI demo",
      "reference_images": ["/absolute/path/girl-front.png"],
      "prompt": "A concise Ref2VA prompt referring to <Picture 1>.",
      "duration_seconds": 12,
      "output_name": "demo"
    }
  ]
}
```

在 clone 下來的儲存庫中執行：

```bash
python3 scripts/runner.py batch \
  --manifest /absolute/path/jobs.json \
  --gpu A100 \
  --setup-timeout 10800 \
  --timeout 3600 \
  --output-dir /absolute/path/outputs \
  --progress /absolute/path/outputs/progress.json
```

全新 runtime 的第一個工作要負責安裝 ComfyUI 與下載 39–59 GiB 的權重，因此它使用的是 `--setup-timeout`（或 `COLAB_SETUP_TIMEOUT`）；同一 session 的後續工作則用 `--timeout`。Notebook 會在 Colab CLI 終止執行前幾分鐘先用自己的期限停下，所以過慢的工作會回報卡在哪裡，而不是只剩一個生硬的逾時。

runner 會先檢查 Colab CLI 版本，再驗證所有本機輸入，逐工作上傳圖片與 prompt、執行 Notebook、下載並驗證 MP4，再處理下一項。後續工作失敗時，已完成的輸出仍會保留；prompt 一律放在私有暫存目錄，不會寫在 manifest 旁邊。

Session 的處理取決於是誰建立的：沒有 `--session` 時，runner 會自行命名、建立並在結束時停止它自己的 session；有 `--session NAME` 時，若 `colab status` 回報該 session 仍在，就直接重用並且不停止它，否則先建立再於結束後停止。想在佇列完成後停止重用的 session，請加上 `--stop-on-complete`。

也可以直接執行已安裝技能中的 runner，不需要留在 clone 下來的儲存庫：

```bash
python3 "${CODEX_HOME:-$HOME/.codex}/skills/minimax-h3-colab/scripts/runner.py" \
  batch --manifest /absolute/path/jobs.json --output-dir /absolute/path/outputs
```

## Prompt 與圖片規則

- 每個工作需要 1–9 張非空白本機參照圖片。
- 圖片順序固定；`<Picture N>` 只依照 1–9 張上傳圖片的順序對應。
- prompt 檔案會以 UTF-8 讀取（開頭的 BOM 會移除，文字中間殘留的 BOM 會被拒絕）並作為完整 prompt 傳送；runner 不會翻譯、摘要或改寫內容。
- prompt 不可引用超過該工作的圖片數量的 `<Picture N>`。
- 每支影片長度必須是 4–15 秒。
- 工作可以指定 `seed`（0 到 2^64-1）以重現結果；省略時會隨機產生，並記錄在 progress 檔中，之後就能照原樣重跑。
- 工作的 `id` 必須符合 `[A-Za-z0-9_-]{1,48}`；格式錯誤的 `id` 會直接報錯，不會偷偷換成隨機值，因為它會成為遠端檔案名。
- 其他程式需要組合導引式 prompt 時，可以使用 runner 的 `compose_ref2va_prompt` helper，產生 `subject_definitions`、`summary`、`retention_analysis`、有順序的 shot 區塊、`overall_soundscape` 與 `non_diegetic_music`。
- Prompt 寫作可參考 [MiniMax H3 Ref2VA prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)。

參照圖片用於描述人物與畫面的視覺身份，不會自動產生 shot 時間；請在 prompt 裡明確描述時間與鏡頭變化。

## 直接使用 Notebook

`assets/MiniMax_H3_Turbo_Colab.ipynb` 就是 runner 上傳並執行的 Notebook，也可以手動在 Colab 開啟來檢查或除錯。runner 透過環境變數選擇 reference mode、遠端圖片路徑、prompt 路徑、影片長度、seed、輸出路徑，以及 Notebook 自身的期限 `H3_JOB_TIMEOUT_SECONDS`。想在 Colab 畫面看到完整 prompt（預設只印長度與 SHA-256 摘要）時，請在 cell 裡設定 `H3_VERBOSE_PROMPT=1`。請讓本儲存庫中的 Notebook 與 runner 一起維護，以保持相容。

## 疑難排解

| 現象 | 處理方式 |
| --- | --- |
| 找不到 `colab` | 執行 `uv tool install google-colab-cli`，並確認 uv 工具的 bin 目錄在 `PATH`。該 CLI 只支援 Linux 與 macOS。 |
| OAuth 或 usage 失敗 | 互動式執行 `colab --auth=oauth2 usage`，依照印出的 Google 授權流程完成設定。 |
| GPU 配置失敗 | 檢查 Colab 方案、compute-unit 餘額、要求的 GPU 與高記憶體可用性；可嘗試 `--no-high-mem` 或其他支援的 GPU。 |
| 第一個工作逾時 | runtime 很可能還在安裝 ComfyUI 或下載權重。請提高 `--setup-timeout`（或 `COLAB_SETUP_TIMEOUT`），或讓多個工作共用同一個 session。 |
| 批次逾時 | 遠端 kernel 可能仍在執行，不要直接重試。runner 會嘗試停止自己建立的 session，並在 progress state 記錄已完成工作。 |
| Prompt 圖片驗證失敗 | 讓 `<Picture N>` 對應該工作 1–9 個 `reference_images` 的 1 起始順序。 |
| progress log 出現被取代的文字 | 這是刻意的：超過 400 字元的行會被截短，prompt 內容會被占位符取代；上傳的 prompt 仍是完整原文。 |
| 輸出檔存在但沒有聲音 | 影片會保留並列為警告。請安裝 `ffprobe` 檢查串流，或改用 `--require-audio` 讓這種工作失敗。 |

## 離線驗證

不需要 GPU 或 Colab session 即可執行儲存庫測試：

```bash
python3 -m unittest discover -s tests -v
python3 scripts/runner.py --help
./run_colab_inference.sh --help
```

測試會以本機 fake `colab` 與 `ffprobe` 取代真實程式，因此不會消耗 compute units，也不會存取認證資料。Notebook 測試會編譯每一個 code cell，並檢查輸出檔會在媒體檢查之前先複製出來。部分測試在無法適用的環境會跳過：Shell 啟動器需要 POSIX shell，Colab CLI 介面測試需要 `PATH` 上有 `colab`。GitHub Actions 會在 Linux 裝好 CLI 後執行同一套測試。

## 範圍與安全

本儲存庫不包含 Google 憑證、Token、模型權重或產生的影片。Colab session 會消耗帳戶的 compute units。請不要把秘密資料放在 prompt、manifest、log 或上傳檔案中；開始真實批次前，先確認 GPU 與 timeout 設定。

Notebook 會從 `Comfy-Org/MiniMax-H3`、`drbaph/MiniMax-H3-Turbo-Lora-ComfyUI` 與 Qwen3-VL 文字編碼器下載 checkpoint，並透過 ComfyUI 與 ComfyUI-VideoHelperSuite 執行。這些模型與套件各自帶有授權條款與使用規範，本儲存庫不授予任何額外權利；個人實驗以外的用途，請先查閱各自的條款。至於本儲存庫自己的腳本與文件，目前並未附任何授權宣告。
