# dvaaudiodsp.dll 逆向笔记 —— Radius 变速（time-stretch）引擎

> 目标：从 `dvaaudiodsp.dll`（Adobe Audition 2024, 2.3MB, 2697 导出）逆向
> `dsp::stretch::radius::RadiusEngine` 的变速实现，在 radius-rs 中复刻。
> 静态分析：IDA（MCP @ 127.0.0.1:13337）；动态分析：x64dbg（MCP @ 127.0.0.1:3000）。
> 所有地址为 IDA 基址 0x180000000 下的 RVA 可直接换算：`runtime = 0x7FF996630000 + RVA`（本次会话）。

## 1. 对象层级

```
RadiusEngine::Create (0x1468D0, 导出)
  └─ new 0x18 字节 RadiusEngineImpl (ctor 0x146940)
       vtable = ??_7RadiusEngineImpl@radius@stretch@dsp@@6B@ (0x1BF398, 32 槽)
       +0x08 → RadiusImplTD   (ctor 0x147E60; 基类 ctor 0x151280(base, 1024))
       +0x10 → RadiusImplVoc  (ctor 0x14CF60; 基类 ctor 0x151280(base, 8192))
```

- 外层 32 个虚函数全是 thunk：按 `*(BYTE*)(state+8)` 选择 TD / VOC 内引擎。
  （本次动态会话确认勾选"声码器模式"走 VOC 路径。）
- 内引擎对象 0x90 字节，真正算法都在 per-channel 状态里：
  - TD: prepare = 内虚表槽 20 → sub_180149710，new 0xF40 字节/通道
  - VOC: prepare = sub_18014DB20，new ≈0x2500 字节/通道
- 主处理：外层槽 27（0x146CC0）→ 内槽 22；排空渲染：外层槽 28（0x1470A0）。

## 2. 内引擎虚表（VOC，0x1BF6D0；TD 平行结构，0x1BF5B8）

关键槽位（VOC 一侧，槽号 = 偏移/8）：

| 槽 | 函数 | 语义（动态+静态确认） |
|----|------|----------------------|
| 1 | 0x14EBC0 (0x1C0) | **参数提交 commit**（见 §4） |
| 3 | 0x14E940 | SetStretch(pending)：`[st+280]=x; flag[296]=1; if(!flag[296旧]) [288]=[264]`…（存 pending double） |
| 4 | 0x14E540 | stretch 变更 + 位置重映射 + 向 renderer ramp 推点 |
| 13 | 0x14E720 | SetPitch(float a3)：`[st+252]=a3; renderer_ratio = 2^(−a3/12) × [st+272]` 尾跳 setRatioF32 |
| 22 | 0x14D7A0 | 处理块入口（DataTail 缓冲 → 调度渲染） |

共享小子函数（作用于 renderer = state+584）：
- `0x15E460 setRatioF32`: `[renderer+0x1D84] = xmm1`
- `0x15E450 setI`: `[renderer+0x52C] = edx`（块大小 1024）、`[renderer+0x528] = 1`
- `0x15E550`: `[renderer+16] = a2`（**stretch**）；`0x15E540`: `[renderer+24] = a2`（**pitch ratio**）
- 两者都触发 ramp 求值器 `0x15E470`
- `0x147B50 rampPush`: 向 `renderer+0x38` 的 std::map 推 (key=位置, v1, v2) 节点（0x38 字节）

## 3. VOC 通道状态字段图（偏移为十进制）

| 偏移 | 类型 | 含义 |
|------|------|------|
| 252 | f32 | pitch（VOC13 的 a3 原值；shell 传入 −P，即 +3 半音存 −3.0） |
| 256 | f64 | **当前 stretch（时间图斜率）**，200% 时为 2.0 |
| 264 | f64 | 当前 pitch 半音 |
| 272 | f64 | **调度比率 = 2^(P/12)**（input_per_output） |
| 280 | f64 | pending stretch（VOC03 写入） |
| 288 | f64 | pending pitch 半音 |
| 296 | u8 | stretch 待提交标志 |
| 297 | u8 | pitch 待提交标志 |
| 304 | i64 | 上次 commit 时的输入位置 |
| 312 | i64 | 时间图输入基点 in_base |
| 320 | f64 | 时间图输出基点 out_base |
| 352 | — | 颗粒合成子对象（发射函数的第 2 参） |
| 448 | DataTail | 输入缓冲 |
| 528 | i32 | 触发保持（GetTotal − 此值 = 可处理帧数） |
| 552 | i64 | **输出时间线位置**（已发射帧计数） |
| 584 | renderer | 见 §5 |
| 696 | 队列 | 输入侧事件队列（transient 增益事件输出） |
| 1768 | i64 | 颗粒环可用输入量 |
| 9312 | i64 | flush 目标位置（−1 = 无） |
| 9320 | i64 | 输入基（DataTail 基） |
| 9328 | i64 | 输入累计帧数 |
| 9360 | f64 | **输入环小数余量 rem**（颗粒发射的输入侧位置） |

## 4. 参数提交 commit（sub_18014EBC0）——核心公式

shell（auaudiocomponents.dll，调用点 0x7FF978076xxx）按序推：VOC03(stretch)、
VOC13(pitch)；处理循环开始时 commit 生效：

```
if (输入基 == 0 或 已前进 ≥ 32 帧):
    if (flag[296]):  # stretch 提交
        out_base = (cur_in − in_base) × stretch_old + out_base   # 时间图基点重映射
        in_base  = cur_in
        stretch  = pending                        # state[256] = pending
        renderer[+16] = stretch                    # 0x15E550 → ramp 重估
        flag = 0
    if (flag[297]):  # pitch 提交
        out_base = (cur_in − in_base) × stretch + out_base      # 同样重映射
        in_base  = cur_in
        pitch_semi = pending                       # state[264]
        ratio = 2^(pitch_semi/12)                  # state[272] = 2^(P/12) —— 正号！
        renderer[+24] = ratio                      # 0x15E540 → ramp 重估
        setRatioF32(renderer, 2^(−state[252]/12))  # renderer[0x1D84]，252=−P → 2^(P/12)
        flag = 0
```

动态实测（伸缩=200%、变调=+3、声码器模式）：
- state[256] = 2.0，state[272] = 1.189207115 = 2^(3/12)
- renderer 斜率 = **2.378414 = stretch × 2^(P/12) = 2.0 × 1.1892**
- renderer[0x1D84] = 1.1892071（float）

## 5. renderer（state+584）字段

| 偏移(十进制) | 含义 |
|------|------|
| 16 | stretch (f64) |
| 24 | pitch ratio (f64) = 2^(P/12) |
| 32 | **斜率 slope = stretch × pitch_ratio**（0x157BC0 重算，见 §7） |
| 40 | intercept（ramp 截距） |
| 48 | basepos（ramp 基位置） |
| 56 (+0x38) | std::map ramp 点表（(key=位置, v1, v2) 节点 0x38B） |
| 88 | "settle" 标志 |
| 0x52C=1324? (见 setI) | 块大小（1024） |
| 1344 | ramp 链表指针 |
| 1352 | curpos（当前位置） |
| 1336 | **quality**（int；≥10 走高质量表） |
| 1384/1396/1448/1452 | hop 几何（见 §7） |
| 2448 | ramp 状态机 state |
| 0x1D84 | ratio (f32) = 2^(P/12) |

## 6. 调度循环（sub_18014DF50）——输出长度怎么来

```
处理 stretch 变更队列(state+328)：在输入位置 K 处：
    out_base = (K − in_base)×stretch_old + out_base; in_base = K; stretch = 新值

可用输入 avail = ring(1768) − 100（安全边距）
若 ramp 队列(state+656) 下一节点 key ≤ avail：在 key 处分帧（v24=0）
否则 v24=1（本块不再有更多输入）

next_out = round( out_pos(552) + (avail − rem(9360)) / ratio(272) )
flush 时 next_out = min(next_out, flush_target)

发射（out_pos < next_out 时）：
    n_out = next_out − out_pos
    sub_1801515C0(engine+8, state+352, ch, hop, rem_floor_to_hop, out, out_off,
                  n_out, ratio, ratio, 1)
    rem += n_out × ratio          # 输入消耗 = 输出 × 2^(P/12)
    out_pos = next_out
```

**实测（200%、+3）**：Δout×1.189207 = Δrem 精确成立（如 2112↔2511.6）。
即调度层只按 pitch 比率做 varispeed 记账（out = in / 2^(P/12)），
**stretch 全部由 renderer 斜率（S×2^(P/12)）在颗粒合成内部改变 hop 实现**。

## 7. 斜率重算 sub_180157BC0（= hop 几何选择）——动态验证版

2026-11-03 晚用 x64dbg 对运行中的 AU 预览（q=2、slope=2.3784、48 kHz）实测：

| 字段 | live 值 | 推导 |
|---|---|---|
| [1336] quality | 2 | 预览低质量（离线 High ⇒ ≥10） |
| [1368] hop | 132 | round(round(7184×0.7/(4<<2))/2.3784) = round(314/2.3784) ✓ |
| [1372] step | 314 | round(132×2.3784) = round(313.95) ✓ |
| [1384] | **7184 = n_write** | 锚点确认 |
| [1448] | **16384 = n_fft** | 锚点确认 |
| [1452] | **8193 = nb_bins** | 锚点确认 |
| [1392] span | 38 | round(7184×0.7/132) ✓ |
| [1396] | 8193 | q<10 ⇒ 直接取 [1452] |
| [2452] max_w | 27 | round(38×0.7×max(√1.0,1))，f=0.7（q<10）——按**更早 slope≈1** 的快照，说明 max_w 不随每次 slope 变更即重算 |
| [2448] 机器状态 | 2（处理中） | appr/settle 参数活跃 |

最终 hop 律（分支按 quality，斜率作用在除法上）：

```
base = round(n_write[1384] × 0.7 / D)      # D = 4<<q (q<10)；{3,3,6}[min(q−10,2)] (q≥10)
hop  = round(base / slope)                 # [1368]
step = round(hop × slope)                  # [1372]
span = round(n_write × 0.7 / hop)          # [1392]
max_w = round(span × f × max(√slope,1))    # [2452]，f = 1.4 (q≥10) / 0.7 (q<10)
```

48 k、High（q≥10）时 D=6：base=838，hop=round(838/slope)（slope 1.19 ⇒ 704，
约 14.7 ms 机器粒）。**机器粒 ≠ 调度表 hop（3592）**：调度表驱动外部读入/输出，
机器粒只进 transient 状态机的查询/保持计数。

### 7.1 反编译定版 + 44.1k 实测（2026-11-05 凌晨）

sub_180157BC0 全量反编译（.tmp/ida/geometry.c），上表公式**逐行确认**，
并补三条新律：

- **n_write 律（两采样率实测钉死）**：`n_write = 3×sr/20 − 16`
  （48k→7184 ✓，44.1k→6599 ✓，live 均吻合）。
- **14 kHz bin 上限（q≥10 才有）**：`[1396] = min(nb_bins,
  round(n_fft×14000/sr))`；q<10 时直接取 nb_bins。48k/High/16384 ⇒ 4778。
- **[1400] = hop×slope（double）**；窗口表重建：逐 bin 相位斜坡
  `2π·k·hop/n_fft` 写 [2208]；Hanning 窗长 `nw×v122[ch]`（v122 初值
  {3,3,6} 系数表同 D！窗口重叠档=除数档）偶数掩码（&0xFFFFFFFE），
  归一 `pow(nw/(nw×0.825),1)`、0.65 系数窗、`v35 = q<10 ?
  2/√(2^q) : 4.0` 的窗值钳 `min(v35×w, 1.0)`。

44.1k 预览实测（q=2、slope=1.189、+3 半音/tempo 100）全对账：
n_write=6599 ✓、D=16 ✓、base=289=round(6599×0.7/16)、hop=243、
step=289=round(243×1.189)、span=19 ✓（19.005）、max_w=13 ✓
（19×0.7×1）、[1396]=4097=nb_bins（无上限，q<10）✓、发射整数位
[424]=512=1.0×2⁹（tempo 100）。

**n_fft 采样率驱动（预览档）**：48k 预览 q=2→16384；44.1k 预览 q=2→
8192（质量切换复测不变）⇒ 至少 q=2 档 n_fft 随 sr；High 档 44.1k 的
n_fft 未定（离线渲染在子进程，未测）——AU 44.1k 语料 A/B（0.85 档，
2026-11-05 已测）支持 8192。

### 7.2 四采样率实测表（2026-11-05，预览档 q=2，逐个开预览探测）

| sr | [1448] n_fft | [1384] n_write | 0.15sr | Δ | base/hop/step/span |
|---|---|---|---|---|---|
| 32k | 8192 | 4816 | 4800 | **+16** | 211/177/210/19 |
| 44.1k | 8192 | 6599 | 6615 | **−16** | 289/243/289/19 |
| 48k | 16384 | 7184 | 7200 | **−16** | 838/704/838/38 |
| 96k | 32768 | 14416 | 14400 | **+16** | —（管线自洽） |

- **n_fft 四点全部与我们的 octave 量化一致**（32k→8192、96k→32768；
  44.1k 为唯一"低于 log2 舍入预测"的例外，降档边界在 44.1k↔48k 之间，
  非标准采样率的边界位置仍未钉）。
- **n_write 的 ±16 符号翻转**（44.1/48k→−16、32/96k→+16）无简单闭式；
  误差 0.1% 级（base ≤0.7、hop ≤1 帧），radius-rs 通用分支保持
  `3×sr/20−16`（两锚点精确 + 另两点 32 帧内近似）。
- 几何管线（base→hop→step→span→max_w）在四个采样率全部自洽 ✓。

## 8. 颗粒发射 sub_1801515C0（部分）

- 入参：(engine+8, state+352, channels, hop, ring_offset(=rem 向下取整到 hop 倍数),
  out, out_off, n_out, ratio(int bits), ratio(double), flag)
- `a9 == 1.0` 且 flag 时：环形直拷贝（ratio=1 的旁路）
- 颗粒调度表：`engine+40`（表1：int 数组）与 `engine+72`（表2：24 字节条目
  (begin,end,…)=颗粒窗口），按 `sched_idx = clamp(((ratio−1)/2)×n_sched)` 选桶，
  桶内按小数位置插值取条目 —— 与 pyradius 的 vc_schedule（我们移植的 28993 条表）同源。
- 输出侧 overlap-add：窗口 × 环数据累加到输出。

## 9. 与 pyradius/radius-rs 的对应与实现（2026-11-02 完成）

- pyradius `VocoderState.set_ratio(semis, quality)` ↔ renderer ratio/quality/hop；
- **变速 = 调度层时间图 out(in) 分段线性（斜率 S），加 renderer hop 缩放 S×2^(P/12)**；
  ~~我们的 stretch.rs 目前用 WSOLA 近似~~ —— 已删除，按本笔记在 vocoder/td 内实现真时间图。
- 待核对：sub_1801515C0 内 hop 与 slope 的精确关系（颗粒输入 hop vs 输出 hop）、
  TD 引擎对应字段（state+368/400/416/424 双精度区，TD13: ratio=2^((P−v5)/12)）。

## 10. 动态抓取实录（2026-11-01 会话）

- AU: Audition 2024 @ D:\Program Files\Adobe\Adobe Audition 2024
- dvaaudiodsp.dll 基址 0x7FF996630000；效果外壳调用点在 auaudiocomponents.dll
- 断点布点见 `.tmp/ida/`（poll_trace.py 自适应跟随新分配的通道状态）
- trace.log 记录了 Apply 全流程：VOC03 → VOC13 → setRatioF32 → commit(0x14EBC0)
  再写 +272 & setRatioF32

## 11. Rust 复刻与验证结论（2026-11-02）

### 双速率定律（两个引擎同构）

| 量 | 声码器 | TD |
|---|---|---|
| 排空速率（drain，读环→最终输出） | `r = 2^(P/12)`（InterpNSamples 的 ratio） | `r`（同左，`pitch_ratio`） |
| 调度斜率（slope，环写入/输入帧） | `S×r`（`VocoderState::slope`，驱动 `vc_calc_sched` out-step 与 `comp_1432`） | `S×r`（`TdState::total_ratio`，驱动 acc 额外颗粒 = 重放窗口 → 环内为时间拉伸素材） |
| 输出长度 | `round(in × S)` | `round(in × S)` |

TD 实测（au-raw.wav，1,393,598 帧）：环写入终值 = slope×in **精确成立**于
+3（1,657,474 vs 1,657,058）、−3（1,172,290 vs 1,171,866）、+3/200%（2,787,332 vs
2,787,196）、+3/50%（829,268 vs 828,609）；排空后输出长度全部落在目标的 0.02% 以内。

### 验证指标（vs AU 参考，包络/频谱相关，42ms hop 包络、12 段对数频谱）

| 情形 | 长度比 | 包络 corr | 频谱 corr | RMS 差 |
|---|---|---|---|---|
| td+3（基线） | 0.9997 | 0.964 | 0.968 | 0.20 dB |
| td−3（基线） | 0.9997 | 0.920 | 0.981 | 0.03 dB |
| td+3 200% | 0.9999 | 0.927 | 0.956 | 0.75 dB |
| td+3 50% | 0.9998 | 0.901 | 0.948 | 0.01 dB |
| vc+3（基线） | 1.0000 | 0.936 | 0.968 | 0.20 dB |
| vc−3（基线） | 1.0000 | 0.906 | 0.911 | 0.30 dB |
| vc+3 200% | 1.0000 | 0.923 | 0.945 | 0.13 dB |
| vc+3 50% | 1.0000 | 0.789 | 0.956 | 0.46 dB |

变速情形全部落入（或接近）纯变调基线区间；采样级相关对 AU 本身就 ~0.03–0.05
（对 pyradius 位级精确的纯变调渲染同样如此——AU 的颗粒相位不同，见
tests/audition_reference.rs 文档）。cargo test 157 项全过，含全长位级验收
（vocoder_matches_pyradius_reference / vocoder_acceptance_render /
vocoder_pitch_matches_audition）。

### 踩过的三个坑（TD 实现时）

1. **排空相位推进量写成了 `total_ratio × count`** —— 排空按 `pitch_ratio` 读取却
   跳过 `(S−1)×` 已读内容，任何 tempo≠100 直接频谱错乱。正确值 = 实际消费量
   `pitch_ratio × count`。
2. **颗粒泵停止条件用了 target_out** —— 变速靠重放窗口而非多喂输入，泵应在
   真实输入耗尽时停（`nstop = nframes`），否则 2× 变速会处理 2× 的"输入"
   （后一半是 EOF 后的零），输出翻倍。
3. **外层供料循环的界也只看 target** —— 50% 变速时 target < in，供料在 52%
   处断粮；界须取 `max(target, nframes)`（1115 行的 break 与声明处都要改）。

### 残余 → 已解决（2026-11-02 深夜第二次实测）

vc 50% 包络 corr 0.789 偏低的根因：slope<1 时 `vc_calc_sched` 无条件走 √2 分支。
**AU 实测（用户配合渲染 +3/50%，断点抓取 + renderer dump）判定：分支条件看的是
音高比 r，而不是斜率** —— 排空序列五次验证 n_out=round((ring−100−rem)/r)、
rem+=n_out×r 精确成立；环终值 828795≈slope×in（0.5946×1393598）；renderer
slope=0.5946、quality=2。离线四变体对照 au-vc+3_50 扫描：

| 变体 (in/out) | 包络 corr | 频谱 corr |
|---|---|---|
| 314/187（pyradius √2 按斜率，原实现） | 0.789 | 0.956 |
| 314/157（√2，按 S） | 0.323 | 0.854 |
| **222/132（无 √2，按斜率）✓** | **0.843** | **0.963** |
| 222/111（无 √2，按 S） | 0.321 | 0.851 |

最终定律：`in_step = round(222)`（r≥1）或 `round(222×√2)`（r<1，与纯变调位级
语料一致），`out_step = round(in_step × slope)`。纯变调 ±3 在新定律下逐值不变
（222/264 与 314/264），位级校验全过；唯一行为变化是"音高≥1 且斜率<1"（+3/50%：
222/132）及其镜像（音高<1 且斜率≥1）。按 S 步进的两个变体崩到 0.32——反证环
必须按斜率步进。`vc_calc_sched` 签名改为 (g, pitch_ratio, slope, step)。

### 自动化验收（tests/audition_reference.rs，2026-11-02 起真实运行）

参考查找已指向 `.local-testdata/`（`RADIUS_AUDITION_REF_DIR` 仍可覆盖；输入回退
`au-raw.wav`），AU 验收测试不再空跳过，全过：

```text
td +3: level -0.21 dB, contour corr 0.938      td +3 200%: level -0.77 dB, contour corr 0.866
td -3: level +0.03 dB, contour corr 0.892      td +3  50%: level -0.02 dB, contour corr 0.867
vc +3: pitch +3.012 (AU 自身 +2.992), level +0.20 dB
vc -3: pitch -2.997 (AU 自身 -2.988), level +0.43 dB
vc +3 200%: level +0.11 dB, contour corr 0.871
vc +3  50%: level +0.41 dB, contour corr 0.818   (修正前 0.772)
```

新增 `tempo_renders_match_audition`（四个变速情形：精确长度 + round(in×S) 断言、
电平 <6 dB、轮廓 corr >0.5）；`td_output_matches_audition_perceptually` 改用
`au-td±3.wav`；`vocoder_pitch_matches_audition` 兼容 `au-vc±3.wav` 命名。
cargo test 全套 157 项 + 验收全过，tempo=100 路径与 pyradius 位级一致不变。


## 12. 六张常量表与全采样率（2026-11-02 深夜，静态分析）

### DLL 里没有那六张表

对 dvaaudiodsp.dll 整个镜像做字节签名搜索（IDA `find_bytes`）：

- fir48/fir44 的 f32 抽头字面量、以及按 f64 编码（低 32 位为 0）的同一数据：0 命中；
- win48/win44 各频带中窗的签名序列、synth44 的签名序列：0 命中；
- Radius 区域（0x18014xxxx–0x18016xxxx）内 44100/48000/32000 的 int 或 float
  立即数（IDA `find type=immediate`）：0 命中。

结论：原引擎不查任何按速率挑选的表。采样率是运行时数据（渲染器 +0x60/+0x108
携带 48000，quality=2 在 +1336），一切在初始化时按 sr 现算。

### 分频器设计器（sub_18015E560 → sub_1800FA4B0）

vocoder 初始化 `sub_18015E560` 内的关键调用：

```text
sub_1800FA4B0(a1+2512, quality>=10 ? 1 : 4, 2048,
              {150.0f, 800.0f, 4500.0f},   ; 三个分频点（Hz）
              bitcast(sr),                  ; 采样率作参数
              0x3DCCCCCD,                   ; 0.1（过渡带系数）
              9.0)                          ; DesignFIR 窗参数
```

设计器反编译要点（~16.8KB 伪代码）：

- 1025 个频率 bin（2048/2+1），bin 频率 = bin×rate/2048；
- 相邻分频点在 log-f 空间的几何平均处放过渡带中心（`edge = (log f_k + log
  f_{k+1})/2`）；DC 哨兵 −10.0，被压扁的低频带哨兵 −20.0，Nyquist = rate/2；
- 过渡带半宽 = `(1−0.1)×min(log 距离)`；
- 各带期望增益 = log-f 上的升余弦（升沿 `0.5−cos(π(x−lo)/(hi−lo))·0.5`，降沿
  `cos(...)·0.5+0.5`），相邻带幅度互补（和≈1，数值上快照实测为精确 1）；
- 最后 `dsp::FIR::DesignFIR(a1+320, response, 2048, 9.0, taps, …)` 合成抽头。

**设计是 Hz 锚定的**（150/800/4500 Hz 不随速率变），所以连续时间冲激响应在
任何速率下同形——这是时间轴重采样重构的依据。

### 窗定律（两个快照速率数值验证）

- 频带 1–3：纯 Hann，L = round(n_write×{0.5, 0.3, 0.2})，起始 (nw−L)//2，
  能量归一 Σw² = n_write；
- 频带 0：两段式，上升沿 = hop（精确），下降沿 ≈ 0.325×nw，起点
  round(0.044×nw)，峰值尺度是**跨速率恒定常数 1.9793857**（两快照逐位相同）；
- 合成窗：Hann^0.7，长窗 ×√(短/长)；48k 为 2930/1256（A 长且带尺度），44.1k
  为 1152/2688（角色互换）——非规范速率按 48k 约定随 n_write 缩放。

### radius-rs 的全采样率实现（src/vocoder/rate_gen.rs）

- `supported_rate` 放开为 8000..=192000；44.1/48k 仍走快照表（48k 位级语料
  红线不动），其余速率用生成库；
- FIR：48k 快照时间轴重采样 `h_sr[n] = (48000/sr)·h_48k(n·48000/sr)`，
  带限 sinc 插值（Kaiser β=10，±48 抽头），降速率时截止在目标 Nyquist；
  >Nyquist 时 2048 抽头装不下全长冲激响应，全局 DC 锚定恢复带 0 增益；
- 窗/合成窗：按上述定律生成。快照自身的个别长度取整（48k band2=2154 vs
  round(0.3nw)=2155 等）无单一规则可复现，生成库用最接近的干净规则（≤1 样本
  差，仅存在于无参考的速率上）。

### 对比测试（tests/rate_generic.rs）

- 生成窗库 vs 两个快照：干净规则的频带 ≤1 ULP / 7.6e-5，其余 ≤2e-3；
- 重采样 FIR vs DLL 自家 44.1k 快照：−6dB 边带差几 Hz、响应差 <1e-4（20 kHz
  以下）、幅度互补和在 20.5 kHz 以下 <1%（快照本身精确平坦为 1；目标 Nyquist
  前 ~1 kHz 有百分之几下垂——把一条平坦到 24k 的响应降采样到更低 Nyquist 的
  固有代价，DLL 按速率重新设计无此限制）；
- 32k：n_fft=8192 类、tempo100 精确保长、tempo50 精确减半（双速率定律）、
  音高比实测 1.189±0.02；96k（32768 类）/16k 渲染健全；
- CLI：22.05k 完整渲染通过；5000 Hz 仍拒绝（"unsupported sr"）；
- 全套 cargo test 170 项过，48k 位级校验与 AU 验收不变。


### 表的移除（2026-11-02 凌晨，用户指令）

六个快照表连同 table_data/ 目录与生成工具一并删除，引擎全速率走设计器：

- FIR：log-f 升余弦 spec（边带 150/800/4500 Hz、半宽 H=0.70 ≈ ±1 倍频程，
  同一 H 在两个快照上同时最优——这是"设计常数"而非单速率巧合的证据）
  频率采样成 2048 抽头；全通 δ 放在 1023（引擎环形散射硬编码补偿的延迟）。
  对照退役快照：−6dB 边带落在相同 bin（141/164、797/820、4477/4523 Hz@48k）、
  响应最大偏差 ≈2%（rms 0.4%）、bin 网格上幅度互补精确成立；
- 窗/合成窗：沿用上文本节测得的定律（窗 ≤1 ULP、合成窗 Hann^0.7）。

对比验证：48k 29s 验收输入，删表前后二进制渲染——包络 corr 0.78(+3)/
0.70(+3@50%)、对数频谱 corr 0.9997/0.9994、RMS 电平差 <0.01 dB；采样级
corr 塌到 ≈0.2（颗粒相位解耦，等价滤波数据的固有现象，与"旧引擎 vs AU
本身 ≈0.03–0.05"同理）。AU 真值验收全过且指标与表版本同级（vc+3 音高
+3.008 vs +3.012、电平 +0.20 dB 持平、vc+3/50% 轮廓 corr 0.822 vs 0.818）。
pyradius 位级语料随表退役，vc_acceptance 的参照测试改为感知阈值。


### 分频器定律完整反编译与窗定律修正（2026-11-02 上午）

上文的 H=0.70 拟合是过渡方案；本轮把设计链条完整反编译，定律本身落地：

- 链路：`sub_18015E560` → `sub_1800FA4B0`（spec 构造）→
  `dsp::FIR::DesignFIR@0x1800348D0`（hermitian c2r 逆 FFT，1/N）→
  `dsp::AudioProcessor::ApplyKaiserWindow@0x1801634D0`（长 2047 Kaiser，
  β=9，模式 2 无后归一化）；
- 斜坡中心 = 相邻分频点的 log 几何平均；半宽 = 0.9×min(中心到两侧几何
  中点的 log 距离)；最外法兰 `ln(75)` 与 `geo-mid(4500, nyq)`；bin 相位
  f64 计算后一次转 f32，边界比较在 f32。44.1k"宽度异常"同定律自然消解；
- 对照两张退役表：每 tap 最大相对差 2.9e-7（2–3 ulp；残差是 DLL 内 c2r
  FFT 运算序，非定律）。

窗定律同轮修正（两快照 12 个长度数据点定案）：

- 频带 1–3：`L = 2·⌊r·⌈nw/2⌉⌋`（48k {3592,2154,1436}、44k
  {3300,1980,1320} 全中——上文"无单一规则"已被此规则推翻）；能量目标
  `Σw² = 2·⌊nw/2⌋`（奇 nw 时旧目标差 1）；
- 频带 0：fall/start 取 floor，常数钉在 0.32489 / 0.044（区间两端命中
  两快照）；
- 数值残差 ≤3e-7 相对（DLL `cosf` 多项式 vs 正确舍入 cos；f32 原生变体
  试过更差）。

渲染对比：29s 验收输入 +3 半音，删表前后采样级 corr 0.22 → 0.78，包络
0.78 → 0.88，前 5s corr 0.98。隔离实验（临时 env 钩子加载退役表，已还原）
证明剩余差距为确定性混沌：双表加载 → 与表时代引擎逐位一致（corr
1.000000）；仅 FIR 差 2–3 ulp → corr 每 5s 0.98→0.76 衰减（逐粒相位反馈
放大任意 f32 扰动）。长渲染逐位一致需逐指令克隆 DLL 浮点运算序，非定律
问题。详见 `docs/TABLE_PROVENANCE.md`。

## 13. vocoder 瞬态子系统完整反编译（2026-11-03）

> 背景：AU 的 vocoder 打击乐/鼓瞬态保真依赖一整套 pyradius 未移植（我们
> 也没有）的子系统。本轮从本 IDB（与 §6–§8 同构建）完整解码其架构。
> pyradius 移植了"==2 门控分支"但从未移植驱动它的状态机与检测器——
> 其 `transient_state_2488 = 0` 是硬编码，我们的引擎因此永远处于稳态。

### 13.1 三大部件

1. **输入侧检测器** `sub_180153EA0`（process_block `sub_18014D7A0` 每块调
   `detector(state+696, 输入, 块长, ...)`）：混音输入进 DataTail（4 chunk
   前瞻缓冲），按 stride 分 chunk（stride=[q+52]，推断 1024），对每 chunk：
   - 双分辨率加窗 FFT（窗/FFT 尺寸在 [q+280]/[q+496]/[q+500]，两套频谱
     [q+100]/[q+116] 与 [q+236]/[q+252]）；
   - 功率谱→24 频段求和（段边界表 [q+916]，24 个 int）；频段能量前向+
     反向一阶 IIR 零相位平滑（系数表 [q+1016]）；
   - 三次开方 = x^(1/8) 压缩（v61 长度 [q+256]、v74 长度 [q+260]、24 段
     SIMD 三连 sqrt）；
   - 9 槽循环历史（[q+592]/[q+536]/[q+696]/[q+800] 各持有 float* 环），
     谱通量 `F = Σ_band [B(t+1)+B(t+2)+B(t+3)+B(t+4) − B(t−1)−B(t−2)
     −B(t−3)−B(t−4)]`，正项全加、负项按 −0.1 计（不对称惩罚）；
   - 除以频段数得 chunk 分数，写入分数数组（[q+0] 起的 vector）。
   后处理：`sub_1801525E0`（0xab3，逐 9-chunk 组精化）、`sub_180153190`
   （非极大值抑制：邻域 [score×a6×0.7 若 a9] 更高者清零，半窗
   round(a5×a8/stride+0.5)）、`sub_180153570`（增益曲线：阈值 a7 以上的
   chunk 在 ±round(a5×a8/stride) 窗内取加权包络）。

#### 0xab3（sub_1801525E0）完整定律与移植（2026-11-04，已移植）

对象：a1+13=stride(128)、a1+44=sr、a1+226=已精化计数、a1+66=base、
a1[64]=live 分数缓冲、a1[3]=精化快照、a1[70]=原始分数、a1+227=S1 状态。

- **S1 锐化**：`x'[i] = x[i] − 0.8·(a·x[i] + (1−a)·x'[i−1])`，
  `a=TimeToIirA(0.1, sr/stride)`，状态跨调用持久，DC 增益 0.2（去均值
  +沿锐化）；
- **S2 快照**：锐化 span 拷入 a1[3]；
- **S3 阈值**：`x ≤ 1/a1[45] → 0`（a1[45] 未定，暂 1.0，TODO(0xab3) 动态
  确认）；
- **S4 五点交换**：live[i]>阈值时在 snap[i−3..=i+1] 找 argmax（平局取
  先）≠i 则 swap——峰值前向迁移；
- **S5 四趟抑制级联**（龄带逐趟加宽）：窗 t=0.01/0.01/0.025/0.05 s
  （rA=4/rC=9/rD=19 chunk@48k），factor=1/1/2.7/6，杀 i ⟺ 窗内有
  `live[k]·raw[k] > live[i]·factor·raw[i]`——**乘积键=精化分×原始分**；
- 未移植：两次 sub_180153570 增益曲线（τ≈0.03/gain 3.0、τ≈0.035/gain
  1.0）与宿主事件导出（滞后 ≈51 chunk——宿主流远慢于机器查询，佐证
  双速结构：机器读 live，宿主读精化缓冲）。

移植：`TransientQueue::refine()`（S1–S5），8-chunk 块节流（DLL 每
process block 一次）；A/B（AU 三基准）：假 state-2 触发 **−69%~−93%**
（534→35 @tempo50），tempo200/50 轮廓持平或升、attack 三升，tempo100
轮廓 −0.017（S3 阈值猜测嫌疑最大）。

#### B2 噪底真参考（2026-11-04 调查，未移植）

我们的 b_710（谱#2 幅度包络）**域就错了**。真链（0x18015c8f0/
0x18015d289）：分析谱逐 bin Amp2DB → FFTFwd（dB 谱再 FFT = **倒谱
域**）→ 交错对 re²+im² 功率 → `TimeToIirA(τ, sr/机器hop)` IIR 平滑进
a1+7768（τ=0.005 ⟺ state==2）；第二处先以全局标度 √(acc/(n+den))
归一化 a1+7816，再用同律 IIR **把工作谱（a1+7864）拉向该参考（吸引
子）**。移植需先建 dB→FFT→功率→IIR 链并重推 phase.rs B2 的比较语义
（相位重置阈值的消费方式整个不对位）；ts_rpt_b2 维持默认关。
2. **分数队列**（renderer+112 = state+696，即 §3 的"transient 增益事件输
   出"）：`{base=[q+48], stride=[q+52], head=[q+264], scores=[q+0..8]}`。
   chunk i 的名义位置 = base + i×stride（i 为未压缩索引）；分数存
   scores[i−head]。查询原语：
   - `sub_180153BD0(q, pos, start, thr, mode, cap)`：v14=(pos+stride/2−
     base)/stride；mode 1 向后找（idx−head 递增，score>1/thr 命中返回
     base+stride×idx），mode 2 向前找，mode 0 双向；start 通常 1。
   - `sub_180153CD0(q, pos, range)`：[pos, pos+range] 内**分数最大**的
     chunk 位置（无则 −1）。
   - `sub_180153A70(q, pos, range)`：[pos±range] chunk 分数的
     RMS = sqrt(mean(score²))。
   输出侧事件流（process_block LABEL_10）：新 chunk 若 score>0，向宿主发
   `{位置=base+idx×stride, 增益=clamp(score×0.1, 0, 1)}`（16 字节记录）。
3. **调度状态机**（在颗粒链 `sub_180158D20` 内，0x56A1 大函数，含 ACS/
   APC/RPT/合成全链）。字段（renderer 相对）：

   | 偏移 | 含义 |
   |------|------|
   | 2448 | 状态 state ∈ {0 稳态, 1 接近, 2 保持/瞬态处理, 3 瞬态回放} |
   | 2452 | 瞬态窗口上限 max_w |
   | 2456/2460 | 保持计数目标/当前 |
   | 2464/2468 | 接近计数目标/当前 |
   | 2472/2476 | 接近段 in-hop / out-step |
   | 2480 | 接近段 hop 增量（置 0） |
   | 2484 | 保持段结束后的 hop 增量 |
   | 2488 | 瞬态回放保持数（≥1） |
   | 2492/2496/2500/2504 | 瞬态段 in-hop / 计数 / out-step / 末粒 hop 补余 |
   | 1360 | 上次接受的瞬态位置（防重） |
   | 1344 | 输入读位置（*v8，每粒 += hop） |
   | 1352 | 输出环形游标 curpos |
   | 1368/1372 | 稳态 in-hop/out-step（调度表在外部步进） |
   | 96 | 采样率（窗口都是 sr×0.05/0.1 等秒量） |
   | 1336 | quality（≥10 走高质量；2794 块仅 <10） |

### 13.2 状态机转移（舍入一律 (int)(x±0.5) 远离零）

- **0→1**（拉伸侧，slope>1）：`ev = QueryFwd(inpos)`；接受条件
  `ev>sr×0.01 && ev≠last && ev−inpos < round(sr×0.05)`；
  next/prev 事件距离 d±=|ev−ev_next/prev|/sr → 斜坡 ramp(d)=
  0/1/(d−0.02)/0.13（d∈[0.02,0.15] 线性）；
  `hold_next=clamp(round((ramp(d+)×0.9+0.1)×max_w),1,max_w)`，hold_prev
  同理；**2456 = round((hold_prev+hold_next)/2)**；
  `mid_out = (ev−in_base)×slope+out_base − hop1368×hold_prev×0.5`；
  `N=2464 = max(1, round((mid_out−curpos)/step1372))`；
  `2476 = round((mid_out−curpos)/N)`；`2472 = round((ev−0.5×hop×hold_prev
  −inpos)/N)`；2480=0；**2488 = N**；2500=2476；
  `tail = round((hop×2456 + 2N×2472 + …)/slope)`→ `2492 = 2472 +
  (残差整除 N)`，`2504 = 残差 mod N`；DataTail(renderer+2248) 清零。
- **0→2**（压缩侧，slope≤1，**两段式**，0x180159bf3–0x180159e9e）：
  1. `cand = QueryMax(inpos, round((hop×0.5×max_w+step)/slope))`；
  2. `near = Query(cand, start=1, thr, mode=0)`（**双向**找最近事件），
     d=|cand−near|/sr → ramp → **2456 = clamp(round((ramp×0.9+0.1)×max_w),1)，
     此写发生在接受判定之前，接受失败也保留**（影响下一粒第二段范围）；
  3. `best = QueryMax(inpos, round((hop×0.5×2456+step/slope)))`；接受
     `best≠−1 && best≠last && best>sr×0.01`：2456 重写为
     `max(1,int((best−inpos)×2×slope/hop))`；本粒 hop/out-step 直接改为
     对齐值（v30=round((best−hop×2456×0.5)−inpos)，v853 同构）；
     **2484 = round(回稳态校正值)**；DataTail 清零（LABEL_164）。
- **1**：hop=2472；2468→2464 后 →2（2460=0，hop+=2480）。
- **2**：hop=1368（稳态）；2460 到 2456 的一半时 renderer+3568 清零；
  到达 2456 后：若 `slope>1 && 2488` →**3**（2496=0）；同时在到达点用
  `round(sr×0.1)` 窗口重新 QueryFwd——若命中新事件则重算全部计划并 →1
  （覆盖为接近段，见 0x1801592e8 块：pre-roll N=min((ev−inpos)×0.9,
  dist_prev×0.8)/hop×min(slope,1)）。
- **3**：hop=2492（末粒 +=2504）、out-step=2500；2496→2488 后 →0。
- **任意(1/2)→0**：斜率变更标志（renderer+88，prestep `sub_180158A60`
  处理 ramp 链表更新时间图后置位）触发时，inpos 按新时间图重映射，
  状态清 0。
- 每粒出口：`inpos += hop`；`renderer+1376=hop`；`+1380=Δ`；
  `+3568 += hop`；`2448 = state`。稳态时 hop=1368、out-step=1372 由外
  层调度表步进（= 我们已移植的 vc_calc_sched 同源）。

### 13.3 state==2 的六处链路门控

| 位置 | 级 | 稳态值 | state2 值 |
|------|----|--------|-----------|
| 0x18015a25c | ACS 包络 IIR τ | 0.1 s | **0.05 s** |
| 0x18015af55 | APC v15（=我们 phase.rs 的 0.5 因子） | 1.0 | **0.5** |
| 0x18015c2e9 | RPT 分支选择：state≠2 走 B3（队列分数 RMS 三元组 `RMS(inpos±hop)`，`cur>两侧 && cur>1.2` 判瞬态）；state==2 走 B2 | B3 | **B2** |
| 0x18015c8f0 / 0x18015d289 | 噪声路径包络 IIR τ ×2 | 0.01 s | **0.005 s** |
| 0x18015b69a | 峰区增强抑制（state≠2 才允许 ×10 增强） | 允许 | 抑制 |

τ 都经 `TimeToIirA(τ, sr/hop)` 折算。注意 pyradius 把"==2"误系到偏移
2488（本构建 2488 是回放保持计数）；正确字段是 2448。

### 13.4 与我们端口的对位（已实现，2026-11-03晚）

实现落在 `src/vocoder/transient.rs`（`TransientQueue`/`TransientDetector`/
`TransientSched`），经 `VocoderConfig::transient_sched` 门控，**默认开启**
（AU 一致性优先；pyradius/libradius 仅作代码参考，不再是输出基准）。
pyradius 位级回归测试用 `set_transient_sched(false)` 显式退出。

- 队列：`TransientQueue` 精确移植 BD0/CD0/A70 三查询 + base/stride/head
  紧凑化；检测器分数按绝对 chunk 索引入队（`push_at`，检测器首个可评分
  chunk 是 4）。**stride = 128 帧**（x64dbg live：renderer+112+52 的
  stride 字段，2026-11-03），即 2.7 ms chunk、±4 梳 span ±10.7 ms。
- 检测器：结构移植：128 帧 chunk、512 点加窗 FFT（窗跨 4 chunk）、24 个
  对数频段（40 Hz–Nyquist）、x^(1/8) 三次开方、9 槽环、±4 非对称梳
  （负项 ×−0.1）、÷24 归一、±2 chunk NMS（0.7 因子）。自归一化设计使
  阈值 >1.0 保持 DLL 量纲（实测鼓点击穿，峰值比爆发 chunk 提前 ≤1 chunk）。
  DLL 的精确窗/频段/IIR 表仍未抓到（构造函数未定位）。
- 状态机：§13.2 全转移精确移植（含两段式压缩侧）。机器粒 = §7 的
  hop1368（48 k High ⇒ 704 帧 ≈ 我们 3.2 个子粒）：机器按**游标追踪**
  步进（上一计划输入跨度耗尽即重新决策），计划期 in/out 目标按
  share=(目标−已到)/剩余子粒 精确分摊；state==2 门控持续整个计划窗。
- hop/out-step 重定时（`ts_hop_override` 诊断开关，**默认关**）：x64dbg
  live 证实 AU 每块（~1024 帧）内 out 速率在 1.0→4.2× 间摆动即此机制；
  但我们端口开了它后轮廓 corr 反而崩（0.65→0.52/0.11，两版几何皆试），
  说明我们子粒格点与 DLL 执行层还有错位——保留为实验路径。
- max_w = round(round(n_write×0.7/hop)×f×max(√slope,1))，f=1.4（q≥10，
  我们 quality=37）/0.7（q<10），live q=2/slope1 快照 = 27 = round(38×0.7) ✓；
- b_710（RPT B2 的噪底参考，pyradius 恒 0）：改为谱#2 的快速包络——
  τ=0.01（稳态）/0.005（state2）一阶 IIR（`TimeToIirA(τ, sr/hop)`），
  逐通道在 ev_acs 内更新。这是 DLL 0x18015c8f0/0x18015d289 两处 IIR 的
  对位（第二处 1452 谱长遍历）；
- RPT B3 三元组 = `rms_around(inpos−hop/inpos/inpos+hop, hop)`；
- 未移植（记录在案）：DataTail(renderer+2248) 接受事件时清零（我们无对
  应缓冲，接受事件时输出不额外清段）；0x18015b69a 峰区 ×10 增强抑制
  （我们的 find_peaks/AFC 链无该增强路径）；检测器精确窗/频段/IIR 表。
- 纯变调（tempo=100、pitch>0）在 AU 中 slope>1 同样触发本子系统——这解
  释了我们/ pyradius 与 AU 采样级 corr 只有 0.03–0.05 的部分成因。
- 实测（AU 参考语料 29 s，`tests/au_transient_ab.rs`，10 ms 包络、
  ±30 bin 最优滞后）：off 轮廓 corr 0.6494/attack 窗 0.4903；仅门控
  （默认）0.6435/0.5024（无害且略优）；门控+重定时 0.5154/0.4191
  （attack 上升比 1.069 达标但轮廓受损）——故默认仅门控。

#### 2026-11-04 审计轮（全链路 TD+VC 校对，双 subagent 逐条对位）

审计揪出并当轮修复的 5 个真缺陷（修复后默认轮 corr
**0.6577/0.6332/0.6363**，tempo100/50 双超 off 基线，历史最优）：

1. **门控读数 0/1 vs 判 `==2`**（vocoder.rs：state2 bool 写入、六处门
   全判 ==2）——六门自上线以来一直惰性，"gates-only"测量实为惰性门。
   修复：写 RAW state 0..3。修复后 B2 路径立即现形劣化
   （0.48/0.41/0.30）。
2. **query_max 零分接受**（D34）：窗口内全零分时返回首 chunk 位置而非
   −1，平稳段向压缩侧持续供假事件。修复：best≤0 → −1。
3. **NMS 只对 s>1.0 应用**（D30）：DLL 对全部分数抑制。修复：去阈值。
4. **TimeToIirA rate 基准两处取错粒度**（D50）：ACS 用 sr/step(≈264)、
   b_710 用 sr/3592，DLL 用 sr/机器 hop(≈704)——b_710 的 a≈0.999 近乎
   无平滑，正是 B2 劣化的根因之一。修复：两处改机器 hop（sched 关闭
   时保留旧基准，pyradius 金标路径不变）。
5. **RPT B3 三元组窗宽用调度 hop 3592**（D59）：DLL 用机器 hop≈704。
   修复：改机器 hop。

**B2 定案**：即便修好 rate 基准，我们的 b_710 仍是结构性猜测，B2 实测
三参考全面劣化——`ts_rpt_b2` 配置默认关（全程 B3），DLL 的 B2 分支留待
噪底参考反编译后再启。隔离实验：B3 强制+活门 = 0.6413/0.6393/0.6363。

**TD 侧**（双速率定律/三坑/残余全部落地确认）顺手修一处：排空计数
trunc → round（与文档五次实测定律及 VC 侧一致）。

**第二轮（同日，遗留清理）**：
- margin 7×128 → 覆盖最大前向查询窗（`max(sr×0.1, 8380)+896`，对齐到
  stride）——tempo200 轮廓 0.6332→0.6408、attack 0.4134→0.4273、state-2
  假触发 843→771；
- ratio==1 直拷旁路（§8 D17）：tempo100+pitch0 不再全程声码化；
- set_geometry 除数表补全（D12：q≥10 → {3,3,6}[min(q−10,2)]，q<10 →
  4<<q；f 同步 1.4/0.7）；
- state2 非到达粒 out_step 改机器 step=round(hop×slope)（D48，仅
  ts_hop_override 路径生效）；
- 0xab3（sub_1801525E0）：**已移植**（`830a68d`，假触发 −69~−93%）；
  B2 噪底：**已完整解码+移植**（§13.8 十一步谱门，`4d12df1`/`b97e0e9`），
  实测三基准劣化（0.60/0.54/0.52）→ ts_rpt_b2 按条款保持默认关；DLL 喂
  给门的谱对象与 7556 比率缓存门控语义仍存疑。
- 流式参数提交链（§2 槽1/3/4 pending/commit/基点重映射）：**移植中**
  （2026-11-05 凌晨，含 D46 机器斜率变更复位与调度锚点连续性）；
- sub_180153570 增益曲线/宿主事件流未移植（UI 侧，音频影响未证实）；
- **×10 峰区增强已完整解码**（2026-11-05 凌晨，L2566-2606）：条件
  `(score肩差 + 谱相干性) > 1 ∧ 高频bin(sr×bin > (a1+1448)×4000) ∧
  state≠2`，肩差=`(score−下界)×0.25`（score<6 封顶）、相干性=
  `(spec−0.4)/0.3`∈[0,1]（带 0.4-0.7）；动作=**噪声合成环逐 bin 权重
  v424→10**（环体 L2582-2606：2080 激励 += 量化(1872 幅度−2080 现值)，
  3992/4096 键流双缓冲）。**不移植的原因**：动作目标是我们未移植的噪声
  合成环（2080/3992/4096 机制），默认链音频无效果（与 +3568 同掩蔽）；
  假移植成检测器分数增益会失真——DLL 的 ×10 不进检测路径。待噪声环
  移植时按此定律直接接入；
- step_base=222 仅 48k 锚定，44.1k 未验证缩放律（TD 报告观察项，需
  44.1k 参考音频）；
- renderer+3568 记账（已解码 2026-11-04 二轮，未移植）：+3568 = 距上次
  B3 接受瞬态的帧计数（每粒 +=hop，L4294；重置点清零 L1068）。B3 接受
  （`cur>两侧 ∧ cur>1.2`，L3060）且 `+3568 > sr×0.003`（144 帧@48k）时：
  `start = int(n_fft×1500/sr + 0.5)`（=512@48k/16384；无 bin 基项——
  v600/v11/v13 是 ±0.5 舍入常数），把通道谱尾 `[start, nb_bins)` 从缓冲
  renderer+1872（+8×ch）整段拷入 renderer+2080（+8×ch）——噪声激励
  缓冲重置，节流 144 帧；随后 +3568=0（L3065-3087）。**已移植**
  （2026-11-04 三轮）：pyradius 硬编码 e08=0.0 使节流闸恒关（死代
  码），已接真计数器 ts_frames_since_transient（每机器粒 +=in_hop）；
  工作缓冲=noise_gain，参考源暂用 mag_copy（真参考=DLL 噪声合成环的
  +1872，待该环移植后重接，代码内 TODO 标注）。因默认链不跑噪声合成
  环，A/B 逐位不变（预置接线，130 单测绿）。**第二清零点已接**（2026-11-05
  凌晨，`e07af2c` 后续）：state2 内 2460 跨 2456 一半时机器发一次性
  `mid_hold_clear` 脉冲，vocoder 清同一计数器（§13.2 L527 定律）；A/B 仍
  逐位一致（尾重置受 144 帧闸与噪声环缺位双重掩盖）。**仍开放**：第二处
  1872→2080 拷贝点 0x18015c627（L3239）未解码；
- state2 非到达粒 out_step=输入 hop（DLL=round(hop×slope)，仅
  ts_hop_override 实验路径受影响）；
- set_geometry 除数表分支不完整（q<10 的 4<<q 与 q∈{10,11}→3；当前
  quality=37 恒走 6 不触发）；
- step_base=222 仅 48k 锚定，44.1k 未验证缩放律（TD 报告观察项）；
- nch>2 时 TD 音高/瞬态分析仅用前两通道和（与 C 参考一致，信息项）。

本节所有地址基于当前 IDB（RE 文档 §6–§8 同一构建）；pyradius 注释里的
`@0x16D354` 等地址属于另一构建，不可混用。

### 13.5 发射层架构（2026-11-03 深挖，sub_180149F80）

瞬态重定时的**真实机制**：不是扭曲 pos/writepos，而是**逐粒 ratio 调制**。

- `sub_180149F80`（0x117E 字节）每个通道一条发射循环（ch1 直取，ch2 经
  a2+3664 的 DataTail 环），共享 a2+480 的**事件队列**（0x30 字节节点，
  MemoryRecycler 回收）：节点 `[+32]` = 输入断点位置、`[+40]` = ratio
  （int 位型）。
- 循环体（L560-587 / L736-762）：
  `emit(engine+8, state+352, ch, hop=[a2+592]+[a2+600], ring_off=pos mod hop,
  out, out_off, n_out=(node_pos−acc)/ratio, ratio, ratio_d, 1)`；发射后
  `[a2+408] = 下一节点[+40]` 换 ratio，节点回收。
- 输入累计器（ch2 = [a2+792] double）按 `n_out×ratio` 前进；**调度表/环
  偏移格子完全不动**（ring_off = pos mod hop）——ratio 断点在格子内改变
  出步长。
- hop = [592]+[600] 两段和（疑稳态 hop + settle/瞬态增量，写入点待查）。
- 我们端口的等效实现路径（下一步）：机器计划窗 (in_hop H_j, out_step S_j)
  ⇒ 有效 ratio_j = S_j/H_j，作为**发射级 ratio 断点**注入（不动表格子、
  不动 pos），影响谱峰定位/相位推进/OAC 窗伸缩的 per-granule ratio 钩子。
  现有 `ts_hop_override` 的 pos 扭曲方案已被此发现证伪（corr 崩 0.10 的
  根因）。
- **ratio 调制版已实现并测量**（2026-11-03 深夜）：`ts_hop_override` 语义
  更正为"输出步长 × plan_ratio/slope"（输入格子/表格完全不动）。A/B：
  100% corr 0.097→**0.526**、200% 0.078→**0.448**（相对 pos 扭曲版 5×
  改善）——机制证实，但仍低于 off（0.65/0.64），残余劣化推断为 OAC
  COLA 失配：发射窗长固定而出步长不均。DLL 每次发射还同步改
  hop=[592]+[600]（半边未移植），补上它（writepos 间隔与窗长同步按
  ratio_j 缩放）是下一步。仍默认关。
- **持久偏移实验（已回退）**：把输出偏移做成跨子粒累计（对位 DLL 输入
  累计器 [a2+792] += n_out×ratio）更糟——100%/200%/50% = 0.36/0.23/
  0.00，200% 全局滞后 −290 ms。结论：DLL 的累计器是**双侧**的（输入
  位置与输出位置都由事件队列推进，appr/settle/replay 计划对着绝对
  in/out 位置自校正），我们只推输出侧必然无界漂移。正确的完整移植
  要让 cursor 也脱离调度表由机器驱动（"16 子粒=1 DLL 粒"的经验映射
  届时消解）——架构级改造，留待下一轮。当前最优仍是局部 ratio 调制
  （0.526/0.448，默认关）。
- x64dbg 脚本：`.tmp/x64dbg/`（call.py、dump_renderer.py、trace_emitter.py、
  trace_granules.py）；IDA 侧 `.tmp/ida/decomp_to_file.py`（xrefs_to 用
  `addrs=[...]`，decompile 用 `addr=...`）。

### 13.6 发射驱动 sub_18014DF50——双队列架构完全解码（2026-11-04 深夜）

**已移植（同日四轮）**：ts_hop_override 路径重写为双队列式**绝对重锚
追踪**——每计划窗开窗锚定 (ts_win_in0, ts_win_out0=ts_cum_out)，窗内
每子粒按进度 frac 追踪累计输出 `desired = win_out0 + round(in_hop×rate_j)
×frac`，advance=round(desired−cum)，输入格子过冲在窗边界自纠（两侧）。
默认**启用**。A/B（AU 三基准）：

| 基准 | off | 仅门控 | **双队列（新默认）** |
|---|---|---|---|
| tempo100 | 0.6446/0.4913 | 0.6407/0.5070 | **0.6935/0.5909** |
| tempo200 | 0.6442/0.4313 | 0.6424/0.4411 | 0.6398/0.4324（持平） |
| tempo50 | 0.6277/0.4787 | 0.6363/0.4932 | **0.6474/0.5012** |

旧单侧局部调制（0.53 崩溃）被本机制取代；tempo100 +0.049/attack
+0.10。副作用修正：TD 验收钉值 1_393_118/1_393_182 → 1_393_119/
1_393_183（排空 trunc→round 律，§11）。

**live 实锤**：当前会话的活跃发射路径是 14DF50 的调用点 0x18014E33A
（149F80 的双通道循环一次都没走——那是另一模式）。0x4d5 字节全解码
（`.tmp/ida/emit3.c`）：

**摄入队列**（engine+328/336，0x38 字节节点）：
```
while (队列非空 && node[+32] ≤ 可用帧) {
    新out_base = (node[+32] − in_base) × rate + out_base
    in_base = node[+32]; out_base = 新out_base        // 基准对重锚
    if (node[+40] == 0) rate = node[+48]              // 标志位0→换速率
    回收节点
}
```
节点 `[+32]`=输入位置、`[+40]`=换率标志、`[+48]`=新速率（double）。

**发射循环**（engine+656/664，0x30 字节节点）：
```
hop = [424]+[432]                                     // 发射 hop 对（engine 布局）
target = rint((node[+32] − in_acc)/rate + out_pos)    // 节点输入断点经当前 rate 映射
n = target − out_pos
emit(a1+8, [352], ch, hop, in_acc mod hop, out, off, n, rate, rate_d, 1)
in_acc += n × rate          // 输入累计器（[9360] double）——双侧!
out_pos = target            // [552]
rate = 下一节点[+40]        // [272] int 位型 ↔ double 双存
```

**要点**：
1. **双侧自洽**：输入位置按 `n×rate` 前进、输出位置按节点的输入断点经
   当前 rate 重锚——绝对位置钉死，无漂移。我失败的持久偏移版缺的正是
   这个节点级重锚。
2. 换率只发生在节点边界，粒内 rate 恒定（发射 n 帧、耗 n×rate 输入）。
3. 发射 hop 对在 engine 布局是 +424/+432（≠ renderer 的 592/600——
   两套对象，跳过之前"写入点找不到"的疑问）。
4. 我们的移植路径定案：维护 (in_acc, out_pos, rate) 三元组，机器计划
   窗边界即节点：rate_j = plan.out_step/plan.in_hop，n_j =
   rint((node_in − in_acc)/rate_j + out_pos) − out_pos，重锚两侧。
   稳态节点即 slope（平滑包络 3928/3936 见 §13.5 尾）。

### 13.7 队列生产端（静态解码进行中）

- 158D20 全函数反编译 `.tmp/ida/machine_pretty.c`（4323 行）。接受律与
  端口逐位复核吻合：L1446-1450 =
  `hold_target = max(1, int((best−inpos)×2×slope/hop))` ✓、L1468-1473 =
  settle_hop_inc rint 中点律 ✓。
- 节点只有一个分配点：L1491 `Allocate(0x30)`（零初始化）——稳态与瞬态
  共用填充路径；速率历史环 [3928..3936]（float，尾槽写当前速率 v173，
  τ 律 `expf(sr/hop1376×v858)` IIR 平滑 3920）；`sub_180160610`（327
  行）= 速率/调度数学；节点 [+32]/[+40]/[+48] 实际填充在 L1598 之后
  （下一目标）。
- 动态分析经验（2026-11-04）：
  1. **离线渲染在子进程**——主进程 DSP 全冷；
  2. 中途附加会破坏音频线程（DSP 永不执行 + UI 异常循环于固定地址）；
     必须在效果实例化前附加；
  3. MCP 禁 `script.execute`（erun 不可用）；周期性异常会饿死断点轮询；
     条件 0 断点吞不掉异常事件；健康会话下要**轮询等待**（running 时
     sleep 0.2s），不能连发 debug_run；
  4. 模块基址随效果会话重建漂移，脚本每次 `mod.base()` 重解析。

### 13.8 B2 真子系统——倒谱谐波抑制噪声门（2026-11-05 凌晨，解码完成，参考链已移植）

sub_180158D20 L3270-3766 完整钉死。**B2 根本不是相位重置分支**，而是一
个作用于分析谱重合成前的谱域噪声门，pyradius 的 b_710-vs-相位语义与其
无关。十一步定律（全部在 src/vocoder/spectral_gate.rs 头注释对位）：

| # | 定律 | 缓冲/地址 |
|---|------|----------|
| 1 | db[k]=Amp2DB(spec[k]) | 7672 |
| 2 | dB 谱正 FFT → 交错倒谱 | 7720（scratch 7592） |
| 3 | c[q]=re²+im²，IIR 平滑 | 7768，a=TimeToIirA(τ, sr/机器hop)，τ=0.005(state2)/0.01 |
| 4 | F0 检测：倒谱峰评分 argmax，频带 150–800 Hz，clamp | v624=(L=7608)·sr/n_fft，f0=v624/q |
| 5 | 2·q_f0 处 4 bin 锥形 ×(g,g,0.25,0.25)，其上倒谱全部清零（谐波梳去除） | 7720 |
| 6 | 逆 FFT → 去谐波 dB 包络 | 7624 |
| 7 | 原始 dB 的 8 趟双向 IIR（音高同步宽度 0.12·f0）+ 低段取分支6/交叉渐变混合 | 7672/7624 |
| 8 | 逐 bin 增益 g=DB2Amp(clamp(斜率差,−40,+20)×标度) | 7816，步长=*(7556) |
| 9 | 可选能量归一化 g×=√(Σs²/Σ(gs)²) | 7554 门 |
| 10 | 吸引子 work += (g−work)·a | 7864（逐通道） |
| 11 | **消费**：spec[k] ×= work[k]，k ≥ bin(0.8·f0) | 0x18015d3a2（L3717-3763），PolarToCart/IFFT 之前 |

**门控顺序定律（2026-11-05 二轮，L3256-3268 钉死）**：进入门前先整段
拷贝 `1872[ch]←1976[ch]`、`1664[ch]←1768[ch]`；打分（L2131+ 峰区扫描）
与 b_710 全部读**未门控**的 1768，门只乘**合成副本** 1664（消费点
0x18015d3a2 之前）。触发条件 `7552≠0 ∧ [7556]≠1.0（=有变调）∧
[7564]≠0`，每机器粒每通道一次。radius-rs 已按此重接（ev_assembly 的
polar→cart 边界、合成副本 scratch、ts_plan_gen/ts_gate_gen 代数计机器
粒率、ratio≠1 条件）；A/B 从 0.60/0.54/0.52 升至 0.6255/0.5678/0.5503
——顺序修正有效但未达标，默认仍关，剩余疑点=步骤常数或喂入谱细节。
**第二拷贝点 0x18015c627 已解码**（L3232-3246）：逐 bin **首次初始
化**——`2080[ch][bin]←1872[ch][bin]`（噪声激励以幅度副本起步）+
`2256[ch][bin]=1.0f` 单次标志（条件=标志≠v569(=1.0) 或窗条件）。

**移植状态（2026-11-05 完成）**：全部 11 步已实现并接入主链
（spectral_gate.rs `SpectralGate::apply`，ev_acs 在 b_710 更新后、
peaks/AFC/assembly 消费前应用；ts_rpt_b2 && transient_sched 门控，
**默认 false 待 AU A/B**）。标量补齐：live 探测 7552=1(使能)、
7554=1(能量归一化)、7556=pitch ratio（动态 2^(P/12)，兼步骤 8 步长与
步骤 5 宽度因子）、7560=1.0、7564=1.0、7568=40、7572=800、
7608(L)=4096；反编译 v591=0.75（锥形 g，L979 v848）、v861=4.0 Hz
（混合低界，L977）、ε=1e-12（L963）、稳态 τ v613=0.01（L957）；
f0 clamp 实为 **[150, 832]**（0x44500000=832，旧笔记 800 系四舍五入
误读）；步骤 7 混合边界 = [bin(4 Hz), max(bin(9 Hz), L)]，即去谐波
分支主导全带、音高同步平滑分支仅在 Nyquist 附近。FFT 布局修正：我们
的 (N+2) 打包半谱与 DLL 2N 交错（镜像清零）对实包络等价（其逆变换
弃虚部）。138 单测绿（新增：锥形/清零边界、双向平滑单调、增益
clamp[−40,+20] 与带下继承、能量归一化恒等、双通道吸引子隔离）。

#### 13.8.1 门输入源与消费语义（2026-11-05 深夜，活体验证）

B2 剩余劣化根因排查（A/B 0.63/0.57/0.55 vs 默认 0.69/0.64/0.65）：

- **1768 = 幅度谱**（内容实测全正、min 0.40、max 267、零交叉 0——绝非
  时域）；它同时是 0xab3 峰搜索（L2131）、噪声环 A 的 argmax（L2200）、
  能量统计（L3169）与门输入复制源（LABEL_768 L3268）的**共享输入**。
- **1872/1976 = 相位缓冲**（内容含负值，15E560 的 atan2f 暂存实锤）——
  修正 13.9 的"幅度源"标签；噪声环实为相位键流演化（对默认链无影响，
  ts_noise_loop=false）。
- **填充链**（hw 写断点 + 栈走查）：流式 feed 14DF50 → 158960 →
  158D20 体内（经 16MB CRT 向量模块的 memcpy，源=分析管线输出区）→
  每机器粒发布进 1768。写入 RIP=bigmod+0x98C0C9（vmovdqa 4× 展开）。
- **结构铁证**：DLL 检测器与门共用 1768 ⇒ 门的打分输入=AFC/一致性
  等价物**之前**的原始分析谱。
- **消费语义修正**（0x18015d3a2）：`spec[k] *= work[k]`——增益从原始
  副本 1664 算出，**乘进合成谱**（打分源与应用目标是两个缓冲）。我们
  此前把两者绑在同一 slice 且喂 AFC 后谱。
- **移植修正**（src/vocoder/spectral_gate.rs `apply_split` + vocoder.rs
  ev_assembly）：打分喂 `mag_copy`（ev_afc 前快照），增益原位乘到
  AFC 后的 `ch.mag`；新增单测 `split_scores_raw_and_gates_target`
  （150 lib 绿，13 套件 ok）。A/B 待父代理复测。
- **未移植子律**（如实记录）：L3167-3183 的比率反馈（Σ1664² vs Σ1768²
  + 25/23/1500/2000 常数、span[1392] 参与的 v583/v584 增益补偿）——
  常数语义未验证，按"先证后动"原则暂缓。

### 13.9 噪声合成激励环（2026-11-05 凌晨，解码+移植完成）
`sub_180158D20` 的逐通道逐机器粒激励演化（L2050-2660、L3232-3268）。
缓冲拓扑（全部钉死）：

| DLL | 内容 | 我们对应物 |
|-----|------|-----------|
| 2080 | 激励态（逐机器粒演化） | noise_gain[ch] / NoiseLoop.exc |
| 1872 | 激励的幅度源（每粒刷新） | ch.mag_copy |
| 1976 | 1872 的源（上一分析谱） | ch.mag |
| 2256 | 逐 bin 单次初始化标志 | flag |
| 2176 | 噪声输出谱（环 A 写入） | out（消费点在 158D20 外，未接） |
| 3992/4096 | 键流双缓冲（前/后槽） | key_cur/key_prev |
| 3192 | 逐带相位累加器 | band_phase（推进律为推断：每粒 += hop·2π/n_fft） |

粒内顺序（程序序）：环 A → 环 B → 环 D → 初始化 → 源刷新（LABEL_768）。
- **初始化**（L3232-3246）：`flag≠1 或 force（带噪声强度>0.75）→
  exc=src、flag=1`——激励以幅度副本起步；
- **环 A**（L2232-2261）：带内 argmax(合成谱)+抛物线插值 → 峰位 pk；
  逐 bin `v361=(pk+带相位)·(hop·2π/n_fft)·0.5`，
  `out[bin]=out_step·((wrap2π(src[bin]−mag[pk_bin])−v361)+v361)/hop+exc[pk_bin]`；
- **环 B**（L2440-2606）：bin 斜坡 `v376=−2·frac((hop−out_step+drift)/
  v173)·v173·π/n_fft`；`v413=bin·v376+src−exc`，wrap2π 后取绝对值
  （0x7FFFFFFF 位掩码）为闸值；权重律=键流单调提升（0.75/0.25，比
  (1,4)）+零键端点拉（0.2/0.15，π）+×10（§13.4）+基包络（未解码，
  取 1）；`w≥|v415|` 时 `exc += wrap2π(v447)`；键流逐 bin 无条件推进，
  **残差恰在闸关时入库**；
- **环 D**（L2627-2719）：`q=out_step/hop`，`g=√ramp(q)·[stretch>1]`，
  ramp 界 `lo=0.5+g·0.5/σ、hi=0.5+g·4/σ`（σ=+3560 未解码，取 1）；
  `exc[b] += wrap2π(wrap2π(src[b]−src[r]) + wrap2π(exc[r]−exc[b]))·
  ramp(score)`——注意 exc==src 时两 wrap 项相消（数学空操作），环 D
  只在激励偏离源时起作用；
- **+3568 重定向**：环启用后尾重置写穿 NoiseLoop.exc（真 2080 对象），
  键流/标志与激励态保持一致；参考源仍=mag_copy（1872 本就是
  mag_copy 的对应物，本轮把 1976=mag 钉死）。

**未解码/未移植**（如实）：稳态随机化环（L2792-2960：3808 历史环拉入，
环的填充点不在已读段内）；基权重包络（L2348-2406 的 powf 带表，
.rdata 常量未 dump）；逐 bin 分数/平坦度（DLL 逐 bin 数组，我们按机器
粒广播标量）；v173/v894（时基归一化，取 1）。

**移植状态**：src/vocoder/noise_loop.rs 全环实现（环 A/B/D+初始化+键流
双缓冲+×10），7 新单测（初始化单次性+force 重置、wrap2pi 律、键流双缓
冲推进与入库律、×10 门控、环 D 分数拉引、输出谱确定性+相位旋转、多通
道隔离）；接线=新配置 `ts_noise_loop`（默认 false），机器粒步进点逐通
道调用，激励镜像回 noise_gain。149 lib 测试绿。A/B 留给父代理：建议
先单独开 ts_noise_loop（默认链渲染应改变——noise_gain 现在是活态），
再与 ts_rpt_b2 组合（B2 门读的 1664 合成副本与激励态相互作用）。

### 13.10 消费点搜寻闭环——2176/2080 无读者，噪声环系残留机制（2026-11-05）

穷尽式静态扫（全库字节模式 `48/4C 8B/8D ?? 80/20-08/50-07 00 00`，
即对 [reg+0x880]/[reg+0x820]/[reg+0x750] 的全部寻址形态）+ 偏移表
（a1+3120..3264 槽，0xC30..0xCC0 载入点全数定位）：

| 缓冲 | 全库触碰点 | 结论 |
|---|---|---|
| **2176 输出谱** | 158D20 自写、1563C0 **析构回收**、15E560 **按地址传分配器**（初始化） | **无内容读者** |
| **2080 激励** | 158D20 体内环 A/B/D/初始化/+3568；PitchCorrectDSP 系函数命中系其自身布局同偏移巧合 | 无外部读者 |
| **1872 幅度源** | 158D20 体内 + **15E560 两处**：`1872[ch][k]=atan2f(im,re)`（复用为相位暂存）、整段拷入 **3808 历史环**（= §13.9 未解码填充点的真身 0x18015f9fd/0x18015fa40） | 15E560 复用/归档 |

**定论**：噪声合成环的输出（2176）与激励态（2080）在此构建中**从不
混入音频重合成**——环是自包含的残留机制（写而不读）。这解释并**证
明**了 ts_noise_loop A/B 逐位一致是正确对齐而非缺线：DLL 侧同样无
音频效果。radius-rs 保留该模块作保真镜像（默认 false），**不向
ev_assembly 添加任何混入**（那反而偏离 DLL）。3808 历史环由 15E560
从 1872 填充、被 158D20 L2792-2960 拉入稳态随机化——是 1872 的下游
而非噪声环输出的下游。
