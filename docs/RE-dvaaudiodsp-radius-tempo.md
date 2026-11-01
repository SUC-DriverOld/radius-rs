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

## 7. 斜率重算 sub_180157BC0（= hop 几何选择）

```
slope = a2 (传入 = ramp 求值的当前目标)
quality = renderer[1336]
if quality >= 10:  hop = round(sr[1448] × 14000 / sr_hw[96])  (clamp by [1452])
else if slope < 1.0: hop = round(sr[1384] × 0.7 / (4 << quality))
else:               hop = round(sr[1384] × 0.7 / {1,3,6}[min(quality,2)])  (quality<10 分档)
```
（pyradius 的 `set_ratio(semis, quality)` 的 quality→hop 表与此对应；
具体表值待进一步核对。）

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

