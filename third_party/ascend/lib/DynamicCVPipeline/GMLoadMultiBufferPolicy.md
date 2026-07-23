# GM-load 多缓冲预算分配方案（GMLoadMultiBufferPolicyPass）

## 1. 背景与目标

在 `DynamicCVPipeline` 中，核间（inter-core）buffer 与从 GM load 搬入的数据都要分配多缓冲（multi-buffer）来重叠传输与计算。此前每类缓冲的份数是**全局单值**（`BufferCountManager` 的 `IntraCore=2 / InterCore=1 / LoadStore=1`，由用户手动透传），一刀切、与 UB/L1 容量、计算块图、实际流水完全脱钩。

本 pass 让 **GM load 的多缓冲深度按 per-load 决策**：在 UB / L1 字节预算约束下，结合计算块流水结构决定每个 GM load 该 single 还是 double。inter-core 保持封顶 2，仅作为预算的固定预扣项。

**范围**：只处理核间 buffer（作预扣）与 GM load buffer（可变深度），不动核内（IntraCore）缓冲。

**一句话结论**：深度是一个 `{single, double}` 的二元选择。吞吐随计算模式变、块的运行时间不可信，故**抛开时间估算**，纯用拓扑结构给每个 load 一个**相对紧急度**（§2.4），再由预算背包（§2.5）按紧急度从高到低在 UB/L1 内依次分配 double。

---

## 2. 设计

### 2.1 内存分层与双池

不同缓冲落在不同物理内存，不能放进同一预算池：

| buffer 类型 | 物理位置 | 归属预算池 |
|---|---|---|
| Vector 的 GM load | UB | UB 池 |
| Cube 的 GM load | L1 | L1 池 |
| inter-core C→V（L0C→UB） | 落地 UB | UB 池 |
| inter-core V→C（UB→L1） | sender 在 UB / receiver 在 L1 | UB 池 + L1 池 |

**分池依据**：
- **inter-core**：其 transfer alloc 由 `SplitDataflow` 用带 `hivm::AddressSpaceAttr` 的类型创建，直接读 memref 的 memory space（`UB` / `L1`）即可归池，无需理解方向语义。
- **GM load**：backing alloc 是裸 `memref<...>`，**不带** address space，因此回退用「首个消费计算块的 core type」判定（`CUBE_ONLY→L1`，其余→UB）。

### 2.2 预算模型（朴素求和）

两个独立同构的背包池：

```
budget_UB = 248KB * ub_budget%     (可调, 可超 100%)
budget_L1 = 512KB                  (恒为满额, 不可调)
约束：  Σ depth(e) * size(e) ≤ budget_pool
```

- **UB 预算可调、可超 100%**——UB 后续会被 `PlanMemory` 复用，故此处只做「未复用前」的朴素求和、允许超额，靠复用兜底。
- **L1 预算恒为满额、不设选项**——L1 **无**下游复用，朴素求和即精确占用，直接填满物理 L1。
- **`ub_budget` 即整个 per-load policy 的开关**：设了才启用（同时驱动 UB 与 L1 两池）；不设则保持旧全局 `LoadStore` 行为。
- **inter-core 固定预扣**：每条 inter-core 边按「实际 UB 多 buffer 数量」`n∈{1,2}`（当前取全局 `InterCore` 值）预扣 `n×size` 到对应池。

### 2.3 深度是二元 {single, double}，用相对紧急度而非绝对时间

深度只有两档，`kDoubleBuffer = 2`。**要不要 double 是一个相对紧急度问题，而非绝对判定**——硬件吞吐随计算模式变，`block` 的运行**时间**不可信，任何"除以吞吐"得到的绝对量（节拍、窗口）都建在流沙上。因此本方案**抛开一切时间估算**，只用两类**确定量**：块依赖图的**拓扑结构** + 每个块/load 的**静态元素数、字节数**。

不去判"哪些一定要 / 一定不要"，而是给每个 load 一个**纯拓扑的相对紧急度**（§2.4），预算按紧急度从高到低**依次分配** double（§2.5）。

### 2.4 纯拓扑紧急度：暴露度

物理直觉：一个 load 的 double 越紧急，当且仅当它的消费块越**暴露**——即消费块前面能垫着掩盖它搬运的**同核**工作越少。掩盖靠"发射顺序"保证：`C_i` 发射前它的同核前驱 `P_i` 必在算（同核串行），这段时间搬运（另一条 pipe）可并行；single buffer 下下一批搬运正好插在这里。

**掩盖量（纯拓扑，吞吐无关，`sameCoreMaskWork`）**：

```
maskWork(C) = Σ_{P ∈ C 的同核拓扑祖先}  elems(P)
```

- 只累加 **同核** 祖先的 **元素数**（`Σ results 的 numElements`），**不除吞吐**；
- 祖先 = 块 DAG（同迭代 SSA cross-block 边，loop-carried 边无 defining op 天然排除）上 `C` 可达的前驱，且 `core == core(C)`；
- 对方核的块不算（在另一条 pipe，是否落进窗口不确定）。

**紧急度**：

```
urgency(L) = size(L) / maskWork(C_L)
```

搬得越多（`size` 大）、能垫的越少（`maskWork` 小）→ 越紧急。`maskWork = 0`（消费块是源头、无同核前驱）→ `urgency = ∞`，最紧急（完全没东西掩盖）。

**为什么这个比值的排序吞吐无关**（关键论证）：双池天然把消费块按核分开——UB 池全 vector 消费、L1 池全 cube 消费。**同一池内** `size` 都是字节、`maskWork` 都是同核元素数；两个未知吞吐（DMA、该核）对全池是**公共常数因子**。真实紧急度 `= (size/dmaThr)/(maskWork/coreThr) = (size/maskWork)·(coreThr/dmaThr)`，那个未知因子对全池相同，**不改变排序**。故按 `size/maskWork` 排出的**相对顺序与真实紧急度完全一致**——未知吞吐只影响"绝对阈值"（而我们正好不做绝对判定）。

> 例（lit Case 1，UB 池两个 `64×64 f32` load）：喂源头块 `V0` 的 → `maskWork=0` → `urgency=∞`；喂 `V3`（前有 `V1→V2`，`512×512`）的 → `maskWork = 512·512·2 = 524288` → `urgency ≈ 0.031`。前者远比后者紧急。

### 2.5 背包（按紧急度依次分配）

`urgency` 定完后，按池做预算分配：所有 load 先置 `depth=1`（single 基线，占 `1×size`）；按 **urgency 降序**排，从最紧急的开始给 double（`depth→2`，占 `+size`），预算够就给、不够就跳过（让后面更小的仍有机会），直到遍历完。

> 这直接实现「哪些更关键先分配」：预算充足时全部 double（掩盖的也不牺牲）；预算紧张时**暴露的（高 urgency）优先拿 double，被掩盖的（低 urgency）先降 single**。

### 2.6 输出契约

给每个多缓冲的 GM load 的 backing alloc 附一个 `annotation.mark`，带静态属性 `hivm.multi_buffer = <depth> : i32`（与 costmodel、compile_hint 三方共用同一 key，读写经共享的 `gmload::getMultiBufferOverride` / `gmload::findAllocMark`）。
- `depth == 1` 不打 mark（等价默认单缓冲）。
- **用户 hint 是最终方案上的纯覆盖，不是决策输入**：算法阶段对所有 load（含被 hint 的）一视同仁地算 urgency + 背包；被 hint 的 load 的输出保留其原 `hivm.multi_buffer`（不重写）。关键在于 hint 的**数字**既不进 urgency、也不改变它占用的预算份额（占用由算法 double 决定）——因此**改一个 hint 只重写该 load 自己那一个点，绝不连锁改变其他 load**。（曾否决「hint 按用户值预扣预算」，那会让一个 hint 挤占同池他人份额、造成与预期不符的连锁变化。）

### 2.7 Pass 定位

插在 `SeparateMemoryFromCompute` umbrella 内：

```
AsyncLoadHoisting  →  GMLoadMultiBufferPolicy(本 pass)  →  AddMultiBufferToGMLoad
```

此时计算块图（`ssbuffer.block_id` / `ssbuffer.core_type`）已稳定、inter-core transfer alloc 已存在（可估占用），而下游展开尚未发生。

---

## 3. 算法总览（伪码）

```
runOnOperation(module):
  ub% = readBudget(module.ssbuffer.ub_budget, default 100)
  budgetUB = 248KB * ub% ;  budgetL1 = 512KB
  interN  = clamp(global InterCore, 1, 2)
  graph   = buildBlockGraph(module)                 # 每块 elems + core + preds（纯拓扑）

  # ① inter-core 固定预扣（按 address space 分池）
  for alloc with ssbuffer.transfer_id:
      pool = poolOf(alloc.addressSpace)
      used[pool] += interN * bytes(alloc)

  # ② 枚举 GM load，算纯拓扑紧急度
  for ml in collectMarkedOps(module):               # gm_load_bufferable
      C = firstConsumer(ml)                          # 消费块
      e.size, e.pool(addrspace→回退 consumer core), e.pinned
      e.urgency = e.size / sameCoreMaskWork(C, graph)   # §2.4（纯拓扑）

  # ③ 双池预算背包：按 urgency 降序，从最紧急的开始给 double
  knapsack(UB loads, budgetUB, used[UB])            # 见 §2.5
  knapsack(L1 loads, budgetL1, used[L1])

  # ④ 输出
  for e: if not e.pinned and e.depth>1: mark(e.alloc, hivm.multi_buffer=e.depth)
```

---

## 4. 实现改动清单

| 文件 | 改动 |
|---|---|
| `include/DynamicCVPipeline/Common/Utils.h` | 新增属性 key `kUbBudget`(`ssbuffer.ub_budget`)、`kMultiBuffer`(`hivm.multi_buffer`) |
| `include/DynamicCVPipeline/Common/MultiBufferOverride.h` + `lib/.../Common/MultiBufferOverride.cpp` | **新增** 共享的 `gmload::getMultiBufferOverride` / `gmload::findAllocMark`（读取/定位 alloc 或其 mark 上的 `hivm.multi_buffer`），供 policy、`AddMultiBufferToGMLoad`、umbrella 三方复用 |
| `include/DynamicCVPipeline/SeparateMemoryFromCompute/GMLoadMultiBufferPolicyPass.h` | **新增** pass 声明（`getDependentDialects` / `createGMLoadMultiBufferPolicyPass` / `registerGMLoadMultiBufferPolicyPass`） |
| `lib/DynamicCVPipeline/SeparateMemoryFromCompute/GMLoadMultiBufferPolicy.cpp` | **新增** 策略 pass（分池 / 纯拓扑紧急度 `size/sameCoreMaskWork` / 按 urgency 降序的双池预算背包 / 输出 mark） |
| `lib/DynamicCVPipeline/SeparateMemoryFromComputePass.cpp` | umbrella 中插入 policy pass；放宽 `depth<=1` 门槛为「存在 per-load 请求（budget 属性或 hint mark）也继续」 |
| `lib/DynamicCVPipeline/SeparateMemoryFromCompute/AddMultiBufferToGMLoad.cpp` | `group.depth` 改为优先读 per-load `hivm.multi_buffer`（`resolveGroupDepth` 经 `gmload::getMultiBufferOverride`），回退全局 `LoadStore`；trip-count 剪枝改用 per-context 最大 depth |
| `triton_ascend.cc` | 新增 `set_ub_budget` binding，写 module 属性 |
| `backend/compiler.py` | `NPUOptions` 新增 `ub_budget`；`ttir_to_linalg` 中透传到 module 属性 |
| `bin/RegisterTritonDialects.h` | 注册 policy pass，使 `triton-opt --gm-load-multi-buffer-policy` 可用 |

---

## 5. 数据流

```
用户 (compiler options / al.multibuffer)
   │  ub_budget                                     │ compile_hint("hivm.multi_buffer", N)
   ▼                                                 ▼
compiler.py 透传 → set_ub_budget                  annotation.mark {hivm.multi_buffer=N}
   │  module attr ssbuffer.ub_budget                 │ (backing alloc 上)
   ▼                                                 │
GMLoadMultiBufferPolicyPass ── 分池/预扣/纯拓扑urgency/按紧急度背包 ──┘（hint 最终覆盖）
   │  在 backing alloc 打 annotation.mark {hivm.multi_buffer=depth}
   ▼
AddMultiBufferToGMLoad ── group.depth = per-load 值（回退全局）── 展开多缓冲循环
```

---

## 6. 测试与验证

### 6.1 Lit 回归测试

`third_party/ascend/unittest/Conversion/General/DynamicCVPipeline/SeparateMemoryFromCompute/gm_load_multi_buffer_policy_test.mlir`，三个用例：

- **Case 1 紧急度排序 + 预算紧张**：同池两 load——喂源头块 `V0`（`maskWork=0` → 高 urgency）与喂 `V3`（前有 `V1→V2` → 大 `maskWork` → 低 urgency）；`ub_budget=20%` 只够 double 一个 → 暴露的（`V0`）拿 double（mark），被掩盖的（`V3`）降 single（无 mark）。
- **Case 2 预算充足 → 都 double**：`ub_budget=100%`，低 urgency 的被掩盖 load 不被牺牲——urgency 只定顺序、不定资格。
- **Case 3 用户 hint 遵循**：预置 `hivm.multi_buffer=3` 被原样保留。

### 6.2 验证矩阵（已全部通过）

| 验证点 | 结论 |
|---|---|
| 分池 | inter-core 走 address space；GM load 走 consumer core-type 回退（cube→L1、vector→UB） |
| inter-core 固定预扣 | 按 address space 预扣 `n×size` 到对应池 |
| 双池独立 | UB 耗尽时 L1 的 load 不受影响 |
| 纯拓扑紧急度 | `urgency=size/sameCoreMaskWork`；源头块消费(`maskWork=0`)→ ∞ 最紧急 |
| 按紧急度依次分配 | 预算紧张：暴露的先 double、被掩盖的降 single；预算充足：全部 double |
| 用户 hint 遵循 / 不扰动他人 | 预置值原样保留；hint 数字不进预算，不改别的 load |
| mark 契约 | `annotation.mark %alloc {hivm.multi_buffer=N}` |
| 下游展开 | 不同 depth 的 group 各自正确展开，无 error |

---

## 7. 复现命令

### 7.1 增量编译（policy pass + opt 工具）

```bash
cd python/build/cmake.macosx-11.0-arm64-cpython-3.12
ninja bin/triton-opt          # 编译 DynamicCVPipeline 库 + triton-opt
```

### 7.2 跑 lit 回归测试

```bash
OPT=python/build/cmake.macosx-11.0-arm64-cpython-3.12/bin/triton-opt
FC=<llvm-install>/bin/FileCheck
T=third_party/ascend/unittest/Conversion/General/DynamicCVPipeline/SeparateMemoryFromCompute/gm_load_multi_buffer_policy_test.mlir
$OPT --gm-load-multi-buffer-policy --split-input-file $T | $FC $T && echo PASS
```

### 7.3 单独观察决策（debug 输出）

```bash
$OPT --gm-load-multi-buffer-policy -debug-only=gm-load-multi-buffer-policy \
     $T --split-input-file 2>&1 | grep '\[gm-load'
# 例（Case 1，同池两 load，预算只够一个 double）:
#  [gm-load-multi-buffer-policy] load: pool=UB size=16384 maskWork=524288 urgency=0.031 target=2  # 被掩盖→降 single
#  [gm-load-multi-buffer-policy] load: pool=UB size=16384 maskWork=0 urgency=1.6e10 target=2       # 暴露→double
#  [gm-load-multi-buffer-policy] assigned depth=2 to load size=16384
```

### 7.4 串联 policy + 下游展开

```bash
$OPT --gm-load-multi-buffer-policy --gm-load-multi-buffer <input>.mlir
```

### 7.5 全量构建（接入 Python 编译流程）

```bash
bash build.sh    # 重建 _C.so
```
之后编译 kernel 时通过 `NPUOptions` 传 `ub_budget`（即启用 per-load policy；L1 自动填满），或在 kernel 里用 `al.multibuffer(tensor, N)` 显式指定某个 load。

---

## 8. 遗留项与固有边界

**待校准 / 精化（非阻塞）**：
1. **inter-core 份数 `n` 用全局 `InterCore` 值**，未复算 `AddMultiBufferOuterScope` 的 flag 预算降级（`maxFlagId+groupCount≥15→single`）；被降级的边应按 1 份预扣。
2. **cube 块的 `elems` 用 `M·N` 近似**（忽略 reduction 维 K）。这只影响 cube 池（L1）内 `maskWork` 的相对大小，不影响 UB 池；如需更准可接真实 `M·N·K`。
3. **`maskWork` 只算同核祖先**：跨核块在另一条 pipe，是否落进 buffer 空窗不确定，故不计（保守，倾向判"更紧急"）。

**为什么这版是"纯拓扑"的**：整条链路只用**块拓扑**（`preds` / 同核祖先可达）与**静态形状**（`elems` / `size`），**不含任何吞吐常数、不算任何时间量**。紧急度是**相对排序**，双池内未知吞吐是公共因子、约掉不改变序（§2.4 论证）。

**刻意的取舍**：本版**不做**绝对的 "single/double 判定"（那需要吞吐→时间，不可信），只做**相对紧急度 + 预算依次分配**。因此："预算充足时所有候选都 double"是设计使然（不是 bug）——把"到底哪些真需要 double"的绝对判断，交给预算总量（用户经 `ub_budget` 控制稀缺度）与 hint（§2.6）显式兜底。深度 >2 亦同理，仅由 hint 提供。
