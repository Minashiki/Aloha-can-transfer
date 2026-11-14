# M4 交付记录（r2）

按 `tasks/M4_review_r2.md` 最终结论第 1–7 项改完并重跑。环境仍是 `conda run -n AIControl`。增益、力矩上限、速度上限、成功阈值、`random_can` 和夹爪开合量没有改。没有改 `docs/`、`.gitignore`、`environment.yml`、`scripts/check_env.py`、`aloha/`、`assets/scene/`、M2 计算模块、`configs/m1_pick_place.yaml`、`configs/m3_track.yaml`。没有做 git 写操作。`.gitignore` 在任务开始前就有未提交改动，这次没有再动它。

## 改动文件

- `src/aloha_pp/control/quintic.py`：preview 本拍以本次目标为终点；等待模式留下的缓存结点直接丢弃，并置 `dropped_knot`。用过 `q_future` 之后，缺省的下一次调用按 `q_future = q_target` 做中心差分。`reset` 清掉这个状态。纯 `None` 流仍走原来的等待模式。
- `src/aloha_pp/control/tracking.py`：`set_target` 把上面的状态反映到 `reference_mode`。`StepSample` 增加 `dropped_knot`。
- `src/aloha_pp/planning/metrics.py`：掉落的“松开”改为两指都不接触；新增 `release_split`。
- `src/aloha_pp/planning/planner.py`：把实测后的放置路点记在 `place_tcp`，供分析使用。规划的时间律没有改。
- `configs/m4_task.yaml`：掉落注释改成与代码一致。阈值没变。
- `scripts/m4_run.py`：日志增加 `dropped_knot` 和 `place_tcp`。`--zero-arm-friction` 只在内存里把 6 个臂关节的 `dof_frictionloss` 设为 0，日志写到 `outputs/m4/ablation_nofriction/`。
- `scripts/m4_analyze.py`：释放拆分列、E2 分布、TCP 参考的脚注、按 tick 去重的 IK 耗时、摩擦消融对照。
- `scripts/m3_run.py`、`scripts/m3_analyze.py`：`--out`，缺省仍是 `outputs/m3`。
- `tests/test_m4_reference.py`、`tests/test_m4_task.py`。

## 命令

```text
conda run -n AIControl python -u scripts/m4_run.py
```

`section_s 122.24`，`total_s 127.04`。E1 六组、E2 五十个种子、E4 十二组都跑完，E1 和 E2 的 `success` 全是 True。`dropped_knot` 在这 249225 个物理步里是 0：这些实验从第一拍就带 `next_action`，没有走进新的切换分支。

```text
conda run -n AIControl python -u scripts/m4_run.py --zero-arm-friction
```

`total_s 4.43`。fixed_1 medium 和 fixed_2 medium 都成功。名义摩擦是 `[0, 2.0, 1.15, 0, 0, 0]`（肩 2.0、肘 1.15，其余为 0）。消融日志里这六个数都是 0。`aloha/` 没有改。

```text
conda run -n AIControl python -u scripts/m4_analyze.py
conda run -n AIControl python -u scripts/m3_run.py --out outputs/m4/m3_regression
conda run -n AIControl python -u scripts/m3_analyze.py --out outputs/m4/m3_regression
conda run -n AIControl python -m pytest tests -q --tb=line
conda run -n AIControl python scripts/check_env.py
conda run -n AIControl python scripts/m1_grasp_smoke.py
```

`m3_run.py` 的 `total_s` 是 28.28，输出都在 `outputs/m4/m3_regression/`。重跑前后 `outputs/m3/` 下 94 个文件的 mtime 和大小一致，这次没有写那个目录。汇总表与 `tasks/M3_delivery_r2.md` 的 40 行逐格相同，`cell_diffs 0`。

`63 passed in 3.47s`。`ALL CHECKS PASSED`。`GRASP_OK`（`hold_lift 0.0851`，`max_slip_m 0.002780`）。

E1 / E2 / E4 与 r1 的 `summary.csv` 在共有列上没有超过 1e-9 的差，`metric_diffs 0`。共有列包括成功标记、关节 RMSE、两种 TCP 误差、饱和、力矩峰值、放置误差、滑移和肩关节 `q̈_d` RMS。

## 1. F1 放置误差

r1 把几毫米主要写成松手后大约 6 mm 的下落。这个归因不成立。开环斜坡开始时，肩关节 `q−q_d` 已经在 −6.3 到 −6.7 mrad，肘在 −4.0 到 −4.6 mrad，罐子相对格子中心的 xy 已经有 3.2–4.7 mm。

`open_*` 是 `planner_state == OPEN` 的第一个物理步，也就是斜坡开始，手指这时还接触罐子。`detach_*` 是两指都接触过之后、第一次两指都不接触的物理步。TCP 分量是实测位置减去 `place_tcp`。`detach_can_xy_disp` 是从实际脱离到结束的 xy 位移向量的模。单位：长度 m，关节 rad。

| 组 | 最终 xy | open TCP xy / z | open Can xy / z | open 肩 / 肘 | detach Can xy / z | 脱离后 xy 位移 |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_1 fast | 0.003810 | 0.002008 / 0.003212 | 0.004188 / 0.009180 | −0.006321 / −0.003968 | 0.003926 / 0.008430 | 0.000138 |
| fixed_1 medium | 0.004633 | 0.002235 / 0.003501 | 0.003560 / 0.009622 | −0.006685 / −0.004544 | 0.003332 / 0.008855 | 0.001366 |
| fixed_1 slow | 0.005417 | 0.002196 / 0.003518 | 0.003185 / 0.009963 | −0.006596 / −0.004467 | 0.002992 / 0.009274 | 0.002468 |
| fixed_2 fast | 0.004043 | 0.002181 / 0.003222 | 0.004726 / 0.009326 | −0.006560 / −0.004316 | 0.004493 / 0.008561 | 0.000464 |
| fixed_2 medium | 0.004744 | 0.002304 / 0.003306 | 0.003829 / 0.009565 | −0.006748 / −0.004634 | 0.003674 / 0.008839 | 0.001173 |
| fixed_2 slow | 0.005834 | 0.002237 / 0.003255 | 0.003424 / 0.009942 | −0.006626 / −0.004487 | 0.003298 / 0.009220 | 0.002719 |

斜坡起点的 Can xy 与复核表一致（例如 fixed_1 medium 3.560 mm，肩 −6.685 mrad）。按实际脱离时刻，脱离之后的 xy 位移是 fast 0.14 / 0.46 mm、medium 1.37 / 1.17 mm、slow 2.47 / 2.72 mm。slow 不能再说成大约 1 mm。脱离瞬间罐子仍在格子中心上方 8.4–9.3 mm，后面还会落下；最终 xy 和三维误差几乎一样，竖直方向落到了格子高度。

E2 五十条，min / median / max：

| 量 | 最小 | 中位数 | 最大 |
| --- | --- | --- | --- |
| open TCP xy | 2.23 mm | 2.26 mm | 2.28 mm |
| open Can xy | 2.26 mm | 3.56 mm | 3.72 mm |
| open 肩误差 | −6.73 mrad | −6.72 mrad | −6.68 mrad |
| 脱离后 xy 位移 | 0.12 mm | 1.23 mm | 1.52 mm |
| 最终 xy | 2.38 mm | 4.50 mm | 4.77 mm |

摩擦消融只跑了两组 medium。肩、肘的 `dof_frictionloss` 被设成 0 之后：

| 组 | 最终 xy | open TCP xy | open Can xy | open 肩 | open 肘 | 脱离后 xy 位移 |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_1 medium，摩擦 0 | 0.483 mm | 0.051 mm | 1.070 mm | 0.15 mrad | 0.13 mrad | 0.96 mm |
| fixed_2 medium，摩擦 0 | 0.518 mm | 0.053 mm | 1.344 mm | 0.13 mrad | 0.13 mrad | 1.20 mm |

同一组名义摩擦下的最终 xy 是 4.63 mm 和 4.74 mm，肩误差约 −6.7 mrad。关掉这两项摩擦之后，肩、肘的稳态误差和大部分放置误差一起消失。这两组上，摩擦带产生了那几毫米里的大部分，不只是和误差同时出现。关掉以后仍剩大约 0.5 mm，脱离后的 xy 位移也还在，所以下落仍有一份，不能把每一毫米都写成摩擦。抓取时测到的 Can–TCP 偏移也会变，放置路点跟着变；`q−q_d` 的缩小不依赖这条重新瞄准。slow 和 E2 没有做消融。没有改增益，没有加积分，规划器也没有补偿这个误差。

## 2. F2 参考模式切换

有 `q_future` 时，本 tick 的终点是本次目标。缓存里还没播放的等待模式结点被丢掉：不播放，也不进入这次或以后的中心差分。`dropped_knot` 只在真的丢掉一个缓存结点时为真。差分用的上一结点是上次实际播放的目标。

流一旦用过 `q_future`，之后缺 `next_action` 就按 `next_action = action` 处理。终点速度是 `(target − prev) / (2T)`，终点加速度是 `(prev − target) / T²`，不是零。连续把同一个目标再发一次时，prev 已经等于 target，终点速度和加速度才是 0，参考停在目标上。`reset()` 把 `_seen_preview`、缓存和 `dropped_knot` 清掉，之后的纯 `None` 又是 M3 的等待模式。

超速仍走 M3 的制动并打速度超限标记。限幅段的终点是制动后的位置，测试只检查标记，不要求一拍精确到达。

测试在 `tests/test_m4_reference.py`：

- 前三拍 `None` 再转 preview：丢掉 0.06 那个缓存结点，0.08 在本拍末到达，差分用的是上次播放的 0.04 而不是被丢掉的 0.06。
- 匀速斜坡第 6 拍缺 future：整段 `q_d` 不减，`q̇_d` 不换向，该拍末仍是 0.12。
- 斜坡末尾连续三次缺 future：第一段停在 0.18，段内没有越过目标；下一段终点速度和加速度为 0，段内超调 0.001650139136 rad；再下一段整段停在 0.18。
- 预告 B、实际下发 C：终点是 C，速度用 A 和 D，不用 B。段起点等于上一段终点。
- `v_max = 0.2`、目标步进 1 rad：超限标记为真，终点不是 1 rad。
- 纯 `None` 与 `push(q)` 逐步一致；`reset` 之后再 `None` 回到缓存、保持。

## 3. N1 掉落判定

`lost` 改为两指都不接触：`~(left | right)`，并且处于搬运状态、罐心高于格子中心 2 cm、连续不少于 10 个物理步。低于抓取高度 `drop_z`（0.03 m）的分支保留。单指失接触 20 步不算掉落；两指都失接触 9 步不算，10 步算。配置注释与这段一致。这次 56 条任务日志仍然全部成功，分类结果没有变。

## 4. F3 口径

`outputs/m4/summary.md` 在主表前后写了：E1 和 E2 的 `tcp_ref_rmse` 比的是实测 TCP 和规划器时间律；动作在本拍终点取样，比这条时间律提前一拍，这已经抵消 preview 的 50 ms 滞后。比较滞后只用 E4。M5 如果用规划器的动作做示范，也带着这一拍提前量。`schedule.py` 的采样没有改。

## 5. F5 IK 耗时

E1 和 E2 合在一起，每个控制拍只取物理步日志里下标 0、25、50……的那一个值。当前目标：中位数 88.008 µs，最大 257.560 µs。下一目标：中位数 73.780 µs，最大 239.988 µs。两个最大值都远小于 50 ms。这是求解时间，不是整拍的规划或调度时间。复核时的 76.6 / 71.6 µs 是另一次运行的墙钟，跟踪指标这次没有变。

## 6. F4 与 M3 回归

`scripts/m3_run.py --out outputs/m4/m3_regression` 和 `scripts/m3_analyze.py --out outputs/m4/m3_regression`。不带 `--out` 时目录仍是 `outputs/m3`，这次没有用那条缺省路径。`outputs/m3/` 的文件清单在重跑前后相同。

## 7. 重跑核对

E1、E2、E4 的原有指标与 r1 的汇总表一致（`metric_diffs 0`）。这些运行没有丢掉任何结点。摩擦消融是额外的两组，不写进 `outputs/m4/runs/`。

## 存疑

- 消融只有 fixed_1 / fixed_2 的 medium 各一次。slow 档脱离后 2.5–2.7 mm 的 xy 位移没有在零摩擦下重测。
- 零摩擦也会改变闭合时测到的抓取偏移，放置路点因此不同。关节误差的缩小说明跟踪本身变了；最终那 0.5 mm 里还有下落和重新瞄准的份，消融没有把它们拆成两个独立实验。
- IK 耗时是墙钟。同一次二进制在不同负载下会和复核的微秒数差一截；两次都远小于一个控制周期。
- 段内超调 0.00165 rad 来自单元测试里的连续缺拍。任务日志里规划器每拍都给了 `next_action`，这条停止段没有出现在 E1 / E2 里。
