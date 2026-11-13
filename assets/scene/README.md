# 场景资产

来源：robosuite **v1.5.1**（`__version__ = "1.5.1"`），commit `a071383d53568ab798eb315c0e95357911be922d`。
仓库只拷贝了 PickPlaceCan 用到的文件，运行时不引用 `third_party/robosuite`。
许可证是 robosuite 的 MIT，见同目录 `LICENSE`。

PickPlaceCan 的 arena 是 `BinsArena`，没有单独的桌子模型。桌子来自同版本的 `table_arena.xml`，用来把 ALOHA 底座托在桌面高度。

## 文件

| 文件 | robosuite 路径 | 改动 |
|---|---|---|
| `LICENSE` | 仓库根目录 `LICENSE` | 原文 |
| `textures/soda.png` | `robosuite/models/assets/textures/soda.png` | 原文，Can 贴图 |
| `textures/light-wood.png` | `robosuite/models/assets/textures/light-wood.png` | 原文，左 bin |
| `textures/dark-wood.png` | `robosuite/models/assets/textures/dark-wood.png` | 原文，右 bin |
| `textures/light-gray-floor-tile.png` | `robosuite/models/assets/textures/light-gray-floor-tile.png` | 原文，地面 |
| `textures/ceramic.png` | `robosuite/models/assets/textures/ceramic.png` | 原文，桌子 |
| `textures/steel-brushed.png` | `robosuite/models/assets/textures/steel-brushed.png` | 原文，桌腿 / bin 腿 |
| `meshes/can.msh` | `robosuite/models/assets/objects/meshes/can.msh` | 原文。缩放写在 XML 里，不改网格 |
| `pick_place_can.xml` | 由 `arenas/bins_arena.xml`、`arenas/table_arena.xml`、`objects/can.xml` 改写 | 见下 |

## 布置和缩放

坐标：机器人前方是 base 系 +X，左侧是 +Y。左 bin 取，右 bin 放。

- **地面**：`bins_arena.xml` 的地板，放到 z = -0.45。
- **桌子**：`table_arena.xml` 的陶瓷桌面和金属腿。桌面顶在 z = 0，与 `vx300s_1_base` 网格最低点对齐，所以底座放在桌面高度。尺寸改成盖住底座（半尺寸 0.22 × 0.16 × 0.02 m），腿接到地面。
- **Bin**：`bins_arena.xml` 的 bin1（浅木，无隔板）和 bin2（深木，十字隔板，四格）。线性尺寸统一乘 **0.36**。原地板全尺寸 0.40 × 0.50 m，缩放后 0.144 × 0.180 m。地板顶面在 z = 0。腿按这个高度接到 z = -0.45，半径取 0.008 m（按比例是 0.0036 m，加粗一点才看得见）。
- 左 bin 中心（base 系）：(0.30, 0.125, -0.0072)。右 bin 中心：(0.30, -0.125, -0.0072)。
- 放置目标格沿用 robosuite 里 Can（物体 id 3）的规则：右 bin 的 +X、+Y 那一格。格子中心相对 bin 中心是 (0.036, 0.045) m。
- **Can**：网格等比例 **0.90**。未缩放时包围盒约 5.00 cm × 8.01 cm，缩放后直径 **4.50 cm**、高 **7.21 cm**。外观仍是 `can.msh` + `soda.png`。碰撞改成圆柱（方案 B），半径 0.02251 m，半高 0.03604 m，中心与网格几何中心对齐。原因：方案 A 的凸网格在夹持力下会慢慢陷入，3 s 内罐子相对 TCP 滑了约 9 mm；换成圆柱后滑移约 2.4 mm。
- **agentview**：初值来自 `bins_arena.xml` 的 `agentview`（`pos="1 0 1.75"`，`quat="0.653 0.271 0.271 0.653"`）。场景缩小后改到 `pos="0.71 -0.18 0.416"`，看向两个 bin 中间偏上的 `(0.26, 0, 0.14)`，`fovy="48"`。相机 +X 指向画面右侧，+Y 指向上，视线沿 -Z。两个 bin 的外沿顶角落在归一化坐标 [-1, 1] 内。
- **TCP**：不在这个场景文件里。`aloha/aloha.xml` 的 `left/tcp` 是唯一的工具中心点，放在两侧指垫中点；坐标轴跟 `left/gripper_link` 一样，+X 是接近方向。
- 光照在 robosuite 那盏方向光之外加了顶灯、两侧补光，以及更亮的 headlight，让 96×96 能看清罐子。腕部相机仍是机器人模型里的 `wrist_cam_left`。
