# MedSeg — 医学 CT 图像交互式分割平台

> 基于 MedSAM2 的医学 CT 影像交互式分割 Web 平台，支持点击提示、框选提示、掩码传播等多种分割方式，面向 CT 肺结节检测等临床影像标注场景。

---

## ✨ 功能特性

- 🖥️ **Web 端 CT 浏览器** — 支持 `.mhd/.raw` 及 `.nii/.nii.gz` 格式，三平面（轴/冠/矢）交互查看
- 🤖 **AI 交互式分割** — 接入 MedSAM2 GPU 推理服务，支持点击提示、边界框提示、掩码传播
- 👤 **用户系统** — 注册、登录、个人资料、密码重置，基于 Flask-Login + Werkzeug 密码哈希
- 📁 **项目管理** — 多项目、多数据集管理，支持 chunked 大文件分片上传
- 🔌 **服务解耦** — 前端 Flask 应用（端口 8080）与 MedSAM2 FastAPI 推理服务（端口 7001）独立部署
- 🖧 **局域网 / 云端部署** — 支持本地单机、局域网共享及云服务器公网访问

---

## 🗂️ 项目结构

```
MedSeg/
├── run.py                      # 主入口，启动 Flask 前端服务
├── start_medsam2_service.sh    # 启动 MedSAM2 GPU 推理服务的 Shell 脚本
├── requirements.txt            # Python 依赖
├── medsam2_service_app/        # MedSAM2 FastAPI 推理服务
│   ├── app.py                  # FastAPI 应用，/segment /tasks /health 接口
│   └── predictor.py            # MedSAM2 模型加载与推理逻辑
└── medseg/                     # Flask 前端主应用
    ├── config.py               # 全局配置（读取环境变量）
    ├── create_app.py           # Flask app 工厂
    ├── models/
    │   ├── user.py             # 用户模型（SQLite）
    │   └── project.py          # 项目/数据集模型（SQLite）
    ├── routes/
    │   ├── auth.py             # 登录/注册/修改密码路由
    │   ├── dashboard.py        # 项目列表、数据集管理路由
    │   └── ct_viewer.py        # CT 查看器、分割请求路由
    ├── utils/
    │   ├── ct_utils.py         # CT 读取、窗宽窗位、切片转换
    │   └── file_utils.py       # 文件管理工具
    ├── templates/              # Jinja2 HTML 模板
    └── static/                 # CSS / JS / 图片资源
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.4（GPU 推理服务需要）
- [MedSAM2](https://github.com/bowang-lab/MedSAM) 已克隆并配置模型权重

### 1. 创建并激活环境

```bash
conda create -n medseg python=3.10 -y
conda activate medseg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. 配置环境变量（可选）

所有配置均通过环境变量控制，以下为默认值：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MEDSEG_DATA_DIR` | `/opt/medseg_data` | 数据存储根目录 |
| `MEDSEG_PORT` | `8080` | 前端服务端口 |
| `MEDSEG_SECRET_KEY` | `MEDSEG_SECRET` | Flask Session 密钥（**生产环境请修改**） |
| `MEDSAM2_REPO` | `/home/mdc/MedSAM2` | MedSAM2 仓库路径 |
| `MEDSAM2_CHECKPOINT` | 自动检测 `.pt` | 模型权重文件路径 |
| `MEDSAM2_DEVICE` | `cuda:0` | 推理设备 |

### 3. 启动 MedSAM2 推理服务

```bash
# 方式一：使用启动脚本
bash start_medsam2_service.sh

# 方式二：手动启动
export MEDSAM2_REPO=/path/to/MedSAM2
python -m uvicorn medsam2_service_app.app:app --host 127.0.0.1 --port 7001
```

### 4. 启动前端应用

```bash
python run.py
```

启动后访问：
- 本地：http://127.0.0.1:8080
- 局域网：http://\<本机IP\>:8080

---

## 🔌 MedSAM2 推理服务 API

推理服务运行在 `http://127.0.0.1:7001`，提供以下接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 服务健康检查 |
| `POST` | `/segment` | 提交分割任务，返回 `task_id` |
| `GET` | `/tasks/{task_id}` | 查询任务状态与进度 |

分割请求示例：

```json
POST /segment
{
  "volume_path": "/opt/medseg_data/case001.mhd",
  "output_path": "/opt/medseg_data/case001_mask.nii.gz",
  "boxes": [[120, 80, 160, 130]],
  "ww": 1500,
  "wl": -600
}
```

---

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| 前端框架 | Flask 3.1 + Jinja2 |
| 生产服务器 | Waitress |
| AI 推理服务 | FastAPI + Uvicorn |
| AI 模型 | MedSAM2 (SAM2-based) |
| CT 图像处理 | SimpleITK + NumPy + scikit-image |
| 数据库 | SQLite |
| 用户认证 | Flask-Login + Werkzeug |
| 可视化 | Plotly.js + Chart.js |

---

## 📋 支持的数据格式

| 格式 | 说明 |
|------|------|
| `.mhd` + `.raw` | MetaImage 格式（CT 常用） |
| `.nii` / `.nii.gz` | NIfTI 格式 |

---

## ⚙️ 生产部署建议

1. **设置强 Secret Key**
   ```bash
   export MEDSEG_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
   ```

2. **指定数据目录**
   ```bash
   export MEDSEG_DATA_DIR=/data/medseg
   ```

3. **禁用自动打开浏览器**（服务器环境）
   ```bash
   export MEDSEG_AUTO_OPEN=false
   ```

4. **反向代理**（推荐 Nginx 转发 8080 端口）

---

## 📄 License

本项目仅用于学术研究目的。MedSAM2 模型权重的使用请遵循其原始许可证。
