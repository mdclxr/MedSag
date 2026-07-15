import socket
import threading
import webbrowser
import waitress
from medseg import create_app
import multiprocessing
import psutil
import os

app = create_app()
FRONTEND_PORT = int(os.environ.get('MEDSEG_PORT', '8080'))


def get_port_owner_pid(port):
    for conn in psutil.net_connections(kind="inet"):
        if conn.status != psutil.CONN_LISTEN:
            continue
        if not conn.laddr or conn.laddr.port != port:
            continue
        return conn.pid
    return None


def ensure_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
        except OSError as exc:
            owner_pid = get_port_owner_pid(port)
            pid_message = f"占用进程 PID: {owner_pid}。" if owner_pid else "未能识别占用进程 PID。"
            raise SystemExit(
                f"错误: 端口 {port} 已被占用，{pid_message} 请先释放该端口或通过 MEDSEG_PORT 环境变量更换端口。"
            ) from exc


def main():
    port = FRONTEND_PORT
    host = os.environ.get('MEDSEG_HOST', '0.0.0.0')
    ensure_port_available(port)

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"

    display_host = os.environ.get('MEDSEG_DISPLAY_HOST')

    print(fr"""
==============================================
  __  __          _ ____      _    __  __ 
 |  \/  | ___  __| / ___|    / \  |  \/  |
 | \  / |/ _ \/ _` \___ \   / _ \ | |\/| |
 | |\/| |  __/ (_| |___) | / ___ \| |  | |
 |_|  |_|\___|\__,_|____/ /_/   \_\_|  |_|   医学分割平台
==============================================
版本: MedSeg v1.0.0""")

    if display_host:
        print(f"访问地址: http://{display_host}:{port}")
    else:
        print(f"本地访问: http://127.0.0.1:{port}")
        if local_ip != "127.0.0.1" and not local_ip.startswith("198.18."):
            print(f"局域网访问: http://{local_ip}:{port}")
        print(f"提示: 若部署在云服务器，请通过服务器的公网 IP 访问")
        
    print(fr"""==============================================
支持模型:
  - MedSAM2 GPU 推理服务 (端口 7001)
  - MedSAM3 GPU 推理服务 (端口 7002，预留)

数据目录: {os.environ.get('MEDSEG_DATA_DIR', '/opt/medseg_data')}
==============================================
""")

    # 仅在非 Headless (无图形界面) 系统且未被环境变量禁用时自动打开浏览器
    auto_open = os.environ.get('MEDSEG_AUTO_OPEN', 'true').lower() == 'true'
    is_headless = os.name != 'nt' and not os.environ.get('DISPLAY')
    if auto_open and not is_headless:
        open_target = display_host if display_host else "127.0.0.1"
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{open_target}:{port}")).start()
        
    threads = max(4, multiprocessing.cpu_count() * 2)
    waitress.serve(app, host=host, port=port, threads=threads)


if __name__ == "__main__":
    main()
