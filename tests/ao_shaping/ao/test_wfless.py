import os

import matplotlib.pyplot as plt
import pandas as pd

from ao_shaping.wfless.DM_cam import optimize_pib
from ao_shaping.utils.file import get_init_V_by_rms

def test_optimize_pib():
    '''
    args.add_argument("--cam_id", type=int, default=os.environ.get('Far_Cam_ID', 0), help="远场光斑CCD设备ID (default: Far_Cam_ID/0)")
    args.add_argument("-c", "--center", type=parse_tuple, default=(665, 403), help="场光斑CCD中心位置 (default: (665, 403))")
    args.add_argument("-t","--exposure_time_ms", type=int, default=100, help="远场光斑CCD曝光时间 (毫秒) (default: 60)")
    args.add_argument("--epochs", type=int, default=4_000, help="优化迭代次数 (default: 4000)")
    args.add_argument("-r", "--r_bucket", type=float, default=18, help="渲染半径桶大小 (default: 18)")
    args.add_argument("--delta", type=float, default=2, help="优化步长 (default: 2)")
    args.add_argument("--lr", type=float, default=2, help="优化学习率 (default: 2)")
    args.add_argument("--weight_decay", type=float, default=0.0, help="权重衰减 (default: 0.0)")
    args.add_argument("--shrank_iter", type=int, default=300, help="优化迭代次数后收缩半径桶和步长 (default: 300)")
    args.add_argument("--show", type=bool, default=True, help="显示远场光斑CCD图像和优化历史 (default: True)")
    args.add_argument("-s", "--cam_size", type=int, default=200, help="相机开窗大小 (default: 200*200)")
    '''

    cam_id = os.environ['Far_Cam_ID']
    center = (787, 286)
    exposure_time_ms = 500
    epochs = 4_000
    r_bucket = 10
    cam_size = 200
    init_v = get_init_V_by_rms()

    res_list = optimize_pib(
        cam_id=cam_id, center=center, exposure_time_ms=exposure_time_ms, cam_size=cam_size,
        epochs=epochs, r_bucket=r_bucket, init_v=init_v,)
    res_df = pd.DataFrame(res_list)
    max_j_id = res_df['pib'].argmax()
    last_V = res_df.iloc[max_j_id]["_v"]
    max_j = res_df.iloc[max_j_id]['pib']

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    # init image
    ax[0, 0].imshow(res_df.iloc[0]["_img"])
    ax[0, 0].set_title(f"Init Image, pib={res_df.iloc[0]['pib']:.3f}")
    ax[0, 0].axis("off")
    # best image
    ax[0, 1].imshow(res_df.iloc[max_j_id]["_img"])
    ax[0, 1].set_title(f"Best Image, pib={max_j:.3f}")
    ax[0, 1].axis("off")
    # pib history
    ax[1, 0].plot(res_df["pib"])
    ax[1, 0].set_title("PIB History")
    ax[1, 0].set_xlabel("Epoch")
    ax[1, 0].set_ylabel("PIB")
    # best voltages plot bar
    ax[1, 1].bar(range(64), last_V)
    ax[1, 1].set_title("Best Voltages")
    ax[1, 1].set_xlabel("Unit ID")
    ax[1, 1].set_ylabel("Voltage")
        
    plt.tight_layout()
    plt.show()