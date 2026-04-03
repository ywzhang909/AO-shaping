import pandas as pd
import matplotlib.pyplot as plt

from ao_shaping.optimizer.wf.rms import optimizer_rms

def test_wf_optimizer():
    init_V = [0 for _ in range(64)]
    res_list = optimizer_rms(
        init_v=init_V.copy(),
        epochs=20_000)

    res_df = res_list.dataframe
    min_id = res_df["J"].argmin()
    min_iter = res_df.iloc[min_id]
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    # 绘制J的变化趋势
    ax[0, 0].scatter(min_iter["_epoch"], min_iter["J"], color="red", marker="*", s=100)
    ax[0, 0].plot(res_df["_epoch"], res_df["J"])
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel("J")
    ax[0, 0].set_title(f"Min J: {min_iter['J']:.3f} @ epoch {min_iter['_epoch']}")
    # 绘制保存的电压
    ax[0, 1].bar(range(64), min_iter["_v"])
    ax[0, 1].set_xlabel("DM Unit")
    ax[0, 1].set_ylabel("Voltage")
    ax[0, 1].set_title(f"Min J: {min_iter['J']:.3f} @ epoch {min_iter['_epoch']}")
    # 绘制保存的初始波前
    ax[1, 0].imshow(res_df.iloc[0]["_wavefront"][0], cmap='gray')
    ax[1, 0].set_title("init wavefront")
    ax[1, 0].axis('off')
    # 绘制保存的最优波前
    ax[1, 1].imshow(min_iter["_wavefront"][1], cmap='gray')
    ax[1, 1].set_title("opt wavefront")
    ax[1, 1].axis('off')
    plt.show()

if __name__ == "__main__":
    test_wf_optimizer()