import numpy as np
import numba
import pandas as pd
import time
import os
import tqdm
import matplotlib.pyplot as plt

from drivers import NlightDM
import nidaqmx
from nidaqmx.constants import AcquisitionType

ROOT_DIR = './test'
Last_voltage_path = os.path.join(ROOT_DIR, 'last_v.csv')

EPOCHS = 10000 # 最大迭代次数
SHRANK_ITER = 100
GAMMA = 0.5
DELTA = 5  # 扰动
GRADIENT_CLIP = (300/200)

R = 10
V0 = 0

V_MAX = 499
V_MIN = -300

# adam parameters
beta1 = 0.9
beta2 = 0.999

# cooldown momentum spgd
rho0 = 0.99

# ADC
# 定义采样参数
sample_rate = 5000  # 采样率 (Hz)
samples_per_channel = 10  # 每次读取的样本数
num_channels = 1  # 通道数


def gen_file_name(dir, postfix:str=None):

    fname = os.listdir(dir)
    fname = len(fname)+1

    if not postfix: # make dir
        path = os.path.join(dir, str(fname))
        if not postfix and not os.path.exists(path):
            os.makedirs(path)
    else:
        if postfix[0] != '.':
            postfix = '.'+postfix
        path = os.path.join(dir, str(fname))+postfix
    return path

def save_list_to_txt(lst, file_path):
    try:
        with open(file_path, 'w') as file:
            for item in lst:
                file.write(str(item) + '\n')
        print(f"列表已成功保存到 {file_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

def optimizer( delta=DELTA, gamma=GAMMA, algorithm='adam', continue_opt=False):
    """
    优化器
    :param r_bucket: 半径
    :param delta: 扰动
    :param gamma: 学习率
    :param algorithm: 优化算法
    :continue_opt: 是否继续优化
    :return:
    """
    with  NlightDM() as dm, nidaqmx.Task() as task:

        # 添加模拟输入通道
        task.ai_channels.add_ai_voltage_chan("Dev1/ai0")  # 假设使用设备名为 "Dev1"，通道为 "ai0"

        # 配置采样时钟
        task.timing.cfg_samp_clk_timing(
            rate=sample_rate,
            sample_mode=AcquisitionType.HW_TIMED_SINGLE_POINT,
            samps_per_chan=samples_per_channel
        )

        # 启动任务
        task.start()

        dm.reset_all()
        if continue_opt and os.path.exists(Last_voltage_path):
            init_v = np.loadtxt(Last_voltage_path)
        else:
            init_v = np.zeros((dm.DM_Num), dtype=np.float64)
            init_v[0] = V0

        dm.send_voltages(init_v)

        def calc_j():
            data = task.read(number_of_samples_per_channel=samples_per_channel)[:samples_per_channel]
            return np.mean(data)


        history = [{'J':calc_j(), '_v':init_v, 'diff':0, 'gamma':gamma, '_delta': delta, '_epoch':-1}]
        list_J = []
        with tqdm.tqdm(total=EPOCHS, desc=f'SPGD iter {EPOCHS}/{sample_num}', dynamic_ncols=True) as bar:
            for epoch in range(EPOCHS):
                disturb_v = np.random.binomial(1, 0.5, (dm.DM_Num,)).astype(float) * 2.0 - 1.0
                disturb_v = disturb_v * delta
                disturb_v[0] = 0
                # disturb_v = np.zeros_like(init_v)
                dm.send_voltages(init_v+disturb_v)
                pos_j = calc_j()
                list_J.append(pos_j)

                dm.send_voltages(init_v-disturb_v)
                neg_j = calc_j()

                diff = pos_j-neg_j

                if algorithm == 'spgd':
                    # diff = np.clip(diff, -GRADIENT_CLIP, GRADIENT_CLIP)
                    update = gamma * diff* disturb_v

                elif algorithm == 'adam':
                    gradient = - diff * disturb_v
                    if epoch == 0:
                        m = np.zeros_like(init_v)
                        v = np.zeros_like(init_v)
                    @numba.jit(nopython=True)
                    def update_epoch(
                        epoch:int,
                        m:np.ndarray[float],
                        v:np.ndarray[float],
                        gradient:np.ndarray[float],
                        gamma:float
                    ):
                        m = beta1 * m + (1 - beta1) * (gradient)
                        v = beta2 * v + (1 - beta2) * (gradient ** 2)
                        m_hat = m / (1 - beta1 ** (epoch + 1))
                        v_hat = v / (1 - beta2 ** (epoch + 1))
                        update = - gamma * m_hat / ((v_hat)**0.5 + 1e-8)
                        return update, m, v

                    update, m, v = update_epoch(epoch, m, v, gradient)

                else:
                    grad = np.sign(diff)
                    update = grad*disturb_v

                # 梯度裁剪，不要变化太大
                update = np.clip(update, -50, 50)
                init_v = np.clip(init_v + update, V_MIN, V_MAX)

                log = {'J' : neg_j, '_epoch':epoch, '_v':init_v , "diff": diff}
                history.append(log)

                bar.set_postfix({k:v for k,v in log.items() if k[0] != '_'})
                bar.update(1)
        file_path = 'output.txt'
        save_list_to_txt(list_J, file_path)
        return pd.DataFrame(history)


SAVED = False
if __name__ == '__main__':

    for sample_num in range(1):

        dfhistory = optimizer(R)


        disp_grid = (1,2)
        fig, ax = plt.subplots(*disp_grid)
        ax[0].plot(dfhistory['J'].to_list())
        cm3 = ax[1].imshow(np.stack(dfhistory['_v'].to_list()).transpose(), interpolation='nearest', aspect='auto')

        if SAVED:
            saved_dir = gen_file_name(ROOT_DIR)
            plt.savefig(gen_file_name(saved_dir, 'png'))
            dfhistory.to_pickle(gen_file_name(saved_dir, 'pkl'), compression='zip')

        else:
            plt.show()
        plt.close('all')
        last_voltage:np.ndarray = dfhistory.iloc[-1]['_v']

        last_J = dfhistory.iloc[-1]['J']
        print(f"{last_J=}")
        np.savetxt(Last_voltage_path, np.round(last_voltage), fmt='%d')
        time.sleep(2)
