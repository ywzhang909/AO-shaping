 from ao_shaping.optimizer.rl.envs import VectorWaveOpticsSim
 
 
 
 def test_vector_ao_correction():
    # --- 初始化 ---
    propagator = VectorWaveOpticsSim()
    
    # --- 步骤 1: 创建初始光束 (线偏振高斯光) ---
    Ex_init = create_gaussian_field(w0=0.05)
    Ey_init = np.zeros_like(Ex_init)
    print("✅ 初始化完成: 初始光束为X偏振高斯光")
    
    # --- 步骤 2: 分步傅里叶传播 (经过大气) ---
    Ex_prop, Ey_prop = propagator.propagate(Ex_init, Ey_init)
    
    # --- 步骤 3: 自适应光学校正 ---
    print("🚀 开始自适应光学校正 (目标: 径向偏振光)...")
    Ex_final, Ey_final, Ex_tar, Ey_tar = ao_system.correct_to_vector_beam(Ex_prop, Ey_prop, target_mode="radial")
    
    # ================== 7. 可视化 ==================
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    # --- 图1: 初始状态 ---
    rgb_init = calculate_stokes_rgb(Ex_init, Ey_init)
    axes[0].imshow(rgb_init)
    axes[0].set_title("1. 初始光场\n(线偏振高斯光)")
    axes[0].axis('off')
    
    # --- 图2: 传播后 (湍流干扰) ---
    rgb_prop = calculate_stokes_rgb(Ex_prop, Ey_prop)
    axes[1].imshow(rgb_prop)
    axes[1].set_title("2. 传播1km后\n(湍流畸变)")
    axes[1].axis('off')
    
    # --- 图3: 目标模式 ---
    rgb_tar = calculate_stokes_rgb(Ex_tar, Ey_tar)
    axes[2].imshow(rgb_tar)
    axes[2].set_title("3. 目标模式\n(理想径向偏振)")
    axes[2].axis('off')
    
    # --- 图4: 校正后 ---
    rgb_final = calculate_stokes_rgb(Ex_final, Ey_final)
    axes[3].imshow(rgb_final)
    axes[3].set_title("4. AO校正后\n(矢量光恢复)")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()