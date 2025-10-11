import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# 移除了 matplotlib 字体设置，因为现在使用 plotly


# 页面标题
st.title("AO-shaping 数据可视化工具")

# 文件选择器 - 两种方式选择文件
st.subheader("选择pkl文件方式一：上传文件")
uploaded_file = st.file_uploader("选择一个pkl文件", type="pkl")

st.subheader("选择pkl文件方式二：选择当前目录中的文件")
current_dir = os.path.dirname(os.path.abspath(__file__))
pkl_files = [f for f in os.listdir(current_dir) if f.endswith('.pkl')]
selected_file = st.selectbox("选择pkl文件", pkl_files)

# 确定要加载的文件
if uploaded_file is not None:
    file_to_load = uploaded_file
    file_name = uploaded_file.name
else:
    file_to_load = os.path.join(current_dir, selected_file)
    file_name = selected_file

# 加载数据
if st.button("加载数据") or (uploaded_file is not None):
    try:
        # 尝试加载文件
        if uploaded_file is None:
            data = pd.read_pickle(file_to_load, compression="zip")
        
        st.success(f"成功加载文件: {file_name}")
        
        # 数据预处理：合并带下划线的列名
        # 创建列名映射
        column_mapping = {}
        columns_to_drop = []
        
        for col in data.columns:
            if '_' in col and col != '_cam':
                # 去掉下划线的列名
                clean_col = col.replace('_', '')
                # 如果存在不带下划线的同名列，则合并数据
                if clean_col in data.columns and clean_col != col:
                    # 合并数据：将带下划线的列数据追加到不带下划线的列数据后面
                    try:
                        # 如果两列都是数值类型，则合并为数组
                        if isinstance(data[col].iloc[0], (int, float, np.number)) and isinstance(data[clean_col].iloc[0], (int, float, np.number)):
                            # 合并两列数据
                            merged_data = []
                            for i in range(len(data)):
                                clean_data = data[clean_col].iloc[i]
                                merged_data =  data[col].iloc[i] if pd.isna(clean_data) else clean_data
                            data[clean_col] = merged_data
                        # 如果是数组类型，则合并数组
                        elif isinstance(data[col].iloc[0], (list, np.ndarray)) and isinstance(data[clean_col].iloc[0], (list, np.ndarray)):
                            # 合并两列数据
                            merged_data = []
                            for i in range(len(data)):
                                merged_data.append(np.concatenate([data[clean_col].iloc[i], data[col].iloc[i]]))
                            data[clean_col] = merged_data
                        else:
                            # 其他情况，直接合并
                            data[clean_col] = data[clean_col].astype(str) + ", " + data[col].astype(str)
                    except Exception as e:
                        st.warning(f"合并列 {col} 和 {clean_col} 时出错: {str(e)}")
                    # 标记带下划线的列待删除
                    columns_to_drop.append(col)
        
        # 删除带下划线的列
        data = data.drop(columns=columns_to_drop)
        
        # 将数据存储在session state中，以便在组件间共享
        st.session_state['loaded_data'] = data
        st.session_state['file_name'] = file_name
         
    except Exception as e:
        st.error(f"加载文件时出错: {str(e)}")
        st.write("请检查文件格式是否正确")

# 如果数据已加载，则显示数据可视化界面
if 'loaded_data' in st.session_state:
    data = st.session_state['loaded_data']
    file_name = st.session_state['file_name']
    
    # 显示数据基本信息
    st.subheader("数据基本信息")
    st.write(f"数据类型: {type(data)}")
    
    # 在最上面绘制 data.J
    if hasattr(data, 'J'):
        st.subheader("J 值变化趋势")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(data.J))), y=data.J, mode='lines+markers', name='J 值'))
        
        # 找到最小值和最大值的索引
        j_values = data.J
        min_idx = np.argmin(j_values)
        max_idx = np.argmax(j_values)
        min_value = j_values[min_idx]
        max_value = j_values[max_idx]
        
        # 标注最小值点
        fig.add_trace(go.Scatter(
            x=[min_idx],
            y=[min_value],
            mode='markers',
            marker=dict(size=12, color='red', symbol='circle'),
            name=f'最小值: {min_value:.4f} (帧 {min_idx})'
        ))
        
        # 标注最大值点
        fig.add_trace(go.Scatter(
            x=[max_idx],
            y=[max_value],
            mode='markers',
            marker=dict(size=12, color='orange', symbol='circle'),
            name=f'最大值: {max_value:.4f} (帧 {max_idx})'
        ))
        
        fig.update_layout(
            title='J 值随帧数变化',
            xaxis_title='帧索引',
            yaxis_title='J 值',
            width=800,
            height=400,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        st.plotly_chart(fig)
        
        # 添加跳转到最大值和最小值帧的按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"跳转到最小值帧 ({min_idx})"):
                st.session_state['frame_idx'] = min_idx
                st.rerun()
        with col2:
            if st.button(f"跳转到最大值帧 ({max_idx})"):
                st.session_state['frame_idx'] = max_idx
                st.rerun()
    
    # 显示 _statics 内容（如果存在）
    if hasattr(data, 'statics'):
        st.subheader("statics 数据")
        statics_data = data._statics
        
        # 创建 pos 和 neg 两个表格
        if 'pos' in statics_data and 'neg' in statics_data:
            pos_data = statics_data['pos']
            neg_data = statics_data['neg']
            
            # 创建两个并排的列来显示表格
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Pos 数据**")
                pos_df = pd.DataFrame([pos_data])
                st.table(pos_df)
                
            with col2:
                st.write("**Neg 数据**")
                neg_df = pd.DataFrame([neg_data])
                st.table(neg_df)
    
    # 检查数据结构
    if hasattr(data, '_cam'):
        st.write(f"数据包含 {len(data._cam)} 个相机帧")
        
        # 创建帧选择器，使用session state存储帧索引
        if 'frame_idx' not in st.session_state:
            st.session_state['frame_idx'] = 0
            
        def update_frame_idx():
            st.session_state['frame_idx'] = st.session_state['slider_frame_idx']
            
        # 添加手动输入帧数值的功能
        col1, col2 = st.columns([3, 1])
        with col1:
            frame_idx = st.slider("选择帧", 0, len(data._cam) - 1, st.session_state['frame_idx'], key="slider_frame_idx", on_change=update_frame_idx)
        with col2:
            # 手动输入帧数值
            manual_frame = st.number_input("手动输入帧", min_value=0, max_value=len(data._cam) - 1, value=st.session_state['frame_idx'], key="manual_frame")
            # 检查是否需要更新帧索引
            if manual_frame != st.session_state['frame_idx']:
                st.session_state['frame_idx'] = manual_frame
                st.rerun()
                
        st.session_state['frame_idx'] = frame_idx
        
        # 显示当前帧的图片
        st.subheader(f"帧 {frame_idx} 的图片")
        
        # 使用缓存函数来提高性能
        @st.cache_data(ttl=300, show_spinner=False)
        def get_cached_image_data(_data, frame_index):
            if frame_index == 0:
                return _data._cam[frame_index]
            else:
                return _data._cam[frame_index][0], _data._cam[frame_index][1]
        
        try:
            # 根据原始代码的逻辑处理不同帧的图片
            # 特别注意第0帧的特殊结构
            if frame_idx == 0:
                # 第0帧只有1张图片
                img = get_cached_image_data(data, frame_idx)
                # 显示单张图片
                st.write("图片 1:")
                fig = px.imshow(img, color_continuous_scale='gray')
                fig.update_layout(
                    title=f'帧: {frame_idx}, ID: {frame_idx}, RMS:{data.J[frame_idx]:.3f}',
                    width=600,
                    height=500
                )
                st.plotly_chart(fig)
                st.write(f"图片最大值: {np.max(img)}")
            else:
                # 其他帧有两张图片，显示两张图片
                img1, img2 = get_cached_image_data(data, frame_idx)
                
                # 显示第一张图片
                st.write("图片 1:")
                fig1 = px.imshow(img1, color_continuous_scale='gray')
                fig1.update_layout(
                    title=f'帧: {frame_idx}, 图片1, RMS:{data.J[frame_idx]:.3f}',
                    width=600,
                    height=500
                )
                st.plotly_chart(fig1)
                st.write(f"图片1最大值: {np.max(img1)}")
                
                # 显示第二张图片
                st.write("图片 2:")
                fig2 = px.imshow(img2, color_continuous_scale='gray')
                fig2.update_layout(
                    title=f'帧: {frame_idx}, 图片2',
                    width=600,
                    height=500
                )
                st.plotly_chart(fig2)
                st.write(f"图片2最大值: {np.max(img2)}")
        
        except Exception as e:
            st.error(f"显示图片时出错: {str(e)}")
            st.write("尝试显示原始数据结构:")
            st.write(data._cam[frame_idx])
        
        # 显示数据中的其他参数
        st.subheader("数据中的其他参数")
        for col in data.columns:
            if col != '_cam':
                st.write(f"\n{col}:")
                try:
                    # 尝试获取特定帧的数据
                    param_data = data[col].iloc[frame_idx]
                    
                    # 检查是否为数组或numpy array，如果是则用条形图展示
                    if isinstance(param_data, (list, np.ndarray)) and len(param_data) > 0:
                        st.write("数据可视化 (条形图):")
                        # 创建条形图
                        fig = go.Figure(data=[go.Bar(x=list(range(len(param_data))), y=param_data)])
                        fig.update_layout(
                            title=f'{col} 数据分布',
                            xaxis_title='索引',
                            yaxis_title='值',
                            width=800,
                            height=400
                        )
                        st.plotly_chart(fig)
                        st.write("原始数据:")
                        st.write(param_data)
                    else:
                        # 如果是单个数值，检查整个列数据是否可以绘图
                        full_data = data[col]
                        if isinstance(full_data, (list, np.ndarray)) and len(full_data) > 1:
                            st.write("数据可视化 (整个数组):")
                            # 创建条形图显示整个数组
                            fig = go.Figure(data=[go.Bar(x=list(range(len(full_data))), y=full_data)])
                            fig.update_layout(
                                title=f'{col} 数据分布 (所有帧)',
                                xaxis_title='帧索引',
                                yaxis_title='值',
                                width=800,
                                height=400
                            )
                            st.plotly_chart(fig)
                            st.write("当前帧数据:")
                            st.write(param_data)
                        else:
                            st.write(param_data)
                except Exception as e:
                    st.write(f"无法显示 {col} 的帧数据: {str(e)}")
                    st.write(f"显示完整的 {col} 数据:")
                    full_data = data[col]
                    # 检查是否为数组或numpy array，如果是则用条形图展示
                    if isinstance(full_data, (list, np.ndarray)) and len(full_data) > 0:
                        st.write("数据可视化 (条形图):")
                        # 创建条形图
                        fig = go.Figure(data=[go.Bar(x=list(range(len(full_data))), y=full_data)])
                        fig.update_layout(
                            title=f'{col} 数据分布',
                            xaxis_title='索引',
                            yaxis_title='值',
                            width=800,
                            height=400
                        )
                        st.plotly_chart(fig)
                        st.write("原始数据:")
                        st.write(full_data)
                    else:
                        # 如果是单个数值，检查整个列数据是否可以绘图
                        if isinstance(full_data, (list, np.ndarray)) and len(full_data) > 1:
                            st.write("数据可视化 (整个数组):")
                            # 创建条形图显示整个数组
                            fig = go.Figure(data=[go.Bar(x=list(range(len(full_data))), y=full_data)])
                            fig.update_layout(
                                title=f'{col} 数据分布',
                                xaxis_title='索引',
                                yaxis_title='值',
                                width=800,
                                height=400
                            )
                            st.plotly_chart(fig)
                        st.write(full_data)
    else:
        st.warning("数据中没有找到_cam属性")
        st.write("数据结构:")
        st.write(data.head())

# 添加使用说明
with st.expander("使用说明"):
    st.markdown("""
    ### 如何使用此工具
    1. 通过文件上传器选择一个pkl文件，或从下拉菜单中选择当前目录中的pkl文件
    2. 点击"加载数据"按钮
    3. 使用滑块选择要查看的帧
    4. 查看图片和相关参数
    
    ### 注意事项
    - 工具会尝试解析pkl文件中的_cam属性来显示图片
    - 特别处理了第0帧的特殊结构（只有一张图片而不是数组）
    - 如果文件格式不同，可能需要修改代码以适应不同的数据结构
    """)