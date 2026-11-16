---
name: report-generator
description: 从数据中生成专业的markdown和HTML报告，包括图表、表格和分析。
---

# 报告生成器技能

从数据中生成专业的markdown和HTML报告，包括图表、表格和分析。

## 指令

您是一个报告生成专家。当调用时：

1. **分析数据**：
   - 理解数据结构和内容
   - 识别关键指标和见解
   - 计算统计和趋势
   - 检测模式和异常
   - 生成执行摘要

2. **创建报告结构**：
   - 设计清晰、逻辑的部分
   - 创建目录
   - 添加执行摘要
   - 包括详细分析
   - 提供建议

3. **生成可视化**：
   - 为结构化数据创建表格
   - 生成图表（条形图、折线图、饼图、散点图）
   - 添加徽章和指标
   - 包括代码块和示例
   - 格式化数字和百分比

4. **格式化输出**：
   - 生成markdown报告
   - 创建带有样式的HTML报告
   - 导出到PDF
   - 添加品牌和定制
   - 确保响应式设计

## 使用示例

```
@report-generator data.csv
@report-generator --format html
@report-generator --template executive-summary
@report-generator --charts --pdf
@report-generator --compare baseline.json current.json
```

## 报告类型

### 执行摘要报告

```python
def generate_executive_summary(data, title="执行摘要"):
    """
    生成高级执行摘要报告
    """
    from datetime import datetime

    report = f"""# {title}
**生成时间:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

---

## 关键亮点

"""

    # 计算关键指标
    metrics = calculate_key_metrics(data)

    for metric in metrics:
        icon = "✅" if metric['status'] == 'good' else "⚠️" if metric['status'] == 'warning' else "❌"
        report += f"{icon} **{metric['name']}**: {metric['value']}\n"

    report += f"""

---

## 性能概览

| 指标 | 当前 | 之前 | 变化 |
|--------|---------|----------|--------|
"""

    for metric in metrics:
        if 'previous' in metric:
            change = calculate_change(metric['current'], metric['previous'])
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            color = "green" if change > 0 else "red" if change < 0 else "gray"

            report += f"| {metric['name']} | {metric['current']:,} | {metric['previous']:,} | {arrow} {abs(change):.1f}% |\n"

    report += """

---

## 建议

"""

    recommendations = generate_recommendations(metrics)
    for i, rec in enumerate(recommendations, 1):
        priority = rec.get('priority', 'medium')
        emoji = "🔴" if priority == 'high' else "🟡" if priority == 'medium' else "🟢"

        report += f"{i}. {emoji} **{rec['title']}**\n"
        report += f"   {rec['description']}\n\n"

    return report
```

### 数据分析报告

```python
import pandas as pd
import numpy as np
from datetime import datetime

def generate_data_analysis_report(df, title="数据分析报告"):
    """
    生成全面的数据分析报告
    """

    report = f"""# {title}
**日期:** {datetime.now().strftime('%Y-%m-%d')}
**数据集:** {len(df):,} 行 × {len(df.columns)} 列

---

## 目录
1. [数据集概览](#数据集概览)
2. [数据质量](#数据质量)
3. [统计摘要](#统计摘要)
4. [分布](#分布)
5. [相关性](#相关性)
6. [见解](#见解)

---

## 数据集概览

### 基本信息
- **总行数:** {len(df):,}
- **总列数:** {len(df.columns)}
- **内存使用:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
- **重复行:** {df.duplicated().sum():,}

### 列信息

| 列 | 类型 | 非空 | 唯一 | 样本值 |
|--------|------|----------|--------|---------------|
"""

    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].count()
        unique = df[col].nunique()
        samples = df[col].dropna().head(3).tolist()
        sample_str = ", ".join(str(s) for s in samples)

        report += f"| {col} | {dtype} | {non_null:,} | {unique:,} | {sample_str} |\n"

    report += """

---

## 数据质量

### 缺失值

"""

    missing = df.isnull().sum()
    if missing.sum() > 0:
        report += "| 列 | 缺失计数 | 缺失 % |\n"
        report += "|--------|---------------|----------|\n"

        for col in missing[missing > 0].index:
            count = missing[col]
            pct = (count / len(df)) * 100
            report += f"| {col} | {count:,} | {pct:.1f}% |\n"
    else:
        report += "✅ 未检测到缺失值。\n"

    report += "\n### 数据类型问题\n\n"

    # 检查潜在类型问题
    type_issues = []

    for col in df.select_dtypes(include=['object']):
        # 检查列是否为数值型
        try:
            pd.to_numeric(df[col], errors='raise')
            type_issues.append(f"- `{col}` 似乎是数值型但存储为字符串")
        except:
            pass

        # 检查列是否为日期时间型
        try:
            pd.to_datetime(df[col], errors='raise')
            if df[col].str.contains(r'\d{4}-\d{2}-\d{2}').any():
                type_issues.append(f"- `{col}` 似乎是日期时间型但存储为字符串")
        except:
            pass

    if type_issues:
        report += "\n".join(type_issues) + "\n"
    else:
        report += "✅ 未检测到数据类型问题。\n"

    report += """

---

## 统计摘要

### 数值列

"""

    # 添加数值列的统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe()
        report += stats.to_markdown() + "\n"

        # 添加额外统计
        report += "\n### 额外统计\n\n"
        report += "| 列 | 中位数 | 众数 | 标准差 | 方差 |\n"
        report += "|--------|--------|------|---------|----------|\n"

        for col in numeric_cols:
            median = df[col].median()
            mode = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
            std = df[col].std()
            var = df[col].var()

            report += f"| {col} | {median:.2f} | {mode} | {std:.2f} | {var:.2f} |\n"

    report += """

### 分类列

"""

    categorical_cols = df.select_dtypes(include=['object']).columns

    if len(categorical_cols) > 0:
        for col in categorical_cols[:5]:  # 限制为前5个
            report += f"\n#### {col}\n\n"

            value_counts = df[col].value_counts().head(10)

            report += "| 值 | 计数 | 百分比 |\n"
            report += "|-------|-------|------------|\n"

            for value, count in value_counts.items():
                pct = (count / len(df)) * 100
                report += f"| {value} | {count:,} | {pct:.1f}% |\n"

    report += """

---

## 分布

"""

    # 分析数值列的分布
    for col in numeric_cols[:5]:  # 限制为前5个
        report += f"\n### {col} 分布\n\n"

        q1 = df[col].quantile(0.25)
        q2 = df[col].quantile(0.50)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        # 检测异常值
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        report += f"""
**四分位数:**
- Q1 (25%): {q1:.2f}
- Q2 (50%, 中位数): {q2:.2f}
- Q3 (75%): {q3:.2f}
- IQR: {iqr:.2f}

**异常值:** {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)
- 下界: {lower_bound:.2f}
- 上界: {upper_bound:.2f}

"""

    report += """

---

## 相关性

"""

    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()

        report += "\n### 相关矩阵\n\n"
        report += corr_matrix.to_markdown() + "\n"

        # 查找强相关性
        report += "\n### 强相关性 (|r| > 0.7)\n\n"

        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    strong_corr.append((col1, col2, corr_val))

        if strong_corr:
            for col1, col2, corr_val in strong_corr:
                direction = "正相关" if corr_val > 0 else "负相关"
                report += f"- **{col1}** ↔ **{col2}**: {corr_val:.3f} ({direction})\n"
        else:
            report += "未找到强相关性。\n"

    report += """

---

## 见解

"""

    # 生成见解
    insights = generate_insights(df)

    for insight in insights:
        report += f"### {insight['title']}\n\n"
        report += f"{insight['description']}\n\n"

        if 'details' in insight:
            for detail in insight['details']:
                report += f"- {detail}\n"

        report += "\n"

    return report

def generate_insights(df):
    """生成数据见解"""
    insights = []

    # 见解：完整性
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100

    if missing_pct < 1:
        status = "优秀"
        emoji = "✅"
    elif missing_pct < 5:
        status = "良好"
        emoji = "👍"
    else:
        status = "需注意"
        emoji = "⚠️"

    insights.append({
        "title": f"{emoji} 数据完整性: {status}",
        "description": f"总体数据完整性为 {100-missing_pct:.1f}%，缺失值占 {missing_pct:.1f}%。",
        "details": [
            f"总单元格数: {len(df) * len(df.columns):,}",
            f"缺失单元格数: {df.isnull().sum().sum():,}"
        ]
    })

    # 见解：重复记录
    dup_count = df.duplicated().sum()

    if dup_count > 0:
        insights.append({
            "title": f"⚠️ 发现重复记录",
            "description": f"发现 {dup_count:,} 重复行 ({dup_count/len(df)*100:.1f}% 数据集)",
            "details": [
                "考虑移除重复以提高分析准确性",
                "审查业务逻辑以处理重复"
            ]
        })

    return insights
```

### 性能报告

```python
def generate_performance_report(metrics, baseline=None):
    """
    生成性能比较报告
    """

    report = f"""# 性能报告
**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 摘要

"""

    if baseline:
        report += "### 与基线比较\n\n"

        report += "| 指标 | 当前 | 基线 | 变化 | 状态 |\n"
        report += "|--------|---------|----------|--------|--------|\n"

        for metric_name, current_value in metrics.items():
            if metric_name in baseline:
                baseline_value = baseline[metric_name]
                change = ((current_value - baseline_value) / baseline_value) * 100

                if abs(change) < 5:
                    status = "🟢 稳定"
                elif change > 0:
                    status = "🟢 改进" if is_improvement(metric_name, change) else "🔴 退化"
                else:
                    status = "🔴 退化" if is_improvement(metric_name, change) else "🟢 改进"

                report += f"| {metric_name} | {current_value:.2f} | {baseline_value:.2f} | {change:+.1f}% | {status} |\n"

    else:
        report += "### 当前指标\n\n"

        report += "| 指标 | 值 | 状态 |\n"
        report += "|--------|-------|--------|\n"

        for metric_name, value in metrics.items():
            threshold = get_threshold(metric_name)
            status = evaluate_metric(value, threshold)

            report += f"| {metric_name} | {value:.2f} | {status} |\n"

    report += """

---

## 详细分析

"""

    for metric_name, value in metrics.items():
        report += f"### {metric_name}\n\n"

        if baseline and metric_name in baseline:
            baseline_value = baseline[metric_name]
            change = ((value - baseline_value) / baseline_value) * 100

            report += f"- **当前:** {value:.2f}\n"
            report += f"- **基线:** {baseline_value:.2f}\n"
            report += f"- **变化:** {change:+.1f}%\n\n"

            if abs(change) > 10:
                report += f"⚠️ 检测到显著变化。 "
                report += "审查可能影响此指标的近期变化。\n\n"

        else:
            report += f"- **值:** {value:.2f}\n\n"

    return report

def is_improvement(metric_name, change):
    """根据指标类型确定变化是否为改进"""
    # 这些指标越低越好
    lower_is_better = ['response_time', 'error_rate', 'latency', 'load_time']

    for pattern in lower_is_better:
        if pattern in metric_name.lower():
            return change < 0

    return change > 0
```

## HTML报告生成

```python
def generate_html_report(data, title="报告", template="default"):
    """
    生成带样式的HTML报告
    """

    # CSS样式
    css = """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }

        .report-container {
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }

        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }

        h3 {
            color: #7f8c8d;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        th {
            background: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }

        td {
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }

        tr:hover {
            background: #f8f9fa;
        }

        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .metric-value {
            font-size: 2em;
            font-weight: bold;
        }

        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
        }

        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }

        .badge-success {
            background: #2ecc71;
            color: white;
        }

        .badge-warning {
            background: #f39c12;
            color: white;
        }

        .badge-danger {
            background: #e74c3c;
            color: white;
        }

        .chart-container {
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }

        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }

        pre {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }

        .timestamp {
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
    """

    # 生成HTML内容
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        {css}
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <div class="report-container">
            <h1>{title}</h1>
            <p class="timestamp">生成时间: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>

            {generate_html_content(data)}
        </div>
    </body>
    </html>
    """

    return html

def generate_html_content(data):
    """从数据生成HTML内容"""

    html = ""

    # 关键指标部分
    if 'metrics' in data:
        html += "<h2>关键指标</h2>"
        html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">'

        for metric in data['metrics']:
            html += f"""
            <div class="metric-card">
                <div class="metric-label">{metric['name']}</div>
                <div class="metric-value">{metric['value']}</div>
            </div>
            """

        html += "</div>"

    # 表格数据
    if 'table' in data:
        html += "<h2>数据表格</h2>"
        html += generate_html_table(data['table'])

    # 图表
    if 'charts' in data:
        for chart in data['charts']:
            html += f'<h2>{chart["title"]}</h2>'
            html += '<div class="chart-container">'
            html += generate_chart_html(chart)
            html += '</div>'

    return html

def generate_html_table(table_data):
    """从数据生成HTML表格"""

    html = "<table>"

    # 表头
    if 'headers' in table_data:
        html += "<thead><tr>"
        for header in table_data['headers']:
            html += f"<th>{header}</th>"
        html += "</tr></thead>"

    # 行
    html += "<tbody>"
    for row in table_data.get('rows', []):
        html += "<tr>"
        for cell in row:
            html += f"<td>{cell}</td>"
        html += "</tr>"
    html += "</tbody>"

    html += "</table>"
    return html

def generate_chart_html(chart_data):
    """生成Chart.js图表"""

    chart_id = f"chart_{abs(hash(chart_data['title']))}"

    html = f'<canvas id="{chart_id}" width="400" height="200"></canvas>'
    html += f"""
    <script>
        var ctx = document.getElementById('{chart_id}').getContext('2d');
        var chart = new Chart(ctx, {{
            type: '{chart_data.get('type', 'bar')}',
            data: {{
                labels: {chart_data['labels']},
                datasets: [{{
                    label: '{chart_data['title']}',
                    data: {chart_data['data']},
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
    """

    return html
```

## Markdown表格

```python
def generate_markdown_table(data, headers=None, alignment=None):
    """
    从数据生成markdown表格

    alignment: 'left', 'center', 'right' 的列表
    """

    if not data:
        return ""

    # 如果未提供，自动检测表头
    if headers is None:
        if isinstance(data[0], dict):
            headers = list(data[0].keys())
        else:
            headers = [f"列 {i+1}" for i in range(len(data[0]))]

    # 生成表头行
    table = "| " + " | ".join(str(h) for h in headers) + " |\n"

    # 生成对齐行
    if alignment is None:
        alignment = ['left'] * len(headers)

    align_chars = {
        'left': ':--',
        'center': ':-:',
        'right': '--:'
    }

    table += "| " + " | ".join(align_chars.get(a, ':--') for a in alignment) + " |\n"

    # 生成数据行
    for row in data:
        if isinstance(row, dict):
            row_data = [str(row.get(h, '')) for h in headers]
        else:
            row_data = [str(cell) for cell in row]

        table += "| " + " | ".join(row_data) + " |\n"

    return table

# 使用示例
data = [
    {"name": "John", "age": 30, "city": "New York"},
    {"name": "Jane", "age": 25, "city": "San Francisco"},
    {"name": "Bob", "age": 35, "city": "Chicago"}
]

table = generate_markdown_table(
    data,
    headers=['姓名', '年龄', '城市'],
    alignment=['left', 'right', 'left']
)
```

## 图表和可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns

def generate_chart(data, chart_type='bar', title='图表', output='chart.png'):
    """
    从数据生成图表
    """

    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-darkgrid')

    if chart_type == 'bar':
        plt.bar(data['labels'], data['values'])

    elif chart_type == 'line':
        plt.plot(data['labels'], data['values'], marker='o', linewidth=2)

    elif chart_type == 'pie':
        plt.pie(data['values'], labels=data['labels'], autopct='%1.1f%%')

    elif chart_type == 'scatter':
        plt.scatter(data['x'], data['y'], alpha=0.6)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()

    return output

# 用于markdown报告
def embed_chart_in_markdown(chart_path, alt_text="图表"):
    """生成markdown图像嵌入"""
    return f"![{alt_text}]({chart_path})\n"
```

## PDF导出

```python
from markdown import markdown
from weasyprint import HTML

def markdown_to_pdf(markdown_text, output_path='report.pdf', css=None):
    """
    将markdown转换为PDF
    """

    # 将markdown转换为HTML
    html_content = markdown(markdown_text, extensions=['tables', 'fenced_code'])

    # 包装在HTML文档中
    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            {css if css else get_default_pdf_css()}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # 转换为PDF
    HTML(string=html_doc).write_pdf(output_path)

def get_default_pdf_css():
    """PDF导出的默认CSS"""
    return """
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }

        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }

        th {
            background-color: #3498db;
            color: white;
        }

        code {
            background: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
        }

        pre {
            background: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
    """
```

## 报告模板

```python
TEMPLATES = {
    'executive': {
        'sections': ['summary', 'key_metrics', 'recommendations'],
        'style': 'concise'
    },
    'technical': {
        'sections': ['overview', 'detailed_analysis', 'code_examples', 'metrics'],
        'style': 'comprehensive'
    },
    'comparison': {
        'sections': ['baseline', 'current', 'differences', 'trends'],
        'style': 'comparative'
    }
}

def generate_from_template(data, template_name='executive'):
    """从模板生成报告"""

    template = TEMPLATES.get(template_name, TEMPLATES['executive'])

    report = f"# {template_name.title()} 报告\n\n"

    for section in template['sections']:
        report += generate_section(data, section, template['style'])

    return report
```

## 最佳实践

1. **清晰结构化报告**，带有目录
2. **使用视觉层次**（标题、表格、图表）
3. **包含时间戳**，所有报告
4. **为长报告添加执行摘要**
5. **使用一致格式化** 贯穿始终
6. **包括数据源** 和方法论
7. **添加可操作建议**
8. **使用图表表示趋势**，表格表示详细数据
9. **导出到多种格式**（MD、HTML、PDF）
10. **尽可能自动化报告生成**

## 注释

- 保持报告专注和可操作
- 为数据类型使用适当可视化
- 包括摘要和详细视图
- 版本控制报告模板
- 测试不同数据大小的PDF导出
- 考虑HTML报告的可访问性
- 为HTML报告使用响应式设计
- 为性能缓存生成的图表
