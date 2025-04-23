import os
import asyncio
from langchain_openai import ChatOpenAI

# 配置DeepSeek API
model = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3,
    max_tokens=4000
)

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 政策文件配置（使用相对路径）
POLICY_FILES = {
    "财政补贴政策": "财政部发布2025年新能源汽车补贴政策通知.txt",
    "产业发展规划": "国务院办公厅关于印发新能源汽车产业.txt"
}

# 明确定义政策强度标准
POLICY_STRENGTH_DEFINITION = """
【政策强度评估标准（必须严格执行）】
1分（█▌▌▌▌）: 仅原则性表述，无具体措施
   ▶ 示例："支持新能源汽车发展"
2分（██▌▌▌）: 提出方向性要求，无量化指标
   ▶ 示例："加大财政支持力度"
3分（███▌▌）: 明确具体措施，有有限资源保障
   ▶ 示例："安排10亿元补贴资金"
4分（████▌）: 多措施组合，资源充足+时限要求
   ▶ 示例："2023-2025年累计投入100亿元"
5分（█████）: 强制性要求+跨部门协同+追责条款
   ▶ 示例："2025年起新购公务用车100%新能源化，违者追责"
"""

# 深度分析提示词
DEEP_ANALYSIS_PROMPT = f"""作为政策分析专家，请执行：

{POLICY_STRENGTH_DEFINITION}

【分析要求】
1. 识别3个核心主题（按强度降序排列）
2. 每个主题必须包含：
   - 强度评分（严格对照上述标准）
   - 评分依据（引用具体条款）
   - 实施主体（中央部委/地方政府/企业）
   - 政策工具清单
   - 原文关键句

【输出格式】
※主题分析※
1. [主题名称]
   ► 强度: [评分图标] [x/5]
   ► 依据: "直接引用量化指标或强制性表述"
   ► 主体: [实施主体]
   ► 工具: [工具1]、[工具2]
   ► 原文: "政策文本原文摘录"

政策文本：
{{content}}"""

# 优化建议提示词
ADVICE_PROMPT = f"""基于以下分析生成可操作建议：
{{analysis_result}}

【建议规则】
1. 强度≥4分的领域：建议优化实施细节
2. 强度≤2分的领域：建议加强政策力度
3. 必须引用强度评分作为依据

【输出格式】
※优化建议※
1. [主题1]（当前强度: x/5）
   ► 建议: 具体措施（基于: "引用分析依据"）
   ► 类型: 💰财政/⚖️法规/🔧技术
   ► 预期影响: [描述]
"""

# 对比分析提示词
COMPARE_PROMPT = f"""对比分析以下政策：
政策A：{{policy_a_name}}
{{policy_a_content}}

政策B：{{policy_b_name}}
{{policy_b_content}}

【对比维度】
1. 强度差异（相同主题的评分对比）
2. 措施互补性
3. 实施主体协同性

【输出格式】
※对比报告※
一、强度差异
• [主题1]: A[评分] vs B[评分]
   - 差异原因: [分析]

二、措施互补
• A侧重: [措施]
• B侧重: [措施]

三、协同建议
[具体协同方案]"""

async def analyze_policy(file_key, file_name):
    """执行单个政策文件的深度分析"""
    file_path = os.path.join(current_dir, file_name)
    
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(15000)
        
        # 执行深度分析
        analysis_result = await model.agenerate([
            DEEP_ANALYSIS_PROMPT.format(content=content)
        ])
        analysis_text = analysis_result.generations[0][0].text
        
        # 生成优化建议
        advice_result = await model.agenerate([
            ADVICE_PROMPT.format(analysis_result=analysis_text)
        ])
        advice_text = advice_result.generations[0][0].text
        
        # 保存分析报告
        report_path = os.path.join(current_dir, f"{file_key}_分析报告.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"=== 政策深度分析 ===\n")
            f.write(f"文件: {file_name}\n")
            f.write(f"生成时间: {asyncio.get_event_loop().time()}\n")
            f.write("="*50 + "\n")
            f.write(analysis_text + "\n")
            f.write(advice_text + "\n")
        
        print(f"✅ 已生成: {os.path.basename(report_path)}")
        return {
            "name": file_key,
            "content": analysis_text,
            "path": report_path
        }
        
    except Exception as e:
        print(f"❌ 分析{file_key}失败: {str(e)}")
        return None

async def generate_comparison(policy_a, policy_b):
    """生成两份政策的对比报告"""
    try:
        # 读取分析内容
        with open(policy_a["path"], 'r', encoding='utf-8') as f:
            content_a = f.read()
        with open(policy_b["path"], 'r', encoding='utf-8') as f:
            content_b = f.read()
        
        # 执行对比分析
        compare_result = await model.agenerate([
            COMPARE_PROMPT.format(
                policy_a_name=policy_a["name"],
                policy_a_content=content_a,
                policy_b_name=policy_b["name"],
                policy_b_content=content_b
            )
        ])
        compare_text = compare_result.generations[0][0].text
        
        # 保存对比报告
        compare_path = os.path.join(current_dir, "政策对比报告.txt")
        with open(compare_path, 'w', encoding='utf-8') as f:
            f.write("=== 双政策对比分析 ===\n")
            f.write(f"对比对象: {policy_a['name']} vs {policy_b['name']}\n")
            f.write("="*50 + "\n")
            f.write(compare_text)
        
        print(f"✅ 已生成: 政策对比报告.txt")
        return compare_path
        
    except Exception as e:
        print(f"❌ 对比分析失败: {str(e)}")
        return None

async def main():
    print("="*50)
    print("新能源汽车政策分析系统")
    print("="*50 + "\n")
    
    # 检查文件是否存在
    missing_files = [
        f for f in POLICY_FILES.values() 
        if not os.path.exists(os.path.join(current_dir, f))
    ]
    if missing_files:
        print(f"❌ 缺少政策文件: {missing_files}")
        return
    
    # 执行分析
    print("正在分析政策文件...")
    policy_a = await analyze_policy("财政补贴政策", POLICY_FILES["财政补贴政策"])
    policy_b = await analyze_policy("产业发展规划", POLICY_FILES["产业发展规划"])
    
    if policy_a and policy_b:
        # 生成对比报告
        print("\n正在生成对比报告...")
        await generate_comparison(policy_a, policy_b)
    
    # 打印结果文件列表
    print("\n" + "="*50)
    print("生成结果文件：")
    print(f"- {policy_a['name']}_分析报告.txt")
    print(f"- {policy_b['name']}_分析报告.txt")
    print("- 政策对比报告.txt")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())