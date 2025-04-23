import os
import asyncio
from langchain_openai import ChatOpenAI

# 配置DeepSeek API
model = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3,
    max_tokens=2000
)

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 政策文件配置（确保文件名与您实际的文件名完全一致）
POLICY_FILES = {
    "财政补贴政策": "财政部发布2025年新能源汽车补贴政策通知.txt",
    "产业发展规划": "国务院办公厅关于印发新能源汽车产业.txt"
}

# 更简洁的分析提示词模板
PROMPT_TEMPLATE = """请从以下政策文本中提取：
1. 3个最关键的政策方向
2. 每个方向的具体措施
3. 预期影响领域

用以下格式输出分析结果：
================================
【政策方向1】
- 措施：具体措施描述
- 影响：影响领域描述

【政策方向2】
...
================================

政策文本内容：
{content}
"""

async def analyze_policy(file_key, file_name):
    """分析单个政策文件"""
    try:
        file_path = os.path.join(current_dir, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误：文件不存在 - {file_path}")
            return None

        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(15000)  # 限制读取长度

        # 调用API分析
        response = await model.agenerate([PROMPT_TEMPLATE.format(content=content)])
        result = response.generations[0][0].text

        # 保存分析结果（与脚本同目录）
        output_file = os.path.join(current_dir, f"{file_key}_分析结果.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== 政策文件分析 ===\n")
            f.write(f"原始文件: {file_name}\n")
            f.write(f"生成时间: {asyncio.get_event_loop().time()}\n\n")
            f.write(result)
        
        print(f"已生成分析结果文件: {os.path.basename(output_file)}")
        return output_file

    except Exception as e:
        print(f"分析{file_key}时出错：{str(e)}")
        return None

async def main():
    print("开始政策文件分析...\n")
    
    # 为每个政策文件创建分析任务
    tasks = []
    for name, file_name in POLICY_FILES.items():
        print(f"正在处理: {file_name}")
        tasks.append(analyze_policy(name, file_name))
    
    # 执行所有分析任务
    results = await asyncio.gather(*tasks)
    
    # 打印生成的文件列表
    print("\n=== 生成的分析结果文件 ===")
    for result_file in results:
        if result_file:
            print(os.path.basename(result_file))
    
    print("\n分析完成！请查看上述文件获取结果。")

if __name__ == "__main__":
    asyncio.run(main())