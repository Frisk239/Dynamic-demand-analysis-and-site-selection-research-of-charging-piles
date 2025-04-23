import os
from pyvis.network import Network

def load_enhanced_data():
    """加载增强版政策网络数据"""
    return {
        "nodes": [
            {"id": "财政补贴", "type": "财政工具", "size": 98.85, "impact": 0.9},
            {"id": "强制比例", "type": "行政工具", "size": 80, "impact": 0.95},
            {"id": "技术攻关", "type": "技术工具", "size": 16.25, "impact": 0.85},
            {"id": "充电基建", "type": "基础设施", "size": 45, "impact": 0.75},
            {"id": "税收优惠", "type": "财政工具", "size": 30, "impact": 0.8},
            {"id": "标准制定", "type": "行政工具", "size": 25, "impact": 0.7},
            {"id": "公共电动化", "type": "政策目标", "size": 20, "impact": 1.0},
            {"id": "产业升级", "type": "政策目标", "size": 15, "impact": 0.9}
        ],
        "links": [
            {"source": "财政补贴", "target": "充电基建", "value": 23.08, "type": "资金支持"},
            {"source": "财政补贴", "target": "技术攻关", "value": 16.25, "type": "专项奖励"},
            {"source": "强制比例", "target": "公共电动化", "value": 80, "type": "行政约束"},
            {"source": "充电基建", "target": "公共电动化", "value": 50, "type": "设施保障"},
            {"source": "税收优惠", "target": "产业升级", "value": 30, "type": "激励政策"},
            {"source": "标准制定", "target": "技术攻关", "value": 25, "type": "规范引导"},
            {"source": "技术攻关", "target": "产业升级", "value": 40, "type": "技术转化"}
        ]
    }

def draw_interactive_policy_network(data, output_html="enhanced_policy_network_interactive.html"):
    net = Network(height='800px', width='100%', bgcolor='#ffffff', font_color='#000000', directed=True)

    # 类型颜色映射
    type_colors = {
        "财政工具": "#4e79a7",
        "行政工具": "#e15759",
        "技术工具": "#76b7b2",
        "基础设施": "#f28e2b",
        "政策目标": "#59a14f"
    }

    # 添加节点
    for node in data["nodes"]:
        label = f"{node['id']}\n影响力: {node['impact']}\n资金规模: {node['size']}"
        size = 15 + node['size'] * 0.5
        net.add_node(
            node["id"],
            label=label,
            title=label,
            color=type_colors.get(node["type"], "#cccccc"),
            size=size
        )

    # 添加边并手动拉长特殊边
    for link in data["links"]:
        label = f"{link['type']}：{link['value']}"
        edge_params = {
            "value": link["value"],
            "title": label,
            "label": link["type"]
        }

        # 拉长“强制比例”→“公共电动化”这条边
        if link["source"] == "强制比例" and link["target"] == "公共电动化":
            edge_params["length"] = 400

        net.add_edge(link["source"], link["target"], **edge_params)

    # 配置物理引擎参数
    net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 14,
          "face": "Microsoft YaHei"
        },
        "shadow": true
      },
      "edges": {
        "arrows": {
          "to": {"enabled": true}
        },
        "smooth": {
          "type": "dynamic"
        },
        "color": {
          "inherit": false
        },
        "shadow": true
      },
      "physics": {
        "enabled": true,
        "stabilization": {
          "iterations": 200
        },
        "barnesHut": {
          "gravitationalConstant": -12000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.03
        }
      }
    }
    """)

    # 保存为HTML文件
    net.write_html(output_html)
    print(f"✅ 交互式网络图已成功保存至：{output_html}\n👉 请用浏览器打开查看。")

if __name__ == "__main__":
    try:
        data = load_enhanced_data()
        draw_interactive_policy_network(data)
    except Exception as e:
        print(f"绘图出错: {str(e)}")
        import traceback
        traceback.print_exc()

