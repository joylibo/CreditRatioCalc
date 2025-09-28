from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
import os
import json

router = APIRouter()

# DeepSeek API Prompts for credit indicator conversion
CONVERSION_PROMPT = """
你是一个专业的信用评估指标设计专家。你的任务是将约定的平安社区信用评估指标体系转换为其他应用场景下的相应指标体系。

重要要求：
1. 输出必须是纯JSON格式，不要包含markdown代码块、```或其他标记
2. 返回对象必须包含两个字段：converted_data和error（error为空字符串表示成功）
3. converted_data必须是数组，每个元素必须是长度为3的字符串数组：[指标类型, 二级指标名称, 指标描述内容]
4. 指标类型只能是"基本数据"或"行为数据"，不能使用其他值
5. 二级指标名称要简洁明了，描述内容要详细说明该指标的内涵

输入说明：
- 原始指标数据：[{indic_type} {second_indic} {second_desc}]
- 目标应用场景：{target_scene}

请确保返回符合格式要求的纯JSON字符串。
"""

SCENES_MAPPING = {
    "creditParty": "信用党建场景",
    "pension": "社区养老场景",
    "creditPunish": "综合性失信联合惩戒场景",
    "publicService": "公共服务守信主体激励场景"
}

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建模板文件的绝对路径
templates = Jinja2Templates(directory=os.path.join(current_dir, '..', 'templates'))

@router.get("/")
async def root(request: Request):
    """
    请求root的时候，会向用户发送一个页面，页面上包含一个文本框，用户可以输入文本，点击提交按钮后，会将文本发送给后端，后端会返回一个相似度分数。
    """
    return templates.TemplateResponse("similarity-form.html", {"request": request})

@router.get("/credit-conversion")
async def credit_conversion(request: Request):
    """
    信用指标转换页面
    """
    return templates.TemplateResponse("credit-conversion.html", {"request": request})

@router.get("/credit-conversion-data.json")
async def credit_conversion_data():
    """
    信用指标转换数据
    """
    import json
    from pathlib import Path

    data_path = Path(__file__).parent.parent / "templates" / "credit-conversion-data.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

@router.post("/convert-credit-indicators")
async def convert_credit_indicators(request_data: dict):
    """
    信用指标转换接口
    请求体包含：target_scene（目标场景）, indicators_data（原始指标数据）
    """
    target_scene = request_data.get("target_scene")
    indicators_data = request_data.get("indicators_data", [])

    if not target_scene:
        return {"error": "target_scene is required"}

    if target_scene not in ["creditParty", "pension", "creditPunish", "publicService"]:
        return {"error": "Invalid target_scene"}

    # 使用DeepSeek API进行智能转换
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return {"error": "DEEPSEEK_API_KEY environment variable not set"}

        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        # 构建输入数据
        indicators_text = []
        for item in indicators_data:
            if len(item) >= 3:
                indic_type, second_indic, second_desc = item[0], item[1], item[2]
                indicators_text.append(f"[{indic_type} {second_indic} {second_desc}]")

        target_scene_name = SCENES_MAPPING.get(target_scene, target_scene)

        # 构建prompt
        user_prompt = CONVERSION_PROMPT.replace("{indic_type}", str(indicators_text)).replace("{target_scene}", target_scene_name)

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": user_prompt}
            ],
            stream=False
        )

        ai_response = response.choices[0].message.content

        try:
            # 预处理AI响应，移除markdown代码块格式
            cleaned_response = ai_response.strip()

            # 如果包含markdown代码块，去掉```json和```标签
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.replace("```json\n", "").replace("\n```", "").strip()
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.replace("```\n", "").replace("\n```", "").strip()

            # 解析清理后的JSON
            result = json.loads(cleaned_response)

            if result.get("error"):
                return {"error": result.get("error"), "reason": result.get("reason", "")}

            return {
                "target_scene": target_scene,
                "converted_data": result.get("converted_data", []),
                "description": get_scene_description(target_scene)
            }

        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse AI response: {str(e)}", "reason": ai_response[:1000]}
        except Exception as e:
            return {"error": f"Unexpected error during parsing: {str(e)}", "reason": ai_response[:1000]}

    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}



def get_scene_description(target_scene):
    """
    获取场景描述
    """
    descriptions = {
        "creditParty": "信用党建是在党的组织和管理中，系统地引入信用评价机制，通过建立完善的信用记录和评估体系，来加强党员和党组织的自律性和公信力。这种做法旨在通过透明公正的评价标准，促进党内外部的责任和诚信行为，提升党的形象和影响力，同时也通过持续的监督和评估，确保党的政策执行和组织活动的高效性和公正性。",
        "pension": "社区养老是一种将养老服务融入居民日常生活环境的服务模式，它依托社区资源，为老年人提供便捷、贴心的养老服务。这种模式强调在老年人熟悉的居住环境中提供生活照料、医疗保健、精神慰藉等一体化服务，通过充分利用社区内的设施、人力资源以及各类社会组织的参与，不仅能有效减轻家庭养老的压力，还有助于老年人保持社会联系和情感交流，提高他们的生活质量和幸福感。本社区养老的指标评价的对象主要是养老服务人员。",
        "creditPunish": "综合性失信联合惩戒用于评估重点群体或社会主体在公共生活各领域的信用表现，通过整合多部门、多场景的失信信息，对违规、失信行为实施联合惩戒，从而促进社会信用体系完善和公共秩序提升。该指标覆盖交通出行、公共文化、社会保障、公共服务及城市管理等关键领域，体现信用约束的综合性、跨领域和实时性。",
        "publicService": "公共服务守信主体激励用于评估社会主体在公共服务各领域的守信行为，通过对遵守法规、规范行为、积极参与公共事务和履行社会责任的个人或组织给予激励，促进社会信用体系建设和公共服务质量提升。该指标覆盖交通出行、文化服务、社会保障、公共安全、生态环保等关键领域，体现守信激励的全面性、可操作性和正向引导作用。"
    }
    return descriptions.get(target_scene, "")
