from fastapi import FastAPI
from models.mock_models import *
import hashlib


app = FastAPI()


@app.post("/warning_by_scene/")
def credit_warning_handler(request_data: CreditWarningRequest):
    """信用值预警接口
    """
    resident_id = 123
    resident_name = "雷文丽"
    credit_score = 68.5
    response_data = [CreditWarningResponse(resident_id=resident_id, resident_name=resident_name, credit_score=credit_score)]
    return response_data

@app.post("/credit_prediction_by_group/")
def credit_prediction_by_group_handler(request_data: CreditPredictionByGroupRequest):
    """分组信用值预测接口
    """
    scene_name = request_data.scene_name
    group_name = request_data.group_name
    prediction_time = request_data.prediction_time
    random_value = calculate_value_string(scene_name, group_name, prediction_time)
    credit_score = 85.5 + random_value
    response_data = CreditPredictionByGroupResponse(credit_score=credit_score)
    return response_data

@app.post("/credit_prediction_by_resident/")
def credit_prediction_by_resident_handler(request_data: CreditPredictionRequest):
    """个人信用值预测接口
    """
    user_id = request_data.resident_id
    scene_name = request_data.scene_name
    prediction_time = request_data.prediction_time
    random_value = calculate_value(user_id, scene_name, prediction_time)
    credit_score = 85.5 + random_value
    response_data = CreditPredictionResponse(credit_score=credit_score)
    return response_data

@app.post("/perception")
def perception_handler(request_data: CurrentSceneRequest):
    """势态感知的接口，根据场景和一个时间，返回一个势态感知的字典
    """
    if request_data.scene_name == SceneName.SHE_QU:
        response_data = ScenePerceptionResponse(
            current_scene=SceneName.SHE_QU,
            scene_perception={
                "吸毒人员": -0.10,
                "刑满释放人员": 0,
                "精神病人员": 0,
                "低保人员": 0,
                "社区矫正": 0,
                "空巢老人": 0,
                "五保人员": 0.10,
                "残疾人": 0.12,
                "重大疾病致困人员": 0,
                "邪教人员": 0
            }
        )
    elif request_data.scene_name == SceneName.DANG_JIAN:
        response_data = ScenePerceptionResponse(
            current_scene=SceneName.DANG_JIAN,
            scene_perception={
                "党政机关党员": 0.10,
                "国有企业党员": 0.12,
                "民营企业党员": 0,
                "事业单位党员": 0.12,
                "社会组织党员": 0,
                "两新组织党员": 0.18,
                "村社基层组织党员": 0,
                "其他": 0
            }
        )
    elif request_data.scene_name == SceneName.YANG_LAO:
        response_data = ScenePerceptionResponse(
            current_scene=SceneName.YANG_LAO,
            scene_perception={
                "独居老人": 0,
                "高龄老人": 0,
                "伤残老人": 0.01,
                "困难老人": 0,
                "社区养老服务人员": 0
            }
        )
    return response_data


def calculate_value(int_value: int, string_value: str, datetime_value: datetime) -> int:
    # 将int_value和string_value转换为字节串
    int_bytes = int_value.to_bytes(4, 'big')
    string_bytes = string_value.encode('utf-8')
    
    # 计算md5散列值
    hash_input = int_bytes + string_bytes + datetime_value.isoformat().encode('utf-8')
    hash_value = hashlib.md5(hash_input).hexdigest()
    
    # 将md5散列值转换为整数
    hash_int = int(hash_value, 16)
    
    # 取余并映射到0～5的范围
    result = hash_int % 6
    
    return result

def calculate_value_string(string_value1: str, string_value2: str, datetime_value: datetime) -> int:
    # 将int_value和string_value转换为字节串
    string_bytes1 = string_value1.encode('utf-8')
    string_bytes2 = string_value2.encode('utf-8')
    
    # 计算md5散列值
    hash_input = string_bytes1 + string_bytes2 + datetime_value.isoformat().encode('utf-8')
    hash_value = hashlib.md5(hash_input).hexdigest()
    
    # 将md5散列值转换为整数
    hash_int = int(hash_value, 16)
    
    # 取余并映射到0～5的范围
    result = hash_int % 6
    
    return result