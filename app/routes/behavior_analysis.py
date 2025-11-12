from fastapi import APIRouter, Request, HTTPException
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, text
from app.database.database import engine
from datetime import datetime, timedelta
from typing import List, Dict
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

router = APIRouter()

# 获取当前文件的目录
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(current_dir, '..', 'templates'))

@router.get("/behavior-analysis")
async def behavior_analysis_page(request: Request):
    """
    行为分析页面
    """
    return templates.TemplateResponse("behavior-analysis.html", {"request": request})

@router.get("/api/behavior-stats")
async def get_behavior_stats(start_date: str = None, end_date: str = None):
    """
    获取时间范围内的用户数和行为记录数统计
    """
    try:
        # 如果没有提供日期，默认最近3天
        if not start_date or not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

        query = f"""
        SELECT
            COUNT(DISTINCT resident_id) as user_count,
            COUNT(*) as record_count
        FROM unified_resident_behavior_view
        WHERE record_date BETWEEN '{start_date}' AND '{end_date}'
        """

        with Session(engine) as session:
            result = session.exec(text(query)).first()

        return {
            "user_count": result[0] if result else 0,
            "record_count": result[1] if result else 0,
            "start_date": start_date,
            "end_date": end_date
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

# 全局变量存储最新的聚类结果
latest_clustering_result = None

@router.post("/api/cluster-analysis")
async def cluster_analysis(request_data: dict):
    """
    基于resident_id进行聚类分析
    """
    global latest_clustering_result

    try:
        start_date = request_data.get("start_date")
        end_date = request_data.get("end_date")
        n_clusters = request_data.get("n_clusters", 6)

        if not start_date or not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

        # 查询数据，包含文本字段的聚合信息
        query = f"""
        SELECT
            resident_id,
            COUNT(*) as activity_count,
            SUM(CASE WHEN community_service_score IS NOT NULL THEN community_service_score ELSE 0 END) as total_service_score,
            SUM(CASE WHEN community_elderly_service_duration IS NOT NULL THEN community_elderly_service_duration ELSE 0 END) as total_elderly_service_duration,
            SUM(CASE WHEN party_member_duration IS NOT NULL THEN party_member_duration ELSE 0 END) as total_party_duration,
            SUM(CASE WHEN key_resident_score IS NOT NULL THEN key_resident_score ELSE 0 END) as total_key_score,
            -- 奖惩类型编码：奖励=1, 处罚=-1, 其他=0
            SUM(CASE
                WHEN party_member_reward_punish_type = '奖励' THEN 1
                WHEN party_member_reward_punish_type = '处罚' THEN -1
                ELSE 0
            END) as reward_punish_score,
            -- 奖惩原因文本长度（简单文本特征）
            AVG(CASE WHEN party_member_reward_punish_reason IS NOT NULL THEN LENGTH(party_member_reward_punish_reason) ELSE 0 END) as avg_reason_length,
            -- 重点人员描述文本长度
            AVG(CASE WHEN key_resident_description IS NOT NULL THEN LENGTH(key_resident_description) ELSE 0 END) as avg_description_length
        FROM unified_resident_behavior_view
        WHERE record_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY resident_id
        HAVING activity_count > 0
        """

        with Session(engine) as session:
            results = session.exec(text(query)).all()

        if not results:
            return {"error": "没有找到符合条件的数据"}

        # 转换为DataFrame
        data = []
        resident_ids = []
        for row in results:
            resident_ids.append(row[0])
            data.append([
                row[1],  # activity_count
                row[2],  # total_service_score
                row[3],  # total_elderly_service_duration
                row[4],  # total_party_duration
                row[5],  # total_key_score
                row[6],  # reward_punish_score
                row[7],  # avg_reason_length
                row[8]   # avg_description_length
            ])

        df = pd.DataFrame(data, columns=['activity_count', 'service_score', 'elderly_duration', 'party_duration', 'key_score', 'reward_punish_score', 'reason_length', 'description_length'])

        # 数据标准化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)

        # 统计每个簇的样本数
        cluster_counts = {}
        cluster_details = {}
        resident_cluster_map = {}

        for i in range(n_clusters):
            cluster_mask = clusters == i
            cluster_residents = [resident_ids[j] for j in range(len(resident_ids)) if cluster_mask[j]]
            cluster_counts[f"簇{i+1}"] = len(cluster_residents)
            cluster_details[f"簇{i+1}"] = cluster_residents

            # 记录每个居民属于哪个簇
            for resident_id in cluster_residents:
                resident_cluster_map[resident_id] = i + 1

        # 计算簇中心（反标准化）
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

        # 保存聚类结果到全局变量
        latest_clustering_result = {
            "start_date": start_date,
            "end_date": end_date,
            "n_clusters": n_clusters,
            "resident_cluster_map": resident_cluster_map,
            "cluster_counts": cluster_counts,
            "cluster_details": cluster_details
        }

        return {
            "cluster_counts": cluster_counts,
            "cluster_details": cluster_details,
            "cluster_centers": {
                f"簇{i+1}": {
                    "activity_count": round(cluster_centers[i][0], 2),
                    "service_score": round(cluster_centers[i][1], 2),
                    "elderly_duration": round(cluster_centers[i][2], 2),
                    "party_duration": round(cluster_centers[i][3], 2),
                    "key_score": round(cluster_centers[i][4], 2),
                    "reward_punish_score": round(cluster_centers[i][5], 2),
                    "reason_length": round(cluster_centers[i][6], 2),
                    "description_length": round(cluster_centers[i][7], 2)
                }
                for i in range(n_clusters)
            },
            "total_residents": len(resident_ids),
            "n_clusters": n_clusters
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聚类分析失败: {str(e)}")

@router.get("/cluster-detail/{cluster_id}")
async def cluster_detail_page(cluster_id: int, request: Request):
    """
    簇详情页面
    """
    return templates.TemplateResponse("cluster-detail.html", {
        "request": request,
        "cluster_id": cluster_id
    })

@router.get("/api/cluster-detail/{cluster_id}")
async def get_cluster_detail(cluster_id: int, page: int = 1, per_page: int = 50):
    """
    获取簇详情数据
    """
    global latest_clustering_result

    if not latest_clustering_result:
        raise HTTPException(status_code=400, detail="请先执行聚类分析")

    try:
        cluster_name = f"簇{cluster_id}"
        if cluster_name not in latest_clustering_result["cluster_details"]:
            raise HTTPException(status_code=404, detail=f"簇{cluster_id}不存在")

        resident_ids = latest_clustering_result["cluster_details"][cluster_name]
        start_date = latest_clustering_result["start_date"]
        end_date = latest_clustering_result["end_date"]

        # 分页处理
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_resident_ids = resident_ids[start_idx:end_idx]

        if not page_resident_ids:
            return {
                "data": [],
                "total_count": len(resident_ids),
                "page": page,
                "per_page": per_page,
                "total_pages": (len(resident_ids) + per_page - 1) // per_page
            }

        # 查询这些居民的详细数据
        resident_ids_str = ','.join(map(str, page_resident_ids))
        query = f"""
        SELECT
            resident_id,
            record_date,
            community_service_score,
            community_service_note,
            community_elderly_service_duration,
            community_elderly_service_description,
            party_member_duration,
            party_member_description,
            party_member_reward_punish_type,
            party_member_reward_punish_reason,
            party_member_payment_amount,
            party_member_payment_status,
            key_resident_description,
            key_resident_score
        FROM unified_resident_behavior_view
        WHERE resident_id IN ({resident_ids_str})
        AND record_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY resident_id, record_date
        """

        with Session(engine) as session:
            results = session.exec(text(query)).all()

        # 转换为字典列表
        data = []
        for row in results:
            data.append({
                "resident_id": row[0],
                "record_date": str(row[1]) if row[1] else None,
                "community_service_score": float(row[2]) if row[2] else None,
                "community_service_note": row[3],
                "community_elderly_service_duration": float(row[4]) if row[4] else None,
                "community_elderly_service_description": row[5],
                "party_member_duration": float(row[6]) if row[6] else None,
                "party_member_description": row[7],
                "party_member_reward_punish_type": row[8],
                "party_member_reward_punish_reason": row[9],
                "party_member_payment_amount": float(row[10]) if row[10] else None,
                "party_member_payment_status": row[11],
                "key_resident_description": row[12],
                "key_resident_score": float(row[13]) if row[13] else None
            })

        return {
            "data": data,
            "total_count": len(resident_ids),
            "page": page,
            "per_page": per_page,
            "total_pages": (len(resident_ids) + per_page - 1) // per_page,
            "cluster_id": cluster_id,
            "cluster_name": cluster_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取簇详情失败: {str(e)}")
