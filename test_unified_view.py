#!/usr/bin/env python3
"""
测试UnifiedResidentBehaviorViewModel是否能正确读取数据库视图
"""

from sqlmodel import Session, select
from app.database.database import engine
from app.models.resident_credit_score import UnifiedResidentBehaviorViewModel

def test_unified_view():
    """测试统一居民行为视图模型"""
    try:
        with Session(engine) as session:
            # 查询前5条记录
            query = select(UnifiedResidentBehaviorViewModel).limit(5)
            results = session.exec(query).all()

            print(f"成功查询到 {len(results)} 条记录")
            print("-" * 50)

            for i, record in enumerate(results, 1):
                print(f"记录 {i}:")
                print(f"  居民ID: {record.resident_id}")
                print(f"  记录日期: {record.record_date}")
                print(f"  社区服务评分: {record.community_service_score}")
                print(f"  社区服务备注: {record.community_service_note}")
                print(f"  养老服务时长: {record.community_elderly_service_duration}")
                print(f"  养老服务描述: {record.community_elderly_service_description}")
                print(f"  党员活动时长: {record.party_member_duration}")
                print(f"  党员活动描述: {record.party_member_description}")
                print(f"  党员奖惩类型: {record.party_member_reward_punish_type}")
                print(f"  党员奖惩原因: {record.party_member_reward_punish_reason}")
                print(f"  党员缴费金额: {record.party_member_payment_amount}")
                print(f"  党员缴费状态: {record.party_member_payment_status}")
                print(f"  重点人员描述: {record.key_resident_description}")
                print(f"  重点人员评分: {record.key_resident_score}")
                print()

            # 统计总记录数
            count_query = select(UnifiedResidentBehaviorViewModel)
            total_count = len(session.exec(count_query).all())
            print(f"视图总记录数: {total_count}")

            return True

    except Exception as e:
        print(f"测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试UnifiedResidentBehaviorViewModel...")
    success = test_unified_view()
    if success:
        print("✅ 测试通过！模型类可以正确读取视图数据。")
    else:
        print("❌ 测试失败！")
