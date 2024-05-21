from datetime import date, datetime
from typing import Optional
from sqlmodel import SQLModel, Field, Column, Integer, Float, String, Date, Text

"""
从不同的表中，提取用户的信用分记录，以及行为记录
"""

"""
# 1. 用户信用分记录表，记录了每个用户每一天的信用分，一年两百多万条数据，这有点大，慎重

目前只有 account_id = 2 的5000个用户的数据
用户的身份ID是 resident_id

SHOW CREATE TABLE  resident_credit_score_t;
——————————————————————————————————————————————————————————————
    CREATE TABLE `resident_credit_score_t` (
    `id` int unsigned NOT NULL AUTO_INCREMENT,
    `account_id` int unsigned NOT NULL COMMENT '账户id',
    `primary_id` int NOT NULL COMMENT '一级指标id',
    `create_time` datetime DEFAULT NULL,
    `update_time` datetime DEFAULT NULL,
    `day` date DEFAULT NULL COMMENT '生成日期',
    `score` decimal(10,2) DEFAULT NULL COMMENT '得分',
    `resident_id` int unsigned NOT NULL COMMENT '居民主键id',
    `reason` varchar(255) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '变动原因',
    `credit_assessment` varchar(255) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '信用风险评估',
    `last_score` decimal(10,2) DEFAULT NULL COMMENT '上次得分',
    PRIMARY KEY (`id`) USING BTREE
    ) ENGINE=InnoDB AUTO_INCREMENT=3178770 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='用户评分表'
"""
# 定义模型类 用户信用分
class ResidentCreditScore(SQLModel, table=True):
    __tablename__ = "resident_credit_score_t"

    id: int = Field(default=None, primary_key=True)
    primary_id: int = Field(nullable=False)
    day: Optional[date] = Field(default=None)
    score: Optional[float] = Field(default=None)
    resident_id: int = Field(nullable=False)
    reason: Optional[str] = Field(default=None)
    credit_assessment: Optional[str] = Field(default=None)
    last_score: Optional[float] = Field(default=None)


"""
# 2. 平安社区服务记录表
SHOW CREATE TABLE  community_service_record;
——————————————————————————————————————————————————————————————
    CREATE TABLE `community_service_record` (
    `service_record_id` int NOT NULL AUTO_INCREMENT COMMENT '服务记录ID',
    `service_type_id` int NOT NULL COMMENT '服务类型ID，关联comunity_service_type表的ID',
    `resident_id` int DEFAULT NULL COMMENT '被服务者ID，关联residents_basic_info表的ID',
    `provider_id` int NOT NULL COMMENT '服务者ID，关联residents_basic_info表的ID',
    `service_date` date NOT NULL COMMENT '服务提供的日期',
    `service_score` tinyint DEFAULT NULL COMMENT '服务的评分(0-10分，整数)',
    `notes` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci COMMENT '其他记录或备注',
    `add_day` date NOT NULL COMMENT '添加日期',
    PRIMARY KEY (`service_record_id`) USING BTREE,
    KEY `resident_id` (`resident_id`) USING BTREE,
    KEY `provider_id` (`provider_id`) USING BTREE,
    CONSTRAINT `community_service_record_ibfk_2` FOREIGN KEY (`resident_id`) REFERENCES `residents_basic_info` (`resident_id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
    CONSTRAINT `community_service_record_ibfk_3` FOREIGN KEY (`provider_id`) REFERENCES `residents_basic_info` (`resident_id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
    CONSTRAINT `chk_service_score` CHECK (((`service_score` >= 0) and (`service_score` <= 10)))
    ) ENGINE=InnoDB AUTO_INCREMENT=113401 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci ROW_FORMAT=DYNAMIC COMMENT='平安社区服务类型表'
"""
# 定义模型类 社区服务记录
class CommunityServiceRecordModel(SQLModel, table=True):
    __tablename__ = "community_service_record"

    service_record_id: Optional[int] = Field(default=None, primary_key=True)
    resident_id: Optional[int] = Field(default=None, foreign_key="residents_basic_info.resident_id")
    service_date: date = Field(nullable=False)
    service_score: Optional[int] = Field(default=None)

"""
# 3. 社区养老服务记录表
SHOW CREATE TABLE community_elderly_service_record;
——————————————————————————————————————————————————————————————
    CREATE TABLE `community_elderly_service_record` (
    `evaluation_record_id` int NOT NULL AUTO_INCREMENT,
    `resident_id` int NOT NULL COMMENT '被评价者ID，与residents_basic_info表关联的ID',
    `evaluation_date` date NOT NULL COMMENT '评价日期',
    `evaluator_id` int NOT NULL COMMENT '评价者ID，与residents_basic_info表关联的ID',
    `evaluation_duration` decimal(10,1) NOT NULL COMMENT '该项相关服务持续的时间(单位：小时),不足半小时按半小时计算',
    `caregiving_knowledge` tinyint(1) DEFAULT '5' COMMENT '照顾工作常识(1-5分)',
    `nursing_knowledge` tinyint(1) DEFAULT '5' COMMENT '护理专业常识(1-5分)',
    `life_care_ability` tinyint(1) DEFAULT '5' COMMENT '生活照顾能力(1-5分)',
    `basic_nursing_ability` tinyint(1) DEFAULT '5' COMMENT '基础护理能力(1-5分)',
    `specialized_nursing_ability` tinyint(1) DEFAULT '5' COMMENT '专科护理能力(1-5分)',
    `cultural_nursing_ability` tinyint(1) DEFAULT '5' COMMENT '社会文化护理能力(1-5分)',
    `professional_skills` tinyint(1) DEFAULT '5' COMMENT '职业基本技能(1-5分)',
    `professional_attitude` tinyint(1) DEFAULT '5' COMMENT '职业态度(1-5分)',
    `personal_characteristics` tinyint(1) DEFAULT '5' COMMENT '人格特点(1-5分)',
    `description` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '评价描述或备注',
    `add_day` date NOT NULL COMMENT '添加日期',
    PRIMARY KEY (`evaluation_record_id`) USING BTREE,
    KEY `resident_id` (`resident_id`) USING BTREE,
    KEY `evaluator_id` (`evaluator_id`) USING BTREE,
    KEY `evaluation_record_id` (`evaluation_record_id`) USING BTREE,
    CONSTRAINT `community_elderly_service_record_ibfk_2` FOREIGN KEY (`evaluator_id`) REFERENCES `residents_basic_info` (`resident_id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
    CONSTRAINT `community_elderly_service_record_chk_1` CHECK ((`caregiving_knowledge` in (_utf8mb4'1',_utf8mb4'2',_utf8mb4'3',_utf8mb4'4',_utf8mb4'5'))),
    CONSTRAINT `community_elderly_service_record_chk_2` CHECK ((`nursing_knowledge` in (_utf8mb4'1',_utf8mb4'2',_utf8mb4'3',_utf8mb4'4',_utf8mb4'5'))),
    CONSTRAINT `community_elderly_service_record_chk_3` CHECK ((`life_care_ability` in (_utf8mb4'1',_utf8mb4'2',_utf8mb4'3',_utf8mb4'4',_utf8mb4'5'))),
    CONSTRAINT `community_elderly_service_record_chk_4` CHECK ((`basic_nursing_ability` in (_utf8mb4'1',_utf8mb4'2',_utf8mb4'3',_utf8mb4'4',_utf8mb4'5'))),
    CONSTRAINT `community_elderly_service_record_chk_5` CHECK ((`specialized_nursing_ability` in (_utf8mb4'1',_utf8mb4'2',_utf8mb4'3',_utf8mb4'4',_utf8mb4'5'))),
    CONSTRAINT `community_elderly_service_record_chk_6` CHECK ((`cultural_nursing_ability` in (_utf8mb4'1',_utf8mb4'2',_utf8mb4'3',_utf8mb4'4',_utf8mb4'5'))),
    CONSTRAINT `community_elderly_service_record_chk_7` CHECK ((`professional_skills` in (_utf8mb4'1',_utf8mb4'2',_utf8mb4'3',_utf8mb4'4',_utf8mb4'5'))),
    CONSTRAINT `community_elderly_service_record_chk_8` CHECK ((`professional_attitude` in (_utf8mb4'1',_utf8mb4'2',_utf8mb4'3',_utf8mb4'4',_utf8mb4'5'))),
    CONSTRAINT `community_elderly_service_record_chk_9` CHECK ((`personal_characteristics` in (_utf8mb4'1',_utf8mb4'2',_utf8mb4'3',_utf8mb4'4',_utf8mb4'5')))
    ) ENGINE=InnoDB AUTO_INCREMENT=75601 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='社区养老服务记录表'
"""
#定义模型类 社区养老服务记录
class ElderlyServiceRecordModel(SQLModel, table=True):
    __tablename__ = "community_elderly_service_record"

    evaluation_record_id: Optional[int] = Field(default=None, primary_key=True)
    resident_id: int = Field(foreign_key="residents_basic_info.resident_id")
    evaluation_date: date = Field(nullable=False)
    evaluator_id: int = Field(foreign_key="residents_basic_info.resident_id")
    evaluation_duration: float = Field(nullable=False)
    caregiving_knowledge: int = Field(default=5)
    nursing_knowledge: int = Field(default=5)
    life_care_ability: int = Field(default=5)
    basic_nursing_ability: int = Field(default=5)
    specialized_nursing_ability: int = Field(default=5)
    cultural_nursing_ability: int = Field(default=5)
    professional_skills: int = Field(default=5)
    professional_attitude: int = Field(default=5)
    personal_characteristics: int = Field(default=5)

"""
# 4. 党员活动参与情况表
SHOW CREATE TABLE party_member_activities;
————————————————————————————————————————————————————
    CREATE TABLE `party_member_activities` (
    `id` int NOT NULL AUTO_INCREMENT COMMENT '记录ID',
    `activity_id` int NOT NULL COMMENT '与party_activities表关联的ID',
    `resident_id` int NOT NULL COMMENT '与residents_basic_info表关联的ID',
    `duration` int DEFAULT NULL COMMENT '参与时长（单位：分钟）',
    `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci COMMENT '活动描述或备注',
    `add_day` date DEFAULT NULL COMMENT '添加日期',
    PRIMARY KEY (`id`) USING BTREE,
    KEY `activity_id` (`activity_id`) USING BTREE,
    KEY `party_member_activities_ibfk_2` (`resident_id`) USING BTREE,
    KEY `id` (`id`) USING BTREE,
    CONSTRAINT `party_member_activities_ibfk_1` FOREIGN KEY (`activity_id`) REFERENCES `party_activities` (`activity_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
    ) ENGINE=InnoDB AUTO_INCREMENT=76993 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci ROW_FORMAT=DYNAMIC COMMENT='党员活动参与情况表'
"""
#定义模型类 党员活动参与情况
class PartyMemberActivitiesModel(SQLModel, table=True):
    __tablename__ = "party_member_activities"

    id: Optional[int] = Field(default=None, primary_key=True)
    activity_id: int = Field(nullable=False, foreign_key="party_activities.activity_id")
    resident_id: int = Field(nullable=False, foreign_key="residents_basic_info.resident_id")
    duration: Optional[int] = Field(default=None)
    add_day: Optional[date] = Field(default=None)

"""
# 5. 党员奖惩记录表 
SHOW CREATE TABLE party_member_rewards_punishments;
——————————————————————————————————————————————————————
    CREATE TABLE `party_member_rewards_punishments` (
    `reward_punish_id` int NOT NULL AUTO_INCREMENT COMMENT '奖惩记录ID',
    `resident_id` int NOT NULL COMMENT '与residents_basic_info表关联的ID',
    `type` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL COMMENT '类型（奖励/处罚）',
    `date_issued` date NOT NULL COMMENT '颁发日期',
    `reason` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL COMMENT '原因或依据',
    `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci COMMENT '描述或备注',
    `add_day` date NOT NULL COMMENT '添加日期',
    PRIMARY KEY (`reward_punish_id`) USING BTREE,
    KEY `party_member_rewards_punishments_ibfk_1` (`resident_id`) USING BTREE,
    KEY `reward_punish_id` (`reward_punish_id`) USING BTREE,
    CONSTRAINT `party_member_rewards_punishments_ibfk_1` FOREIGN KEY (`resident_id`) REFERENCES `residents_basic_info` (`resident_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
    ) ENGINE=InnoDB AUTO_INCREMENT=75601 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci ROW_FORMAT=DYNAMIC COMMENT='党员奖惩记录表'
"""
#定义模型类 党员奖惩记录
class PartyMemberRewardsPunishmentsModel(SQLModel, table=True):
    __tablename__ = "party_member_rewards_punishments"

    reward_punish_id: Optional[int] = Field(default=None, primary_key=True)
    resident_id: int = Field(nullable=False, foreign_key="residents_basic_info.resident_id")
    type: str = Field(nullable=False)
    date_issued: date = Field(nullable=False)
    reason: str = Field(nullable=False)

"""
# 6. 重点人员行为记录表
SHOW CREATE TABLE key_residents_active;
——————————————————————————————————————————————
    CREATE TABLE `key_residents_active` (
    `active_id` int NOT NULL AUTO_INCREMENT COMMENT '记录id',
    `resident_id` int NOT NULL COMMENT '重点人员id，关联residents_basic_info表的ID',
    `last_active_date` date NOT NULL COMMENT '上次事件发生的日期/上次探访发生的日期',
    `recorder_id` int NOT NULL COMMENT '记录员id,关联residents_basic_info表的ID',
    `Description` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL COMMENT '情况描述',
    `add_day` date NOT NULL COMMENT '添加日期',
    PRIMARY KEY (`active_id`) USING BTREE,
    KEY `key_residents_active_ibfk_1` (`resident_id`) USING BTREE,
    KEY `recorder_id` (`recorder_id`) USING BTREE,
    KEY `active_id` (`active_id`) USING BTREE,
    CONSTRAINT `key_residents_active_ibfk_1` FOREIGN KEY (`resident_id`) REFERENCES `residents_basic_info` (`resident_id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
    CONSTRAINT `key_residents_active_ibfk_2` FOREIGN KEY (`recorder_id`) REFERENCES `residents_basic_info` (`resident_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
    ) ENGINE=InnoDB AUTO_INCREMENT=148801 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci ROW_FORMAT=DYNAMIC COMMENT='重点人员行为记录表'
"""
#定义模型类 重点人员行为记录
class KeyResidentsActiveModel(SQLModel, table=True):
    __tablename__ = "key_residents_active"

    active_id: Optional[int] = Field(default=None, primary_key=True)
    resident_id: int = Field(nullable=False, foreign_key="residents_basic_info.resident_id")
    last_active_date: date = Field(nullable=False)
    description: str = Field(nullable=False)

"""
# 7. 党员党费缴纳记录表
SHOW CREATE TABLE party_member_payments;
————————————————————————————————————————————————
    CREATE TABLE `party_member_payments` (
    `payment_id` int NOT NULL AUTO_INCREMENT COMMENT '缴费记录ID',
    `resident_id` int NOT NULL COMMENT '与residents_basic_info表关联的ID',
    `payment_date` date NOT NULL COMMENT '缴费日期',
    `payment_amount` decimal(10,2) NOT NULL COMMENT '缴费金额',
    `payment_type` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '缴费方式（如：现金、转账等）',
    `payment_status` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '缴纳状态(''正常缴纳'', ''逾期'', ''未缴纳'')',
    `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci COMMENT '备注信息',
    `add_date` date NOT NULL COMMENT '添加日期',
    `payment_period` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL COMMENT '缴费周期（如：哪一个月度的党费）',
    PRIMARY KEY (`payment_id`) USING BTREE,
    KEY `party_member_payments_ibfk_1` (`resident_id`) USING BTREE,
    CONSTRAINT `party_member_payments_ibfk_1` FOREIGN KEY (`resident_id`) REFERENCES `residents_basic_info` (`resident_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
    ) ENGINE=InnoDB AUTO_INCREMENT=7359 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci ROW_FORMAT=DYNAMIC COMMENT='党员党费缴纳记录表'
"""
#定义模型类 党员党费缴纳记录
class PartyMemberPaymentsModel(SQLModel, table=True):
    __tablename__ = "party_member_payments"

    payment_id: Optional[int] = Field(default=None, primary_key=True)
    resident_id: int = Field(nullable=False, foreign_key="residents_basic_info.resident_id")
    payment_date: date = Field(nullable=False)
    payment_amount: float = Field(nullable=False)
    payment_status: Optional[str] = Field(default=None)

"""
CREATE TABLE `resident_credit_trend_t` (
  `id` int unsigned NOT NULL AUTO_INCREMENT,
  `resident_id` int NOT NULL COMMENT '用户id',
  `primary_id` int NOT NULL COMMENT '一级指标id',
  `account_id` int NOT NULL COMMENT '账户id',
  `score` decimal(10,2) NOT NULL COMMENT '预测分数',
  `create_time` datetime DEFAULT NULL,
  `update_time` datetime NOT NULL,
  `day` date NOT NULL COMMENT '预测日期',
  `current_score` decimal(10,2) DEFAULT NULL COMMENT '当前信用分数',
  `reason` varchar(11) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '分数变动原因',
  PRIMARY KEY (`id`,`resident_id`) USING BTREE,
  KEY `primary_id` (`primary_id`) USING BTREE,
  KEY `account_id` (`account_id`) USING BTREE,
  KEY `day` (`day`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='个人信用预测记录表'
"""
# 定义模型类 居民信用评分预测表
class ResidentCreditTrendModel(SQLModel, table=True):
    __tablename__ = "resident_credit_trend_t"
    id: Optional[int] = Field(sa_column=Column(Integer, primary_key=True, autoincrement=True))
    resident_id: int = Field(primary_key=True)
    primary_id: int
    account_id: int
    score: Optional[float] = Field(default=None)
    create_time: Optional[datetime] = Field(default=None)
    update_time: datetime = Field(..., nullable=False)
    day: date = Field(nullable=False)
    current_score: Optional[float] = Field(default=None)
    reason: Optional[str]