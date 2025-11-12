CREATE VIEW unified_resident_behavior_view AS
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
FROM (
    -- 平安社区服务记录
    SELECT 
        provider_id AS resident_id,
        service_date AS record_date,
        service_score AS community_service_score,
        notes AS community_service_note,
        NULL AS community_elderly_service_duration,
        NULL AS community_elderly_service_description,
        NULL AS party_member_duration,
        NULL AS party_member_description,
        NULL AS party_member_reward_punish_type,
        NULL AS party_member_reward_punish_reason,
        NULL AS party_member_payment_amount,
        NULL AS party_member_payment_status,
        NULL AS key_resident_description,
        NULL AS key_resident_score
    FROM community_service_record
    WHERE resident_id IS NOT NULL
    
    UNION ALL
    
    -- 社区养老服务记录
    SELECT 
        resident_id,
        evaluation_date AS record_date,
        NULL AS community_service_score,
        NULL AS community_service_note,
        evaluation_duration AS community_elderly_service_duration,
        description AS community_elderly_service_description,
        NULL AS party_member_duration,
        NULL AS party_member_description,
        NULL AS party_member_reward_punish_type,
        NULL AS party_member_reward_punish_reason,
        NULL AS party_member_payment_amount,
        NULL AS party_member_payment_status,
        NULL AS key_resident_description,
        NULL AS key_resident_score
    FROM community_elderly_service_record
    
    UNION ALL
    
    -- 党员活动参与记录
    SELECT 
        resident_id,
        add_day AS record_date,
        NULL AS community_service_score,
        NULL AS community_service_note,
        NULL AS community_elderly_service_duration,
        NULL AS community_elderly_service_description,
        duration AS party_member_duration,
        description AS party_member_description,
        NULL AS party_member_reward_punish_type,
        NULL AS party_member_reward_punish_reason,
        NULL AS party_member_payment_amount,
        NULL AS party_member_payment_status,
        NULL AS key_resident_description,
        NULL AS key_resident_score
    FROM party_member_activities p1
    
    UNION ALL
    
    -- 党员奖惩记录
    SELECT 
        resident_id,
        date_issued AS record_date,
        NULL AS community_service_score,
        NULL AS community_service_note,
        NULL AS community_elderly_service_duration,
        NULL AS community_elderly_service_description,
        NULL AS party_member_duration,
        NULL AS party_member_description,
        type AS party_member_reward_punish_type,
        reason AS party_member_reward_punish_reason,
        NULL AS party_member_payment_amount,
        NULL AS party_member_payment_status,
        NULL AS key_resident_description,
        NULL AS key_resident_score
    FROM party_member_rewards_punishments
    
    UNION ALL
    
    -- 党员党费缴纳记录
    SELECT 
        resident_id,
        payment_date AS record_date,
        NULL AS community_service_score,
        NULL AS community_service_note,
        NULL AS community_elderly_service_duration,
        NULL AS community_elderly_service_description,
        NULL AS party_member_duration,
        NULL AS party_member_description,
        NULL AS party_member_reward_punish_type,
        NULL AS party_member_reward_punish_reason,
        payment_amount AS party_member_payment_amount,
        payment_status AS party_member_payment_status,
        NULL AS key_resident_description,
        NULL AS key_resident_score
    FROM party_member_payments
    
    UNION ALL
    
    -- 重点人员行为记录
    SELECT 
        resident_id,
        last_active_date AS record_date,
        NULL AS community_service_score,
        NULL AS community_service_note,
        NULL AS community_elderly_service_duration,
        NULL AS community_elderly_service_description,
        NULL AS party_member_duration,
        NULL AS party_member_description,
        NULL AS party_member_reward_punish_type,
        NULL AS party_member_reward_punish_reason,
        NULL AS party_member_payment_amount,
        NULL AS party_member_payment_status,
        Description AS key_resident_description,
        score AS key_resident_score
    FROM key_residents_active
) AS unified_data
ORDER BY resident_id, record_date;