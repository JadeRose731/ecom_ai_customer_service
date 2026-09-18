-- ch09 数据飞轮:待审队列(review_queue)+ 评估轮次(eval_runs)
-- + 问题池两列:召回快照 retrieved_chunks / 查重归并落点 matched_review_id
-- 注:Plan 称本文件「用户手写已提交」,实际仓库缺失;按 Plan Task 1 ORM 定义补写(偏差记 dev-notes)
SET NAMES utf8mb4;

-- 一行 = 一个去重后的知识缺口;查重命中只累加 occurrence_count 不新建行
CREATE TABLE review_queue (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    normalized_question VARCHAR(512) NOT NULL,
    ai_suggested_answer TEXT NULL,
    occurrence_count INT NOT NULL DEFAULT 1,
    review_status ENUM('待审', '通过', '驳回') NOT NULL DEFAULT '待审',
    approved_answer TEXT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_review_status (review_status),
    INDEX idx_review_updated (updated_at)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;

-- 一行 = 一轮自动化评估;metrics JSON 收各指标,按时间连成趋势
CREATE TABLE eval_runs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    triggered_by ENUM('定时', '手动') NOT NULL DEFAULT '定时',
    dataset_size INT NOT NULL,
    metrics JSON NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;

ALTER TABLE low_confidence_questions
    ADD COLUMN retrieved_chunks JSON NULL,
    ADD COLUMN matched_review_id BIGINT NULL,
    ADD INDEX idx_lcq_unmatched (matched_review_id),
    ADD CONSTRAINT fk_lcq_review FOREIGN KEY (matched_review_id) REFERENCES review_queue (id);
