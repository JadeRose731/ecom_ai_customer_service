-- ch10 主题分类器:归类结果表
-- 注:Plan 称本文件「开发库应用 DDL」但仓库缺失;按 Plan Task 2 ORM 定义补写(偏差记 dev-notes/ch10.md)
SET NAMES utf8mb4;

-- 一行 = 一条问题的一次归类;labels 存命中类目名数组(多标签);已归类的行不重复归(幂等靠 LEFT JOIN 判)
CREATE TABLE topic_classifications (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    question_id BIGINT NOT NULL,
    labels JSON NOT NULL,
    classified_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_topic_question FOREIGN KEY (question_id) REFERENCES low_confidence_questions (id),
    INDEX idx_topic_question (question_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
