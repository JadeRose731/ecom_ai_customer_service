SET NAMES utf8mb4;

CREATE TABLE low_confidence_questions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    conversation_id BIGINT NULL,
    raw_question TEXT NOT NULL,
    source ENUM('retrieval_low_conf', 'self_check', 'user_feedback') NOT NULL,
    reason TEXT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_low_conf_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;

-- 编造个案台账:评估判出的编造答案一题一行,跨轮累计(seen_count),人工处置
CREATE TABLE faith_cases (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    eval_id VARCHAR(64) NOT NULL UNIQUE COMMENT '评估题 id,一题一行',
    query TEXT NOT NULL COMMENT '题目原文',
    answer MEDIUMTEXT NOT NULL COMMENT '被判编造的回答全文',
    citations JSON NULL COMMENT '本轮喂给模型的 Top-K 证据快照(n/问法/正文/section_path/chunk id)',
    status ENUM('unresolved', 'resolved', 'wontfix') NOT NULL DEFAULT 'unresolved',
    resolution VARCHAR(1024) NULL COMMENT '处置说明,已解决与无需解决必填,退回未解决时清空',
    seen_count INT NOT NULL DEFAULT 1 COMMENT '跨轮复发次数',
    last_seen_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at DATETIME NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
