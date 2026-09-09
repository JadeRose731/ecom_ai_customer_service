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
