SET NAMES utf8mb4;

CREATE TABLE knowledge_chunks (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    category VARCHAR(255) NOT NULL,
    questions TEXT NOT NULL,
    answer TEXT NOT NULL,
    section_path VARCHAR(512) NULL,
    content_type VARCHAR(32) NULL,
    is_key_clause INT NOT NULL DEFAULT 0,
    prev_chunk_id BIGINT NULL,
    next_chunk_id BIGINT NULL,
    vector_id VARCHAR(64) NULL,
    vectorize_status ENUM('pending', 'done') NOT NULL DEFAULT 'pending',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_knowledge_chunks_prev FOREIGN KEY (prev_chunk_id) REFERENCES knowledge_chunks (id),
    CONSTRAINT fk_knowledge_chunks_next FOREIGN KEY (next_chunk_id) REFERENCES knowledge_chunks (id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;

CREATE TABLE qa_extraction_staging (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    batch_no VARCHAR(64) NOT NULL,
    source_ref VARCHAR(255) NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    status ENUM('extracted', 'kept', 'discarded') NOT NULL DEFAULT 'extracted',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
