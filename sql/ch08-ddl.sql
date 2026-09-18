-- ch08: 工具调用审计表(ToolAuditLog 对齐;不挂外键——审计不能被引用约束拦)
SET NAMES utf8mb4;

CREATE TABLE tool_audit_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    conversation_id BIGINT NULL,
    tool_call_id VARCHAR(64) NULL,
    tool_name VARCHAR(128) NOT NULL,
    tool_source ENUM('builtin', 'mcp') NOT NULL,
    mcp_server VARCHAR(64) NULL,
    arguments JSON NULL,
    result_summary TEXT NULL,
    status ENUM('成功', '失败', '超时', '校验拦下', '权限拒绝') NOT NULL,
    error_message VARCHAR(512) NULL,
    retry_count INT NOT NULL DEFAULT 0,
    duration_ms INT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
