-- ch07 会话上下文管理:conversations 增加滚动摘要两列
-- summary:早期轮次的滚动摘要文本(后台异步生成,成功后原子更新)
-- summary_upto_msg_id:摘要覆盖到的 messages.id 边界,滑窗从其后接原文
SET NAMES utf8mb4;
ALTER TABLE conversations
    ADD COLUMN summary TEXT NULL,
    ADD COLUMN summary_upto_msg_id BIGINT NULL;
