-- ch03 · 合成历史客服对话(喂挖知识 job)。三段式 + 字面量。
SET NAMES utf8mb4;

SELECT COUNT(*) AS before_conv FROM conversations;

INSERT INTO conversations (user_id, status) VALUES ('seed-u1', '已结束'), ('seed-u2', '已结束');
SET @c1 = (SELECT id FROM conversations WHERE user_id='seed-u1' ORDER BY id DESC LIMIT 1);
SET @c2 = (SELECT id FROM conversations WHERE user_id='seed-u2' ORDER BY id DESC LIMIT 1);

INSERT INTO messages (conversation_id, role, content) VALUES
  (@c1, 'user', '你们发货一般多久啊'),
  (@c1, 'assistant', '现货商品付款后48小时内发货,预售以详情页为准。'),
  (@c2, 'user', '满多少包邮'),
  (@c2, 'assistant', '单笔订单满99元包邮,未满收取10元运费。');

SELECT COUNT(*) AS after_conv FROM conversations;
