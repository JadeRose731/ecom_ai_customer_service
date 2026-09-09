#!/usr/bin/env bash
# ch02 验收演示:三条验收标准
set -euo pipefail
BASE=http://localhost:8000
echo "== 验收1:订单物流(应选中 query_logistics)=="
curl -s $BASE/api/agent -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","message":"订单 1001 的物流到哪了"}' | python -m json.tool
echo "== 验收2:退货政策(query_faq 命中)=="
curl -s $BASE/api/agent -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","message":"退货政策是什么"}' | python -m json.tool
echo "== 验收3:换说法(query_faq 漏召回,预期结果)=="
curl -s $BASE/api/agent -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","message":"邮费是多少"}' | python -m json.tool
