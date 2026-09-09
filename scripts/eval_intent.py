"""意图分类标注评估:核对 classify 七类判对率。需上游可用。
用法:.venv/Scripts/python.exe -m scripts.eval_intent"""
import asyncio

from app.core.intent import classify

# (用户问法, 期望意图)——七类各若干条
SAMPLES = [
    ("订单1001的快递到哪了", "物流"),
    ("我买的东西发货了吗", "物流"),
    ("订单2002现在什么状态", "订单"),
    ("我上周下的单多少钱来着", "订单"),
    ("这款猫粮多少钱一包", "商品咨询"),
    ("智能猫砂盆怎么用啊", "商品咨询"),
    ("我想退货", "退款退货"),
    ("退款一般几天到账", "退款退货"),
    ("我的猫爬架坏了能保修吗", "售后"),
    ("换货进度到哪了", "售后"),
    ("你们这什么破服务,我要投诉", "投诉"),
    ("太差了给我个说法", "投诉"),
    ("你好呀", "闲聊"),
    ("今天天气不错", "闲聊"),
]


async def main():
    passed = 0
    for q, expect in SAMPLES:
        got = await classify(q)
        ok = got == expect
        passed += ok
        print(f"{'✅' if ok else '❌'} {q!r} -> {got} 期望={expect}")
    print(f"\n判对 {passed}/{len(SAMPLES)}(glm 非确定性,抖动如实重跑记录)")


if __name__ == "__main__":
    asyncio.run(main())
