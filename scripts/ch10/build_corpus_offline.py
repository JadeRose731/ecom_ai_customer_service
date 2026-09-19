"""ch10 语料离线代跑:CHAT 上游不可用时的过渡路径(2026-09-19 用户拍板「跳过先走,最后再改」)。
模板造数替代 LLM 预标与模拟:捞池/清洗与真链同口径(desensitize+dedupe,跳过 LLM 修错别字),
labeled 产物只含模拟行(origin=simulated,预标要 LLM,真实池不装标签)、各类补到 TARGET_PER_CLASS,
决定论可复现(固定槽位轮转,无随机)。上游修好跑 make ch10-corpus 即整目录覆盖——本脚本
只是把链路后半段(dataset→train→eval→export→serve→pool)先跑起来用的,不是第二条正路。"""
import json
import pathlib
import sys
from datetime import datetime

from app.core.taxonomy import TOPIC_CLASSES
from app.db import repository
from scripts.ch10.corpus_lib import dedupe, desensitize

OUT = pathlib.Path("data/ch10")
TARGET_PER_CLASS = 100

PRODUCTS = ("猫粮", "冻干", "猫罐头", "猫砂", "猫砂盆", "猫抓板", "猫窝",
            "猫爬架", "猫碗", "逗猫棒", "项圈", "猫零食", "化毛膏", "饮水机", "猫包")

# 每类单诉求模板:{p} 商品槽位;方言/错别字变体单独给,标签仍按诉求算(看穿字面认诉求)
SINGLES: dict[str, tuple[str, ...]] = {
    "退换货": ("{p}不想要了怎么退", "刚拆封的{p}能七天无理由退吗", "{p}退款怎么还没到账",
               "换货流程走一下{p}的", "这{p}我想退了钱怎么原路返回", "{p}退货要自己出邮费吗"),
    "物流": ("我的{p}到哪了", "{p}怎么还不动啊都三天了", "帮我查下{p}的物流",
             "{p}是发的哪家快递", "海外直邮的{p}一般几天到", "{p}说今天到怎么还没送到"),
    "尺码": ("{p}买大了能换小一码吗", "这{p}适合几斤的猫", "{p}有没有大一点的尺寸",
             "{p}的尺寸是多少厘米", "项圈偏码了想换", "{p}码数怎么选"),
    "发票": ("可以开发票吗", "发票抬头开错了能重开吗", "能开增值税发票吗",
             "两笔订单能合并开票吗", "电子发票怎么下载", "买{p}的发票丢了能补开吗"),
    "质量问题": ("{p}开胶了", "{p}破了个洞", "收到的{p}有瑕疵",
                 "{p}电机坏了完全不转", "{p}掉色严重算质量问题吗", "{p}内包装破了东西撒了"),
    "运费": ("{p}包邮吗", "退货运费谁承担", "运费险怎么赔",
             "不满意退货运费自己出吗", "满多少才免运费", "偏远地区加收运费吗"),
    "优惠活动": ("有什么优惠券可以领", "满减活动怎么算的", "这{p}有活动价吗",
                 "双十一有活动吗", "券和满减能叠加用吗", "直播间的优惠还能用吗"),
    "价保": ("刚买就降价了能补差价吗", "保价期是多久", "{p}这两天降价了给补差吗",
             "价保怎么申请", "双十一买的还能价保吗", "保价是按到手价算吗"),
    "支付": ("付不了款一直转圈", "花呗可以分期吗", "扣了两次钱怎么回事",
             "能货到付款吗", "数字人民币支持吗", "{p}下单用白条怎么付不了"),
    "订单修改": ("帮我改下收货地址", "收货电话填错了改一下", "订单还能取消吗",
                 "刚下的单想改颜色", "能改成分期付款吗", "备注忘了写能补吗"),
    "库存补货": ("{p}有货吗", "{p}断货了什么时候补", "有现货吗当天发吗",
                 "{p}补货了提醒我", "店里那款大号{p}还有库存吗", "{p}缺货要等多久"),
    "商品信息": ("这{p}什么材质的", "{p}怎么洗", "冻干怎么保存",
                 "猫粮怎么选幼猫吃的", "{p}几个月的猫能用", "{p}的成分表发一下"),
    "保修维修": ("{p}保修多久", "{p}坏了能修吗", "在保期内维修要钱吗",
                 "能换新吗还是只能修", "维修要寄回去吗运费谁出", "{p}过了保修期修一次多少"),
    "账号": ("登录不上一直验证失败", "忘了密码怎么找回", "换绑手机号怎么操作",
             "怎么注销我的账号", "账号被锁了怎么办", "第三方登录能解绑吗"),
    "会员积分": ("积分怎么用", "会员几级有什么权益", "积分能抵钱吗",
                 "积分过期了还能要回来吗", "生日月会员有礼吗", "积分和券能一起用吗"),
    "评价": ("评价怎么改", "追评在哪写", "晒单有奖励吗",
             "给差评商家能删吗", "带图评价有积分吗", "评价了怎么不显示"),
    "其他": ("转人工", "客服几点上班", "在吗有人吗",
             "你们店是正规的吗", "小程序打不开了", "怎么联系你们店主"),
}

# 多诉求模板:一次字面提到两个诉求,两个类都打(每类 ~7 条,全库约 15% 标签命中来自多诉求)
MULTIS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("{p}买大了想退", ("尺码", "退换货")),
    ("项圈买小了要换货", ("尺码", "退换货")),
    ("{p}开胶了必须退钱", ("质量问题", "退换货")),
    ("{p}破了个洞了退款吧", ("质量问题", "退换货")),
    ("运费谁出啊快递还卡在半路", ("运费", "物流")),
    ("{p}走什么快递运费多少", ("物流", "运费")),
    ("{p}还没发货我想改地址", ("物流", "订单修改")),
    ("货到付款能开发票吗", ("支付", "发票")),
    ("分期付款的订单怎么开发票", ("支付", "发票")),
    ("刚买的{p}降价了双十一这价给补差吗", ("价保", "优惠活动")),
    ("优惠券和积分能一起抵吗", ("优惠活动", "会员积分")),
    ("{p}电机坏了在保修期内修要钱吗", ("质量问题", "保修维修")),
    ("{p}有瑕疵退货运费谁出", ("质量问题", "运费")),
    ("退了这单我的优惠券还回来吗", ("退换货", "优惠活动")),
    ("退货退款的运费险怎么理赔", ("退换货", "运费")),
    ("{p}断货了到货了能提醒我改地址吗", ("库存补货", "订单修改")),
    ("换货的新{p}什么时候发货", ("退换货", "物流")),
    ("{p}怎么用顺便问下积分怎么攒", ("商品信息", "会员积分")),
    ("{p}尺寸多少合适几斤的猫用", ("商品信息", "尺码")),
    ("补货以后涨价了能价保吗", ("库存补货", "价保")),
    ("改完地址运费会变吗", ("订单修改", "运费")),
    ("{p}放保修期内换新要补差价吗", ("保修维修", "价保")),
    ("晒单返券的活动怎么参加", ("评价", "优惠活动")),
    ("{p}有货吗想用积分换购", ("库存补货", "会员积分")),
)

# 方言土话模板(每类 ~2 条,仿「俺买的那玩意儿咋还没到俺这疙瘩」口径)
DIALECT: dict[str, tuple[str, ...]] = {
    "退换货": ("俺买的{p}不中意想给它退喽", "这{p}俺不想要了钱给俺退回来呗"),
    "物流": ("俺买的那玩意儿{p}咋还没到俺这疙瘩", "俺的{p}咋走得这么磨叽啥时候能到啊"),
    "尺码": ("俺这{p}买大发了咋整", "这{p}俺家猫戴着晃荡是不是得换小一号"),
    "发票": ("能给我整个发票不报销用", "发票抬整错了再给俺开一张中不"),
    "质量问题": ("俺这{p}咋刚用就开裂了呢", "这{p}做工忒差了缝都开线了"),
    "运费": ("退这玩意儿邮费还得俺自个儿掏?", "俺们那嘎达发货包不包邮啊"),
    "优惠活动": ("有啥优惠不打折俺可就不买了", "这券咋使唤啊满多少能减"),
    "价保": ("俺前天买的今儿就掉价了给补个差价呗", "这才买几天就便宜了不给我补差说不过去吧"),
    "支付": ("咋付不了款呢急死俺了", "这钱咋扣了两回啊俺就买了一单"),
    "订单修改": ("地址整错了赶紧给俺改改", "刚拍的单子能撤了重下不"),
    "库存补货": ("俺要那{p}咋没货了呢啥时候能补", "这{p}啥时候能来货啊俺等着呢"),
    "商品信息": ("俺想知道这{p}是啥料子的", "这玩意儿咋清洗啊直接水冲中不中"),
    "保修维修": ("俺这{p}罢工了保修期内给修不", "过了保修这玩意儿还能修不"),
    "账号": ("俺这号咋登不上去呢", "密码忘了咋整帮俺弄回来"),
    "会员积分": ("俺那积分咋个使法", "俺这会员等级有啥好处啊"),
    "评价": ("评价写岔了咋改啊", "追评上哪儿写去俺找不着"),
    "其他": ("有人没客服呢", "俺想找个人工说说话"),
}

# 错别字变体:正字 → 误字,把单诉求模板做一版错字替换(标签不变)
TYPOS: tuple[tuple[str, str], ...] = (("退货", "退活"), ("尺码", "尺马"), ("猫粮", "猫量"),
                                      ("发货", "法货"), ("优惠券", "优惠圈"), ("发票", "发瞟"))

POLITE = ("", "请问", "想问下", "帮忙看下", "咨询一下")
TAILS = ("", ",在线等", ",急", ",麻烦快点", "", ",谢谢啦", "")


def _slot_fill(template: str, i: int) -> str:
    # 无槽模板也要有组合空间:前缀×尾缀×商品槽位三轴独立步进,防 dedupe 收走重复
    p = PRODUCTS[(i // 7) % len(PRODUCTS)]
    text = template.format(p=p)
    return POLITE[(i // 2) % len(POLITE)] + text + TAILS[(i // 11) % len(TAILS)]


def _gen_class(cls: str) -> list[dict]:
    out: list[dict] = []
    # 1) 多诉求:含该类的配对模板,槽位轮转出 7 条
    k = 0
    for tpl, labels in MULTIS:
        if cls in labels:
            for j in range(7):
                out.append({"text": tpl.format(p=PRODUCTS[(k + j * 3) % len(PRODUCTS)]),
                            "labels": list(labels), "origin": "simulated"})
            k += 1
    # 2) 方言 2 条 ×3 槽位
    for tpl in DIALECT.get(cls, ()):
        for j in range(3):
            out.append({"text": tpl.format(p=PRODUCTS[j * 5 % len(PRODUCTS)]),
                        "labels": [cls], "origin": "simulated"})
    # 3) 单诉求补到够:模板 × 商品槽位轮转,每第 4 条做一版错别字
    singles = SINGLES[cls]
    i = 0
    while sum(1 for s in out if cls in s["labels"]) < TARGET_PER_CLASS:
        text = _slot_fill(singles[i % len(singles)], i)
        if i % 4 == 1:
            for zheng, wu in TYPOS:
                if zheng in text:
                    text = text.replace(zheng, wu, 1)
                    break
        out.append({"text": text, "labels": [cls], "origin": "simulated"})
        i += 1
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)

    def dump(path: pathlib.Path, rows: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                        encoding="utf-8")

    # 1-2) 捞池与清洗与真链同口径(跳过 LLM 修错别字那步)
    import asyncio
    pool = asyncio.run(repository.list_pool_texts())
    raw = [{"text": p["text"], "labels": [], "origin": "pool"} for p in pool]
    dump(OUT / "corpus_raw.jsonl", raw)
    cleaned = dedupe([{**s, "text": desensitize(s["text"])} for s in raw])
    dump(OUT / "corpus_clean.jsonl", cleaned)
    print(f"捞池 {len(raw)} 条,清洗后 {len(cleaned)} 条(离线代跑:预标待上游,未入训练)")

    # 3) 模板造数:各类补到 TARGET_PER_CLASS(多诉求给命中的每类都记数)
    labeled: list[dict] = []
    counts = {c.name: 0 for c in TOPIC_CLASSES}
    for c in TOPIC_CLASSES:
        rows = dedupe(_gen_class(c.name))
        labeled.extend(rows)
        for s in rows:
            for lb in s["labels"]:
                counts[lb] += 1
    labeled = dedupe(labeled)
    dump(OUT / "corpus_labeled.jsonl", labeled)
    print(f"离线语料 {len(labeled)} 条(全 origin=simulated);各类:{counts}")

    # 4) 抽审导出:真实池列名单(待上游预标)+ 每类模拟抽 5
    rng = __import__("random").Random(42)
    lines = ["# ch10 语料人工抽审(离线代跑版)", "",
             f"> ⚠ 本文件由 build_corpus_offline.py 于 {datetime.now().isoformat(timespec='seconds')} 生成:",
             "> CHAT 上游不可用,模板造数替代 LLM 预标与模拟;上游修好后跑 make ch10-corpus 覆盖本目录。", "",
             "## 真实池问题(待预标,未入训练)", ""]
    lines += [f"- {s['text']}(待预标)" for s in cleaned]
    lines += ["", "## 模拟问题(每类抽 5)", ""]
    for c in TOPIC_CLASSES:
        sims = [s for s in labeled if c.name in s["labels"]]
        lines.append(f"### {c.name}")
        lines += [f"- {s['text']} → {'、'.join(s['labels'])}"
                  for s in rng.sample(sims, min(5, len(sims)))]
        lines.append("")
    (OUT / "sample_review.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"抽审文件已导出:{OUT / 'sample_review.md'}")


if __name__ == "__main__":
    main()
