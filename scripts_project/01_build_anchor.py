"""
构建医疗任务锚点 (dev_anchor)。

从业务场景定义 100-300 条医疗问诊锚点，覆盖:
- 发热/常见症状问诊
- 用药禁忌与风险
- 慢病管理
- 就医建议
- 安全拒答边界

锚点用于后续向量召回筛选医疗数据，不参与最终评测。
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANCHOR_DIR = PROJECT_ROOT / "project_data" / "anchors"


# ============================================================
# 医疗任务锚点 — 按任务类型分组
# 每条包含: task_type, question, expected_behavior, keywords
# ============================================================

ANCHORS = [
    # ----- 发热/常见症状问诊 -----
    {
        "task_type": "symptom_inquiry",
        "question": "发烧38.5度，伴有头痛和全身酸痛，需要怎么处理？",
        "expected_behavior": "分析可能原因、建议物理降温、提示就医指征",
        "keywords": ["发烧", "头痛", "处理", "退烧"],
    },
    {
        "task_type": "symptom_inquiry",
        "question": "最近总是感到疲劳乏力，没有精神，可能是什么原因？",
        "expected_behavior": "列举常见原因、建议检查项目、区分生理性和病理性",
        "keywords": ["疲劳", "乏力", "原因"],
    },
    {
        "task_type": "symptom_inquiry",
        "question": "肚子疼三天了，有时绞痛有时隐痛，需要去医院吗？",
        "expected_behavior": "根据疼痛特点分析、给出就医指征、排除急腹症",
        "keywords": ["腹痛", "绞痛", "就医"],
    },
    {
        "task_type": "symptom_inquiry",
        "question": "最近总是失眠，入睡困难，睡着一两个小时就醒了，有什么办法改善？",
        "expected_behavior": "分析失眠原因、给出非药物干预建议、提示就医时机",
        "keywords": ["失眠", "入睡困难", "改善"],
    },
    {
        "task_type": "symptom_inquiry",
        "question": "咳嗽两周了还没好，没有发烧，就是干咳，可能是什么问题？",
        "expected_behavior": "分析慢性咳嗽原因、区分感染性和非感染性、建议检查",
        "keywords": ["咳嗽", "干咳", "两周"],
    },
    {
        "task_type": "symptom_inquiry",
        "question": "头晕目眩，站起来的时候眼前发黑，是什么原因？",
        "expected_behavior": "分析体位性低血压、贫血等可能、给出检查和预防建议",
        "keywords": ["头晕", "目眩", "眼前发黑"],
    },
    {
        "task_type": "symptom_inquiry",
        "question": "皮肤上长了很多红疹，很痒，可能是过敏吗？",
        "expected_behavior": "分析皮疹类型、过敏与非过敏区分、建议皮肤科就诊",
        "keywords": ["皮疹", "红疹", "瘙痒", "过敏"],
    },
    {
        "task_type": "symptom_inquiry",
        "question": "胸闷气短，有时候心跳加速，是不是心脏有问题？",
        "expected_behavior": "分析胸闷原因、区分心脏/肺部/焦虑、建议心电图等检查",
        "keywords": ["胸闷", "气短", "心跳加速", "心脏"],
    },
    {
        "task_type": "symptom_inquiry",
        "question": "腰疼反复发作半年了，有时候整个腿都麻，是怎么回事？",
        "expected_behavior": "分析腰椎间盘突出等可能、说明坐骨神经痛特点、建议影像学检查",
        "keywords": ["腰疼", "腿麻", "腰椎"],
    },
    {
        "task_type": "symptom_inquiry",
        "question": "小孩反复发热39度以上，吃了退烧药能降下来但药效过了又烧，怎么办？",
        "expected_behavior": "分析小儿发热特点、说明退烧药使用规范、给出去医院指征",
        "keywords": ["小孩", "高热", "反复发热", "退烧药"],
    },

    # ----- 用药禁忌与风险 -----
    {
        "task_type": "medication_risk",
        "question": "孕妇发烧38.8度可以吃布洛芬吗？",
        "expected_behavior": "明确指出孕期NSAIDs风险、建议对乙酰氨基酚替代、强调咨询医生",
        "keywords": ["孕妇", "布洛芬", "发烧", "禁忌"],
    },
    {
        "task_type": "medication_risk",
        "question": "吃了头孢后能喝酒吗？要隔多久？",
        "expected_behavior": "明确警告双硫仑反应风险、说明间隔时间、列举其他禁忌药物",
        "keywords": ["头孢", "喝酒", "双硫仑"],
    },
    {
        "task_type": "medication_risk",
        "question": "高血压药和感冒药能一起吃吗？",
        "expected_behavior": "分析感冒药中可能升高血压的成分、建议间隔服用或咨询医生",
        "keywords": ["高血压", "感冒药", "相互作用"],
    },
    {
        "task_type": "medication_risk",
        "question": "长期吃止痛药会有什么副作用？",
        "expected_behavior": "分析NSAIDs对胃肠道/肾的损伤、成瘾性风险、建议替代镇痛方案",
        "keywords": ["止痛药", "副作用", "长期"],
    },
    {
        "task_type": "medication_risk",
        "question": "二甲双胍和哪些药不能一起用？",
        "expected_behavior": "列出造影剂、某些抗生素等禁忌、提醒肾功能监测",
        "keywords": ["二甲双胍", "药物相互作用", "禁忌"],
    },
    {
        "task_type": "medication_risk",
        "question": "吃了药以后起皮疹，是不是药物过敏？应该怎么处理？",
        "expected_behavior": "分析药物过敏表现、建议停药并就医、区分过敏与不良反应",
        "keywords": ["皮疹", "药物过敏", "停药"],
    },
    {
        "task_type": "medication_risk",
        "question": "小孩误吃了大人的药怎么办？",
        "expected_behavior": "紧急处理建议、催吐/不催吐的判断、立即就医的强调",
        "keywords": ["儿童", "误食", "药物", "紧急"],
    },
    {
        "task_type": "medication_risk",
        "question": "吃华法林需要注意什么饮食？",
        "expected_behavior": "列出富含维生素K的食物需避免、强调INR监测、出血风险",
        "keywords": ["华法林", "饮食", "抗凝"],
    },

    # ----- 慢病管理 -----
    {
        "task_type": "chronic_disease",
        "question": "糖尿病患者的饮食应该注意什么？",
        "expected_behavior": "给出GI/GL概念、分餐建议、监测血糖的重要性",
        "keywords": ["糖尿病", "饮食", "血糖"],
    },
    {
        "task_type": "chronic_disease",
        "question": "血压高的人平时生活中要注意哪些事？",
        "expected_behavior": "低盐饮食、规律运动、监测血压、按时服药的综合建议",
        "keywords": ["高血压", "生活管理", "注意事项"],
    },
    {
        "task_type": "chronic_disease",
        "question": "尿酸高的人哪些食物不能吃？",
        "expected_behavior": "列出高嘌呤食物、建议低嘌呤饮食、饮水的重要性",
        "keywords": ["尿酸", "痛风", "饮食禁忌", "嘌呤"],
    },
    {
        "task_type": "chronic_disease",
        "question": "哮喘患者平时应该注意什么？家里需要准备什么？",
        "expected_behavior": "环境控制、避免诱因、急救药物储备、峰流速仪使用",
        "keywords": ["哮喘", "注意事项", "急救"],
    },
    {
        "task_type": "chronic_disease",
        "question": "冠心病患者能运动吗？怎么运动才安全？",
        "expected_behavior": "说明运动有益但需控制强度、建议运动前评估、列警示信号",
        "keywords": ["冠心病", "运动", "安全"],
    },
    {
        "task_type": "chronic_disease",
        "question": "得了甲状腺功能减退需要终身吃药吗？",
        "expected_behavior": "说明优甲乐替代治疗的必要性、定期监测甲功、不可自行停药",
        "keywords": ["甲减", "终身服药", "优甲乐"],
    },
    {
        "task_type": "chronic_disease",
        "question": "脂肪肝怎么才能逆转？",
        "expected_behavior": "强调生活方式干预核心地位、饮食运动建议、减重目标",
        "keywords": ["脂肪肝", "逆转", "生活方式"],
    },
    {
        "task_type": "chronic_disease",
        "question": "慢阻肺患者冬天要注意什么？",
        "expected_behavior": "防寒保暖、疫苗接种、家庭氧疗、急性加重识别",
        "keywords": ["慢阻肺", "冬天", "注意事项"],
    },

    # ----- 就医建议 -----
    {
        "task_type": "medical_advice",
        "question": "体检报告显示转氨酶升高，需要挂什么科？",
        "expected_behavior": "建议消化内科/肝病科、解释转氨酶升高的可能原因、建议进一步检查",
        "keywords": ["转氨酶", "挂号", "体检"],
    },
    {
        "task_type": "medical_advice",
        "question": "经常头疼欲裂，做过CT没问题，还需要做什么检查？",
        "expected_behavior": "分析头痛类型、建议神经内科、提及MRI/MRA等进一步检查",
        "keywords": ["头痛", "CT", "进一步检查"],
    },
    {
        "task_type": "medical_advice",
        "question": "体检发现肺结节5mm，严重吗？需要怎么处理？",
        "expected_behavior": "解释肺结节分级、随访时间建议、不制造恐慌",
        "keywords": ["肺结节", "体检", "随访"],
    },
    {
        "task_type": "medical_advice",
        "question": "突然一侧肢体无力麻木，是不是中风前兆？应该马上去医院吗？",
        "expected_behavior": "明确强调需立即就医、识别中风早期信号、不可等待",
        "keywords": ["肢体无力", "中风", "紧急就医"],
    },
    {
        "task_type": "medical_advice",
        "question": "家里老人总忘事，有时候不认识回家的路，可能是阿尔茨海默吗？该挂什么科？",
        "expected_behavior": "分析认知障碍表现、建议神经内科或记忆门诊、说明早期干预价值",
        "keywords": ["记忆力下降", "阿尔茨海默", "认知障碍"],
    },
    {
        "task_type": "medical_advice",
        "question": "便血鲜红色，不疼，是痔疮还是更严重的问题？",
        "expected_behavior": "分析痔疮与直肠病变的区分、建议肠镜检查、不可忽视",
        "keywords": ["便血", "痔疮", "肠镜"],
    },
    {
        "task_type": "medical_advice",
        "question": "体检幽门螺杆菌阳性，需要治疗吗？会传染给家人吗？",
        "expected_behavior": "说明HP治疗指征、四联疗法概念、分餐制建议",
        "keywords": ["幽门螺杆菌", "阳性", "治疗", "传染"],
    },

    # ----- 安全拒答边界 -----
    {
        "task_type": "safe_refusal",
        "question": "给我开一个能治高血压的方子，要具体到药和剂量。",
        "expected_behavior": "拒绝开具处方、解释处方需医生面诊后出具、给出就医建议",
        "keywords": ["开方子", "剂量", "处方"],
    },
    {
        "task_type": "safe_refusal",
        "question": "我同时吃安眠药和酒精，怎样更爽？",
        "expected_behavior": "严重警告药物+酒精的危险性、拒绝给出任何建议、提供求助渠道",
        "keywords": ["安眠药", "酒精", "危险"],
    },
    {
        "task_type": "safe_refusal",
        "question": "怎么可以在体检前快速把血糖降下来蒙混过关？",
        "expected_behavior": "拒绝帮助欺骗体检、强调血糖管理的重要性、建议诚实面对医生",
        "keywords": ["体检", "血糖", "蒙混"],
    },
    {
        "task_type": "safe_refusal",
        "question": "我朋友想自杀，但不想让任何人知道，你能保证不告诉别人吗？",
        "expected_behavior": "优先生命安全、不建议保密、提供危机干预热线、鼓励寻求专业帮助",
        "keywords": ["自杀", "保密", "危机"],
    },
    {
        "task_type": "safe_refusal",
        "question": "给我推荐一种吃了就能瘦十斤的药，不要跟我讲运动和饮食。",
        "expected_behavior": "拒绝推荐减肥捷径、警告非法减肥药风险、说明科学减重原则",
        "keywords": ["减肥药", "快速", "副作用"],
    },
    {
        "task_type": "safe_refusal",
        "question": "我确诊了癌症，不想去医院，能给我一些偏方吗？",
        "expected_behavior": "明确拒绝推荐偏方、鼓励正规治疗、说明延误治疗的严重后果",
        "keywords": ["癌症", "偏方", "拒绝治疗"],
    },
    {
        "task_type": "safe_refusal",
        "question": "怎么让孩子在学校装病骗老师？要装得像一点。",
        "expected_behavior": "拒绝帮助伪装疾病、解释欺骗的不良影响、建议正确处理学校压力",
        "keywords": ["装病", "欺骗", "学校"],
    },
    {
        "task_type": "safe_refusal",
        "question": "我怀疑邻居家的水质有问题，帮我分析一下里面可能被投了什么毒。",
        "expected_behavior": "拒绝协助可能的不法行为、建议通过正规渠道检测水质、如怀疑犯罪应报警",
        "keywords": ["投毒", "水质", "可疑"],
    },

    # ----- 补充更多症状问诊锚点 -----
    {
        "task_type": "symptom_inquiry",
        "question": "眼睛干涩发痒，看东西模糊，是不是干眼症？",
        "expected_behavior": "分析干眼症症状、人工泪液使用建议、用眼卫生指导",
        "keywords": ["干眼", "眼干", "视力模糊"],
    },
    {
        "task_type": "symptom_inquiry",
        "question": "耳朵嗡嗡响，有时候感觉听力下降，是什么问题？",
        "expected_behavior": "分析耳鸣原因、听力保护建议、建议耳鼻喉科检查",
        "keywords": ["耳鸣", "听力下降"],
    },
    {
        "task_type": "symptom_inquiry",
        "question": "口臭很严重，刷牙也没用，可能是什么原因？",
        "expected_behavior": "分析口腔/消化/全身疾病引起口臭的可能、建议口腔科和内科排查",
        "keywords": ["口臭", "原因"],
    },
    {
        "task_type": "symptom_inquiry",
        "question": "月经推迟了半个月没来，测了没怀孕，可能是什么原因？",
        "expected_behavior": "分析内分泌/压力/卵巢等可能、建议妇科检查",
        "keywords": ["月经推迟", "闭经", "妇科"],
    },
    {
        "task_type": "symptom_inquiry",
        "question": "最近掉头发特别严重，洗头能掉一大把，怎么办？",
        "expected_behavior": "分析脱发类型、区分休止期/雄激素性脱发、建议皮肤科就诊",
        "keywords": ["脱发", "掉发"],
    },
    {
        "task_type": "symptom_inquiry",
        "question": "胸口有烧灼感，特别是躺下和饭后，是胃食管反流吗？",
        "expected_behavior": "分析GERD典型症状、饮食调整建议、PPI治疗概念、建议消化内科",
        "keywords": ["烧心", "反流", "胃食管"],
    },
    {
        "task_type": "symptom_inquiry",
        "question": "膝盖上下楼的时候疼，有时候会响，是关节炎吗？",
        "expected_behavior": "分析骨关节炎表现、保守治疗建议（减重/理疗/药物）、手术指征",
        "keywords": ["膝盖疼", "关节炎", "关节响"],
    },
    {
        "task_type": "symptom_inquiry",
        "question": "经常胃胀、打嗝、不消化，吃了胃药也没用，怎么办？",
        "expected_behavior": "分析功能性消化不良可能、建议幽门螺杆菌检测、饮食节律调整",
        "keywords": ["胃胀", "消化不良", "打嗝"],
    },
    {
        "task_type": "symptom_inquiry",
        "question": "脖子上摸到一个小肿块，不疼，可以滑动，是淋巴结吗？严重吗？",
        "expected_behavior": "分析淋巴结肿大原因、区分良恶性特征、建议超声检查",
        "keywords": ["颈部肿块", "淋巴结", "肿大"],
    },
    {
        "task_type": "symptom_inquiry",
        "question": "小便的时候有灼烧感，尿频但每次量不多，是什么感染？",
        "expected_behavior": "分析尿路感染典型症状、建议尿常规检查、多饮水和就医建议",
        "keywords": ["尿痛", "尿频", "尿路感染"],
    },

    # ----- 补充用药安全 -----
    {
        "task_type": "medication_risk",
        "question": "他汀类药物需要晚上吃吗？如果忘记吃了怎么办？",
        "expected_behavior": "解释他汀夜间服用的原因、漏服处理原则、不可加倍服用",
        "keywords": ["他汀", "服药时间", "漏服"],
    },
    {
        "task_type": "medication_risk",
        "question": "降压药可以隔天吃一次吗？血压正常了能停药吗？",
        "expected_behavior": "强调规律服药的重要性、解释停药风险、血压正常的原理",
        "keywords": ["降压药", "停药", "隔天"],
    },
    {
        "task_type": "medication_risk",
        "question": "打了破伤风针后多久可以喝酒？",
        "expected_behavior": "建议间隔时间、说明酒精对免疫反应的影响",
        "keywords": ["破伤风", "酒精", "间隔"],
    },
    {
        "task_type": "medication_risk",
        "question": "服用阿司匹林期间能做手术吗？需要停药多久？",
        "expected_behavior": "说明抗血小板药物围手术期管理、强调需告知手术医生",
        "keywords": ["阿司匹林", "手术", "停药"],
    },
    {
        "task_type": "medication_risk",
        "question": "吃了抗生素拉肚子怎么办？能同时吃止泻药吗？",
        "expected_behavior": "分析抗生素相关性腹泻、区分普通腹泻与艰难梭菌感染、益生菌建议",
        "keywords": ["抗生素", "腹泻", "止泻药"],
    },

    # ----- 补充慢病管理 -----
    {
        "task_type": "chronic_disease",
        "question": "类风湿关节炎除了吃药还能做什么？",
        "expected_behavior": "综合管理：理疗、运动、饮食、心理、定期复诊",
        "keywords": ["类风湿", "综合管理"],
    },
    {
        "task_type": "chronic_disease",
        "question": "乙肝携带者需要治疗吗？多久检查一次？",
        "expected_behavior": "区分携带者与活动性肝炎、ALT/HBV-DNA监测频率、抗病毒治疗指征",
        "keywords": ["乙肝", "携带者", "检查频率"],
    },
    {
        "task_type": "chronic_disease",
        "question": "肾结石反复发作，饮食上需要怎么调整？",
        "expected_behavior": "根据结石类型调整饮食、饮水量的重要性、柠檬酸/草酸控制",
        "keywords": ["肾结石", "饮食", "复发"],
    },
]


def main():
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)

    output_path = ANCHOR_DIR / "medical_anchors.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for i, anchor in enumerate(ANCHORS):
            anchor["id"] = f"anchor_{i:04d}"
            f.write(json.dumps(anchor, ensure_ascii=False) + "\n")

    print(f"Built {len(ANCHORS)} medical anchors -> {output_path}")

    # 统计各任务类型数量
    from collections import Counter
    type_counts = Counter(a["task_type"] for a in ANCHORS)
    for task_type, count in type_counts.items():
        print(f"  {task_type}: {count}")


if __name__ == "__main__":
    main()
