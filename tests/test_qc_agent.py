"""离线可跑通的核心组件测试（无需 LLM / 第三方依赖）。"""
import sys
from pathlib import Path

import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_CSV = ROOT / "data" / "sample_cases.csv"

from qc_agent import Config, QcAgent  # noqa: E402
from qc_agent.case_store import CaseStore  # noqa: E402
from qc_agent.knowledge_base import KnowledgeBase  # noqa: E402
from qc_agent.retrieval import TfidfIndex, char_ngrams  # noqa: E402
from qc_agent.reflect import ReflectAgent  # noqa: E402
from qc_agent.schema import InspectionResult, RiskLevel  # noqa: E402


class TestRetrieval(unittest.TestCase):
    def test_char_ngrams(self):
        grams = char_ngrams("引导投资", (2, 3))
        self.assertIn("引导", grams)
        self.assertIn("引导投", grams)

    def test_tfidf_search_ranks_relevant(self):
        docs = ["引导添加炒股福利群投资理财", "宽带提速不加价免费升级", "贷款下款经理微信"]
        idx = TfidfIndex().fit(docs)
        hits = idx.search("炒股投资福利群", top_k=1)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0], 0)


class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.kb = KnowledgeBase(self.cfg.spec_path, self.cfg.rules_path)

    def test_spec_parsed(self):
        self.assertGreater(len(self.kb.sections), 5)

    def test_scenarios_loaded(self):
        scens = self.kb.list_scenarios()
        self.assertIn("引流第三方平台", scens["违规场景"])
        self.assertIn("证券投资类", scens["涉诈场景"])

    def test_search_spec_returns_sections(self):
        hits = self.kb.search_spec("航空公司 改签 屏幕共享", top_k=2)
        self.assertTrue(hits)

    def test_rules_brief_nonempty(self):
        self.assertIn("就高不就低", self.kb.rules_brief())


class TestCaseStore(unittest.TestCase):
    def setUp(self):
        self.store = CaseStore(SAMPLE_CSV)

    def test_loaded(self):
        self.assertGreater(len(self.store), 5)

    def test_retrieve(self):
        hits = self.store.retrieve("航空公司航班取消改签退票领取赔付", top_k=1)
        self.assertTrue(hits)
        self.assertIn("航", hits[0].content)


def _offline_config() -> Config:
    """强制离线（启发式）：忽略 .env 中可能存在的真实 Key，固定用样例语料，保证单测可重复、不联网。"""
    cfg = Config()
    cfg.llm_api_key = ""
    cfg.cases_path = SAMPLE_CSV
    return cfg


class TestHeuristicInspection(unittest.TestCase):
    def setUp(self):
        cfg = _offline_config()
        self.agent = QcAgent(config=cfg, cases=CaseStore(cfg.cases_path))

    def test_mode_is_heuristic_without_key(self):
        if not self.agent.llm.available:
            self.assertEqual(self.agent.mode, "heuristic")

    def test_securities_fraud_detected(self):
        text = (
            "left:我是投顾客服，公司联合上海证券在企辽通创建官方福利群，"
            "免费延长投顾服务，关注官方接待员的服务号，打开微信点击加号搜索一下领取服务置顶取消免打扰。"
        )
        res = self.agent.inspect(text)
        self.assertTrue(res.is_violation)
        self.assertTrue(res.is_fraud)
        self.assertEqual(res.risk_level, RiskLevel.HIGH)

    def test_flight_fraud_detected(self):
        text = (
            "left:这边是南方航空，您的航班由于机场管控取消，请问改签还是退票？"
            "我们航空公司有三百元现金赔付，打开支付宝我指导您领取。"
        )
        res = self.agent.inspect(text)
        self.assertTrue(res.is_fraud)
        self.assertEqual(res.scene_category, "机票退、改签诈骗")

    def test_normal_call_compliant(self):
        text = "left:您好，您家宽带最近用着还好吗？有个提速不加价活动，免费升级到500兆，帮您登记一下。"
        res = self.agent.inspect(text)
        self.assertFalse(res.is_violation)
        self.assertEqual(res.risk_level, RiskLevel.COMPLIANT)

    def test_compliant_outbound_wechat_not_flagged(self):
        text = "left:你好，咱们的产品资料我加你微信发给你看一下，我这边加你哈，备注一下。"
        res = self.agent.inspect(text)
        # 外呼人员主动添加个人微信属合规，不应判为引流违规高风险。
        self.assertFalse(res.is_violation)

    def test_loan_downgrade_cashout_is_violation_not_fraud(self):
        text = (
            "left:我是贷款平台客户经理，帮您把利息下调，您把账户里剩余的两万块钱先提现，"
            "提现好之后系统才能检测到您账户，用不上的话随时可以提前还进来，不要担心。"
        )
        res = self.agent.inspect(text)
        # 最新规范口径：引导用户平台操作提现=【贷款相关】违规高风险，非涉诈。
        self.assertTrue(res.is_violation)
        self.assertFalse(res.is_fraud)
        self.assertEqual(res.scene_category, "贷款相关")
        self.assertEqual(res.risk_level, RiskLevel.HIGH)

    def test_credit_score_inquiry_is_rental_fraud(self):
        text = (
            "left:咱们这边不看征信，您支付宝的芝麻信用分有多少分？截个图发我，"
            "分数够的话帮您在租赁平台走个额度出来周转。"
        )
        res = self.agent.inspect(text)
        # 最新口径：非银行渠道要求/询问芝麻信用分 → 手机租赁套路贷诈骗（涉诈）。
        self.assertTrue(res.is_fraud)
        self.assertEqual(res.scene_category, "手机租赁套路贷诈骗")

    def test_keyword_match_case_insensitive(self):
        text = "left:免费送您一台pos机，激活后刷够流水才能提现，先垫五十九块钱运费。"
        res = self.agent.inspect(text)
        # 关键词库为『POS机』，转写为小写也应命中（大小写不敏感）。
        self.assertTrue(res.is_violation)
        self.assertEqual(res.scene_category, "商品推销")

    def test_kantouxi_requires_amount(self):
        # 『到手』后跟金额 → 砍头息高风险。
        text = "left:我们这边放款快，借一万到手八千，每天还五百五，网贷不上征信。"
        res = self.agent.inspect(text)
        self.assertEqual(res.scene_category, "贷款相关")
        self.assertEqual(res.risk_level, RiskLevel.HIGH)
        # 日常表述『钱到手了』不应误升高风险。
        text2 = "left:您在拍拍贷申请的网贷已经放款，钱到手了记得按时还款就行。"
        res2 = self.agent.inspect(text2)
        self.assertNotEqual(res2.risk_level, RiskLevel.HIGH)

    def test_tax_scam_keywords_no_longer_collide_with_generic_loan_terms(self):
        """个体工商户年报补录收费不应因『营业执照/法人』等通用词与经营贷业务混淆。"""
        text = (
            "left:您这个营业执照做经营贷，您是法人吧，流水做一下增量，"
            "我们收两个点服务费，三十到五十万没问题。"
        )
        res = self.agent.inspect(text)
        self.assertNotEqual(res.scene_category, "个体工商户年报补录收费")

    def test_verification_code_forwarding_fraud_detected(self):
        """要求转发/读出短信验证码，无论借口是什么，都应判定为验证码诈骗高风险。"""
        text = (
            "left:您好，这边是装修业务受理，需要您签字确认一下，麻烦您把刚才收到的那条短信"
            "内容念给我一下，或者直接把那个短信转发到我这个手机号上，验证码是好多？"
        )
        res = self.agent.inspect(text)
        self.assertTrue(res.is_violation)
        self.assertTrue(res.is_fraud)
        self.assertEqual(res.scene_category, "验证码/短信转发诈骗")

    def test_collections_ex_spouse_and_balance_screenshot_detected(self):
        """催收提及联系前妻/要求发送银行卡余额截图，超出正常催收范围，应判违规。"""
        text = (
            "left:你欠的钱不还，我们就联系你前妻了。你把余额截个图发给我，"
            "我要确保你有没有钱，有钱的话我给你转平台里边。"
        )
        res = self.agent.inspect(text)
        self.assertTrue(res.is_violation)
        self.assertEqual(res.scene_category, "违规催收")

    def test_collections_severe_threat_escalated_to_high_risk(self):
        """催收威胁全网公开身份证号/冒充纪检机关等极端手段，应升级为高风险而非默认低风险。"""
        text = (
            "left:你的姓名身份证号等信息将全网公开，你会被列入失信黑名单，"
            "还会移交纪检委和组织部处理。"
        )
        res = self.agent.inspect(text)
        self.assertTrue(res.is_violation)
        self.assertEqual(res.scene_category, "违规催收")
        self.assertEqual(res.risk_level, RiskLevel.HIGH)

    def test_tax_scam_canonical_case_still_detected(self):
        text = (
            "left:我是这边线上税务部中心点的，你这个营业执照还没有年报，"
            "季度税务申报也没做，可能产生一笔补录费用，把补录交上去。"
        )
        res = self.agent.inspect(text)
        self.assertEqual(res.scene_category, "个体工商户年报补录收费")
        self.assertTrue(res.is_fraud)

    def test_ab_loan_escalated_to_fraud(self):
        """最新规范：AB贷（受托支付+背调）升级为涉诈。"""
        text = (
            "left:这次走受托支付，需要你提供一个企业账户收款，我们还要对贷后做合法背调，"
            "提交指定紧急联系人，企业侧双签才能放款。"
        )
        res = self.agent.inspect(text)
        self.assertTrue(res.is_violation)
        self.assertTrue(res.is_fraud)
        self.assertEqual(res.scene_category, "贷款相关-ab贷")

    def test_guide_add_private_wechat_fraud(self):
        """贷款推销场景平台核资后引导添加第三方经理微信，按最新规范判涉诈。"""
        text = (
            "left:安逸花人工审核已经通过了，您符合放款条件，需要您添加我们放款经理的企业微信，"
            "把额度发给下款经理线上走流程。"
        )
        res = self.agent.inspect(text)
        self.assertTrue(res.is_fraud)
        self.assertEqual(res.scene_category, "引导贷款用户添加第三方微信")

    def test_travel_fee_fraud_detected(self):
        """有薪招聘+仓库路线+入职注册费，判套路运诈骗。"""
        text = (
            "left:我们物流公司招司机，有集中仓库和固定路线可以驻站，"
            "入职先交800元平台注册费，承诺第一单就有500补贴。"
        )
        res = self.agent.inspect(text)
        self.assertTrue(res.is_fraud)
        self.assertEqual(res.scene_category, "套路运诈骗")

    def test_fake_invoice_medium_risk_keywords_in_rules(self):
        """企业虚开成本票应出现在企业营销风险规则中。"""
        sc = self.agent.kb.get_scenario("企业营销与招商服务")
        self.assertIsNotNone(sc)
        joined = " ".join(sc.get("risk_rules", []))
        self.assertIn("虚开成本票", joined)


    def test_provident_fund_rule_registered(self):
        """违规提取公积金应为贷款相关中风险规则。"""
        sc = self.agent.kb.get_scenario("贷款相关")
        joined = " ".join(sc.get("risk_rules", []))
        self.assertIn("违规提取公积金", joined)

    def test_taobao_flash_purchase_compliant_in_brief(self):
        """淘宝闪购联盟功能开启应为合规口径。"""
        brief = self.agent.kb.rules_brief()
        self.assertIn("淘宝闪购", brief)

    def test_official_account_same_fraud_pattern_in_disambig(self):
        """关注公众号后对接放款经理应为相同套路（涉诈），分层注入下按检索命中。"""
        entries = self.agent.kb.relevant_disambiguation(
            "贷款推销，引导关注公众号对接放款经理加微信"
        )
        joined = " ".join(entries)
        self.assertIn("公众号", joined)
        self.assertIn("相同套路", joined)

    def test_disambiguation_layered_injection(self):
        """分层注入：brief 只常驻消歧标题索引，正文按待检文本检索命中后才注入。"""
        brief = self.agent.kb.rules_brief()
        # 标题索引常驻。
        self.assertIn("易混场景子类目判别索引", brief)
        self.assertIn("【最新规范·芝麻信用分】", brief)
        # 消歧正文的长尾细节不再全量常驻（降 token）。
        self.assertNotIn("不要仅凭出现『微信』二字就判高风险", brief)
        # 但能按待检文本检索回全文。
        entries = self.agent.kb.relevant_disambiguation("加个微信嘛，我加一下你微信发资料")
        self.assertTrue(any("添加方向" in e for e in entries))


class TestSchema(unittest.TestCase):
    def test_round_trip(self):
        res = InspectionResult(
            is_violation=True, is_fraud=True, risk_level=RiskLevel.HIGH,
            scene_category="证券投资类", explanation="涉诈",
        )
        d = res.to_dict()
        self.assertEqual(d["risk_level"], "高风险")
        res2 = InspectionResult.from_dict(d)
        self.assertEqual(res2.scene_category, "证券投资类")
        self.assertTrue(res2.is_fraud)

    def test_label(self):
        res = InspectionResult(
            is_violation=True, is_fraud=False, risk_level=RiskLevel.LOW,
            scene_category="贷款相关", scene_subtype="提前收取费用",
        )
        self.assertEqual(res.label, "违规-低风险-贷款相关/提前收取费用")


class TestLabels(unittest.TestCase):
    def test_normalize(self):
        from qc_agent.labels import normalize_label, expected_is_fraud, category_matches

        self.assertIn("证券投资类", normalize_label("引导投资"))
        self.assertIn("证券投资类", normalize_label("引导投资理财"))
        self.assertIn("手机租赁套路贷诈骗", normalize_label("手机租赁套路贷看诈骗"))
        self.assertIn("网贷平台退息退费", normalize_label("网贷平台退息退费，涉诈"))
        self.assertIn("企业营销与招商服务", normalize_label("会展招商会收费"))
        self.assertTrue(expected_is_fraud("引导投资"))
        self.assertIsNone(expected_is_fraud(""))
        self.assertTrue(category_matches("证券投资类", {"证券投资类"}))
        self.assertTrue(category_matches("任意", set()))  # 无法归一化时不计入

    def test_non_fraud_hints_latest_spec(self):
        from qc_agent.labels import expected_is_fraud

        # 最新口径：违规非涉诈场景不应因『套路贷/退费』字样误判涉诈。
        self.assertFalse(expected_is_fraud("贷款相关：引导平台操作提现，偏套路贷"))
        self.assertFalse(expected_is_fraud("法律服务：帮退律所费用，成功后收一半服务费"))
        # 涉诈类目仍应判涉诈。
        self.assertTrue(expected_is_fraud("网贷平台退息退费"))
        self.assertTrue(expected_is_fraud("手机租赁套路贷诈骗"))

    def test_compliant_label(self):
        from qc_agent.labels import is_compliant_label

        self.assertTrue(is_compliant_label("合规招商加盟"))
        self.assertTrue(is_compliant_label("品牌招商加盟"))
        self.assertFalse(is_compliant_label("引导投资理财"))
        self.assertFalse(is_compliant_label(""))

    def test_decision_table_in_brief(self):
        cfg = Config()
        kb = KnowledgeBase(cfg.spec_path, cfg.rules_path)
        brief = kb.rules_brief()
        self.assertIn("业务口径判定表", brief)
        self.assertIn("AI推广获客服务", brief)
        self.assertIn("证券投资引流", brief)

    def test_wechat_direction_disambiguation_in_brief(self):
        """微信添加方向判别规则：标题常驻 brief，全文按检索命中注入。"""
        cfg = Config()
        kb = KnowledgeBase(cfg.spec_path, cfg.rules_path)
        self.assertIn("添加方向", kb.rules_brief())
        entries = kb.relevant_disambiguation("加个微信嘛，方便的话我加一下你")
        joined = " ".join(entries)
        self.assertIn("加个微信嘛", joined)

    def test_loan_downgrade_cashout_retired_from_fraud(self):
        """最新规范口径：『贷款降息诱导套现诈骗』已撤销，并入【贷款相关·引导用户平台操作提现】违规高风险。"""
        cfg = Config()
        kb = KnowledgeBase(cfg.spec_path, cfg.rules_path)
        cats = [s.get("category") for s in kb.rules.get("fraud_scenarios", [])]
        self.assertNotIn("贷款降息诱导套现诈骗", cats)
        self.assertIn("引导用户平台操作提现", kb.rules_brief())

    def test_latest_spec_fraud_types_registered(self):
        """对客最新复核规范新增涉诈类目应注册到知识库。"""
        cfg = Config()
        kb = KnowledgeBase(cfg.spec_path, cfg.rules_path)
        cats = [s.get("category") for s in kb.rules.get("fraud_scenarios", [])]
        for name in ("贷款相关-ab贷", "引导贷款用户添加第三方微信", "套路运诈骗"):
            self.assertIn(name, cats)
            self.assertIn(name, kb.rules_brief())
        # 最新规范口径已合入 spec.md，应可检索
        hits = kb.search_spec("套路运诈骗 注册费 固定路线", top_k=3)
        self.assertTrue(hits)
        brief = kb.rules_brief()
        self.assertIn("企业虚开成本票", brief)
        self.assertIn("套路运诈骗", brief)
        self.assertIn("引导贷款用户添加第三方微信", brief)


class TestDedup(unittest.TestCase):
    def test_cluster_near_duplicates(self):
        from qc_agent.dedup import cluster_texts, group_by_cluster

        texts = [
            "我是投顾客服，联合上海证券创建官方福利群，关注官方接待员服务号",
            "我是投顾客服，联合上海证券创建官方福利群，关注官方接待员的服务号",  # 近重复
            "宽带提速不加价免费升级到500兆",
        ]
        clusters = cluster_texts(texts, threshold=0.8)
        groups = group_by_cluster(clusters)
        self.assertEqual(clusters[0], clusters[1])  # 前两条同簇
        self.assertNotEqual(clusters[0], clusters[2])
        self.assertEqual(len(groups), 2)


class TestCache(unittest.TestCase):
    def test_set_get_persist(self):
        import tempfile, shutil
        from qc_agent.cache import ResultCache

        tmp = Path(tempfile.mkdtemp())
        path = tmp / "c.jsonl"
        c = ResultCache(path, model="m1", mode="fast")
        self.assertIsNone(c.get("hello"))
        c.set("hello", {"x": 1})
        c.flush()
        c2 = ResultCache(path, model="m1", mode="fast")
        self.assertEqual(c2.get("hello"), {"x": 1})
        # 换模型命名空间应失效。
        c3 = ResultCache(path, model="m2", mode="fast")
        self.assertIsNone(c3.get("hello"))
        shutil.rmtree(tmp)


class TestConflictClassify(unittest.TestCase):
    def test_classify(self):
        from qc_agent.reflect import classify_conflict
        from qc_agent.schema import InspectionResult, RiskLevel

        # 人工有标签但模型判正常 -> 漏判
        normal = InspectionResult(is_violation=False, risk_level=RiskLevel.COMPLIANT, scene_category="正常")
        c1 = classify_conflict("引导投资", normal)
        self.assertIsNotNone(c1)
        self.assertIn("漏判", c1["conflict_type"])
        # 类目不一致
        wrong_cat = InspectionResult(
            is_violation=True, is_fraud=True, risk_level=RiskLevel.HIGH, scene_category="引流第三方平台"
        )
        c2 = classify_conflict("引导投资", wrong_cat)
        self.assertIn("类目不一致", c2["conflict_type"])
        # 一致 -> 无冲突
        ok = InspectionResult(
            is_violation=True, is_fraud=True, risk_level=RiskLevel.HIGH, scene_category="证券投资类"
        )
        self.assertIsNone(classify_conflict("引导投资", ok))

    def test_bucket(self):
        from qc_agent.reflect import (
            bucket_conflict,
            BUCKET_RELABEL_COMPLIANT,
            BUCKET_REAL_VIOLATION,
            BUCKET_NEED_HUMAN,
        )
        from qc_agent.schema import InspectionResult, RiskLevel

        # 模型判正常 + 招商加盟话术 -> A 回标合规
        a = InspectionResult(is_violation=False, scene_category="正常",
                             explanation="休闲零食品牌招商加盟，合规")
        self.assertEqual(bucket_conflict("引导投资理财", a, "零食店加盟招商"), BUCKET_RELABEL_COMPLIANT)
        # 模型判违规/涉诈 -> B 真违规
        b = InspectionResult(is_violation=True, is_fraud=True, risk_level=RiskLevel.HIGH,
                             scene_category="引流第三方平台")
        self.assertEqual(bucket_conflict("手机租赁套路贷诈骗", b, "微钱包贷款加微信"), BUCKET_REAL_VIOLATION)
        # 模型判正常 + 非典型招商加盟 -> C 待人工
        c = InspectionResult(is_violation=False, scene_category="正常", explanation="商业合作沟通")
        self.assertEqual(bucket_conflict("引导投资", c, "你好在外面打个招呼"), BUCKET_NEED_HUMAN)

    def test_bucket_domain_override_guardrail(self):
        """设备租赁平台招募区域服务商：即便含招商/加盟表面词，也不应误入A桶（回标合规）。"""
        from qc_agent.reflect import bucket_conflict, BUCKET_RELABEL_COMPLIANT, BUCKET_NEED_HUMAN
        from qc_agent.schema import InspectionResult

        content = "left:我们这边是天机宿租设备租赁平台，招募区域服务商，需要先出资锁定区域，长期抽佣管道式收益，这属于招商版块的合作模式。"
        missed = InspectionResult(is_violation=False, scene_category="正常",
                                  explanation="正常品牌招商加盟推广")
        self.assertEqual(
            bucket_conflict("手机租赁套路贷诈骗", missed, content), BUCKET_NEED_HUMAN
        )
        # 普通品牌招商加盟（无域内红旗词）应仍归 A。
        plain = InspectionResult(is_violation=False, scene_category="正常",
                                 explanation="休闲零食品牌招商加盟，合规")
        self.assertEqual(
            bucket_conflict("引导投资理财", plain, "零食店加盟招商"), BUCKET_RELABEL_COMPLIANT
        )


class TestReflectEvolution(unittest.TestCase):
    def test_evolution_adds_examples(self):
        import json, tempfile, shutil
        cfg = _offline_config()
        tmp = Path(tempfile.mkdtemp())
        rules_copy = tmp / "rules.json"
        shutil.copy(cfg.rules_path, rules_copy)
        kb = KnowledgeBase(cfg.spec_path, rules_copy)
        store = CaseStore(SAMPLE_CSV)
        agent = QcAgent(config=cfg, kb=kb, cases=store)
        reflector = ReflectAgent(agent)
        before = len(kb.rules.get("evolved_examples", []))
        stats = reflector.evolve_from_cases(store, verbose=False)
        self.assertEqual(stats["total"], len(store))
        after = len(json.loads(rules_copy.read_text(encoding="utf-8")).get("evolved_examples", []))
        self.assertGreaterEqual(after, before)
        shutil.rmtree(tmp)

    def test_candidate_keywords_isolated_until_promoted(self):
        """演进词先进暂存区：不污染生产 keywords，审核晋升/丢弃后才变更。"""
        import tempfile, shutil
        from qc_agent.reflect import (
            EvolutionProposal,
            list_candidate_keywords,
            review_candidates,
        )

        cfg = _offline_config()
        tmp = Path(tempfile.mkdtemp())
        rules_copy = tmp / "rules.json"
        shutil.copy(cfg.rules_path, rules_copy)
        kb = KnowledgeBase(cfg.spec_path, rules_copy)
        agent = QcAgent(config=cfg, kb=kb, cases=CaseStore(SAMPLE_CSV))
        reflector = ReflectAgent(agent)

        proposal = EvolutionProposal(
            need_evolution=True,
            target_category="贷款相关",
            new_keywords=["测试噪声词甲", "测试噪声词乙"],
            pattern_update="贷款相关：单测演进隔离",
        )
        self.assertTrue(reflector.apply("left:测试内容", "贷款相关", proposal))

        # 未审核：不入生产 keywords，启发式与 get_scenario 均不可见。
        self.assertNotIn("测试噪声词甲", kb.all_keywords().get("贷款相关", []))
        sc = kb.get_scenario("贷款相关")
        self.assertNotIn("candidate_keywords", sc)
        self.assertIn("贷款相关", list_candidate_keywords(kb))

        # 晋升甲、丢弃乙：甲进生产，乙消失，暂存区清空。
        review_candidates(kb, "贷款相关", keywords=["测试噪声词甲"], promote=True)
        review_candidates(kb, "贷款相关", keywords=["测试噪声词乙"], promote=False)
        self.assertIn("测试噪声词甲", kb.all_keywords().get("贷款相关", []))
        self.assertNotIn("测试噪声词乙", kb.all_keywords().get("贷款相关", []))
        self.assertEqual(list_candidate_keywords(kb), {})
        shutil.rmtree(tmp)


class TestExtractJson(unittest.TestCase):
    def test_plain_json(self):
        from qc_agent.agent import _extract_json

        text = '{"is_violation": true, "scene_category": "贷款相关"}'
        obj = _extract_json(text)
        self.assertTrue(obj["is_violation"])
        self.assertEqual(obj["scene_category"], "贷款相关")

    def test_markdown_fenced(self):
        from qc_agent.agent import _extract_json

        text = '分析如下：\n```json\n{"is_violation": false, "risk_level": "合规"}\n```\n完毕'
        obj = _extract_json(text)
        self.assertFalse(obj["is_violation"])

    def test_prose_with_embedded_json(self):
        from qc_agent.agent import _extract_json

        text = (
            "根据规范判定为违规。\n"
            '{"is_violation": true, "is_fraud": true, "risk_level": "高风险", '
            '"scene_category": "证券投资类", "explanation": "引导投资"}'
        )
        obj = _extract_json(text)
        self.assertTrue(obj["is_fraud"])
        self.assertEqual(obj["scene_category"], "证券投资类")

    def test_nested_strings_do_not_break_parser(self):
        from qc_agent.agent import _extract_json

        text = (
            '{"is_violation": true, "evidence_quotes": ["对方说：}不是结束"], '
            '"scene_category": "贷款相关"}'
        )
        obj = _extract_json(text)
        self.assertEqual(obj["scene_category"], "贷款相关")

    def test_trailing_comma(self):
        from qc_agent.agent import _extract_json

        text = '{"is_violation": false, "risk_level": "合规",}'
        obj = _extract_json(text)
        self.assertFalse(obj["is_violation"])

    def test_prefers_inspection_object_when_multiple(self):
        from qc_agent.agent import _extract_json

        text = 'tool args {"query": "贷款"} final {"is_violation": true, "scene_category": "贷款相关"}'
        obj = _extract_json(text)
        self.assertEqual(obj["scene_category"], "贷款相关")


if __name__ == "__main__":
    unittest.main(verbosity=2)
