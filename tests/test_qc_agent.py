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

    def test_loan_downgrade_cashout_fraud_detected(self):
        text = (
            "left:我是贷款平台客户经理，帮您把利息下调，您把账户里剩余的两万块钱先提现，"
            "提现好之后系统才能检测到您账户，用不上的话随时可以提前还进来，不要担心。"
        )
        res = self.agent.inspect(text)
        self.assertTrue(res.is_violation)
        self.assertTrue(res.is_fraud)
        self.assertEqual(res.scene_category, "贷款降息诱导套现诈骗")

    def test_tax_scam_keywords_no_longer_collide_with_generic_loan_terms(self):
        """个体工商户年报补录收费不应因『营业执照/法人』等通用词与经营贷业务混淆。"""
        text = (
            "left:您这个营业执照做经营贷，您是法人吧，流水做一下增量，"
            "我们收两个点服务费，三十到五十万没问题。"
        )
        res = self.agent.inspect(text)
        self.assertNotEqual(res.scene_category, "个体工商户年报补录收费")

    def test_tax_scam_canonical_case_still_detected(self):
        text = (
            "left:我是这边线上税务部中心点的，你这个营业执照还没有年报，"
            "季度税务申报也没做，可能产生一笔补录费用，把补录交上去。"
        )
        res = self.agent.inspect(text)
        self.assertEqual(res.scene_category, "个体工商户年报补录收费")
        self.assertTrue(res.is_fraud)


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
        """微信添加方向判别规则应注入 prompt，降低『我加你』被误判为高风险的概率。"""
        cfg = Config()
        kb = KnowledgeBase(cfg.spec_path, cfg.rules_path)
        brief = kb.rules_brief()
        self.assertIn("添加方向", brief)
        self.assertIn("加个微信嘛", brief)

    def test_loan_downgrade_cashout_fraud_registered(self):
        """新增『贷款降息诱导套现诈骗』应出现在知识库与 prompt 中。"""
        cfg = Config()
        kb = KnowledgeBase(cfg.spec_path, cfg.rules_path)
        cats = [s.get("category") for s in kb.rules.get("fraud_scenarios", [])]
        self.assertIn("贷款降息诱导套现诈骗", cats)
        self.assertIn("贷款降息诱导套现诈骗", kb.rules_brief())


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
