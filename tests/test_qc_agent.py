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
        self.assertTrue(expected_is_fraud("引导投资"))
        self.assertIsNone(expected_is_fraud(""))
        self.assertTrue(category_matches("证券投资类", {"证券投资类"}))
        self.assertTrue(category_matches("任意", set()))  # 无法归一化时不计入


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
