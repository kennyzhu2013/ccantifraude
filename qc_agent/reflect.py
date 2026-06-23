"""演进与反射 Agent（自治闭环）。

当主质检 Agent 的判定与人工标签不一致，或从网页/新案例中发现新型话术时，
反射 Agent 分析盲区、提炼新特征，并把知识固化回 rules.json：
    - 新增高危关键词 → 对应场景 keywords
    - 沉淀错题本 → evolved_examples（下次作为 few-shot 注入主 Agent）
这正是 learn-claude-code 记忆体系『选择/提炼/固化』在反诈领域的落地。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed

from .agent import QcAgent
from .case_store import CaseStore
from .labels import expected_is_fraud, normalize_label
from .retrieval import char_ngrams
from .schema import InspectionResult


@dataclass
class EvolutionProposal:
    need_evolution: bool = False
    target_category: str = "其他"
    new_keywords: List[str] = field(default_factory=list)
    pattern_update: str = ""
    reflection_notes: str = ""


def expected_violation_from_comment(comment: str) -> bool:
    """CSV 的 comment 为人工复核说明：非空通常表示存在违规/涉诈线索。"""
    return bool((comment or "").strip())


def classify_conflict(comment: str, res: InspectionResult) -> Optional[Dict[str, Any]]:
    """判定预测与人工标签是否冲突，并归类冲突类型（供标签治理）。"""
    expected_v = bool((comment or "").strip())
    acceptable = normalize_label(comment)
    types = []
    if expected_v and not res.is_violation:
        types.append("漏判：人工有标签但模型判正常")
    if res.is_violation and acceptable and res.scene_category not in acceptable:
        types.append("类目不一致")
    ef = expected_is_fraud(comment)
    if ef is True and not res.is_fraud:
        types.append("涉诈漏标")
    if not types:
        return None
    return {
        "conflict_type": "；".join(types),
        "acceptable_categories": "|".join(sorted(acceptable)),
    }


def _guess_category(kb, comment: str, content: str) -> str:
    """根据人工标签 + 正文，猜测最匹配的场景类目。"""
    text = f"{comment} {comment} {content}"
    best_cat, best_score = "其他", 0
    for bucket in ("fraud_scenarios", "violation_scenarios"):
        for sc in kb.rules.get(bucket, []):
            kws = sc.get("keywords", [])
            score = sum(1 for k in kws if k and k in text)
            # 标签直接含类目名则强匹配。
            if sc.get("category") and sc["category"] in comment:
                score += 5
            if score > best_score:
                best_cat, best_score = sc.get("category", "其他"), score
    return best_cat


class ReflectAgent:
    def __init__(self, agent: QcAgent, max_evolved: int = 50, max_new_keywords: int = 3):
        self.agent = agent
        self.kb = agent.kb
        self.max_evolved = max_evolved
        self.max_new_keywords = max_new_keywords

    # ---------- 单例反射 ----------
    def reflect(
        self, content: str, prediction: InspectionResult, human_label: str
    ) -> EvolutionProposal:
        if self.agent.llm.available:
            try:
                return self._reflect_llm(content, prediction, human_label)
            except Exception:
                pass
        return self._reflect_heuristic(content, prediction, human_label)

    def _reflect_heuristic(
        self, content: str, prediction: InspectionResult, human_label: str
    ) -> EvolutionProposal:
        expected = expected_violation_from_comment(human_label)
        if prediction.is_violation == expected:
            return EvolutionProposal(need_evolution=False)

        category = _guess_category(self.kb, human_label, content)
        scenario = self.kb.get_scenario(category) or {}
        existing = set(scenario.get("keywords", []))

        # 从正文提炼候选高危 n-gram：在该类目命中最相关、且尚未收录的片段。
        grams = [g for g in char_ngrams(content, (3, 4)) if g not in existing]
        # 取与人工标签字符重叠最高者，作为弱信号补充。
        label_chars = set(human_label)
        grams.sort(key=lambda g: len(set(g) & label_chars), reverse=True)
        new_keywords = [g for g in grams[: self.max_new_keywords] if set(g) & label_chars]

        notes = (
            f"主Agent判定为{'违规' if prediction.is_violation else '正常'}，"
            f"但人工标注『{human_label}』提示应为违规。归类到『{category}』并沉淀错题本。"
        )
        return EvolutionProposal(
            need_evolution=True,
            target_category=category,
            new_keywords=new_keywords,
            pattern_update=f"{category}：{human_label}",
            reflection_notes=notes,
        )

    def _reflect_llm(
        self, content: str, prediction: InspectionResult, human_label: str
    ) -> EvolutionProposal:
        system = (
            "你是高阶反诈策略进化专家。对比主质检Agent的判定与人工标签，"
            "找出知识盲区并提炼可固化的新规则。仅输出 JSON："
            '{"need_evolution":bool,"target_category":str,"new_keywords":[str],'
            '"pattern_update":str,"reflection_notes":str}'
        )
        user = (
            f"【通话内容】{content[:3000]}\n"
            f"【主Agent判定】{prediction.to_json()}\n"
            f"【人工标签/线索】{human_label}\n"
            "若存在盲区或新型变体，请提炼升级方案；target_category 应取自现有场景类目。"
        )
        msg = self.agent.llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        payload = json.loads(_first_json(msg["content"]))
        return EvolutionProposal(
            need_evolution=bool(payload.get("need_evolution", False)),
            target_category=str(payload.get("target_category", "其他") or "其他"),
            new_keywords=list(payload.get("new_keywords", []) or [])[: self.max_new_keywords],
            pattern_update=str(payload.get("pattern_update", "") or ""),
            reflection_notes=str(payload.get("reflection_notes", "") or ""),
        )

    # ---------- 应用演进 ----------
    def apply(self, content: str, human_label: str, proposal: EvolutionProposal) -> bool:
        if not proposal.need_evolution:
            return False
        rules = self.kb.rules
        category = proposal.target_category

        for bucket in ("violation_scenarios", "fraud_scenarios"):
            for sc in rules.get(bucket, []):
                if sc.get("category") == category:
                    kws = sc.setdefault("keywords", [])
                    for kw in proposal.new_keywords:
                        if kw and kw not in kws:
                            kws.append(kw)

        evolved = rules.setdefault("evolved_examples", [])
        evolved.append(
            {
                "label": proposal.pattern_update or human_label,
                "analysis": proposal.reflection_notes,
                "snippet": (content or "")[:200],
            }
        )
        if len(evolved) > self.max_evolved:
            rules["evolved_examples"] = evolved[-self.max_evolved :]

        self.kb.save_rules(rules)
        return True

    # ---------- 冲突扫描（标签治理用，默认不改 rules） ----------
    def scan_conflicts(
        self,
        cases: Optional[CaseStore] = None,
        use_tools: bool = False,
        workers: int = 6,
        limit: Optional[int] = None,
        sample: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        store = cases or self.agent.cases
        if sample and sample < len(store.cases):
            step = max(1, len(store.cases) // sample)
            items = store.cases[::step][:sample]
        else:
            items = store.cases[:limit] if limit else store.cases

        def run_one(case):
            res = self.agent.inspect(case.content, data_id=case.data_id, use_tools=use_tools)
            return case, res

        conflicts: List[Dict[str, Any]] = []
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_one, c) for c in items]
            for fut in as_completed(futures):
                case, res = fut.result()
                done += 1
                if verbose and done % 20 == 0:
                    print(f"  扫描 {done}/{len(items)} ...", flush=True)
                if done % 50 == 0:
                    self.agent.flush_cache()  # 周期落盘，长任务中断不丢进度
                info = classify_conflict(case.comment, res)
                if info is None:
                    continue
                conflicts.append(
                    {
                        "data_id": case.data_id,
                        "human_comment": case.comment,
                        "conflict_type": info["conflict_type"],
                        "model_label": res.label,
                        "model_category": res.scene_category,
                        "model_subtype": res.scene_subtype,
                        "model_is_fraud": res.is_fraud,
                        "model_risk": res.risk_level.value,
                        "acceptable_categories": info["acceptable_categories"],
                        "model_explanation": res.explanation,
                        "content_snippet": case.short(300),
                        "suggested_action": "人工复核：确认应以规范判定还是修正人工标签",
                    }
                )
        self.agent.flush_cache()
        return {"total": len(items), "conflicts": conflicts}

    # ---------- 批量自治演进 ----------
    def evolve_from_cases(
        self, cases: Optional[CaseStore] = None, limit: Optional[int] = None, verbose: bool = True
    ) -> Dict[str, Any]:
        store = cases or self.agent.cases
        stats = {"total": 0, "conflicts": 0, "evolved": 0}
        items = store.cases[:limit] if limit else store.cases
        for case in items:
            stats["total"] += 1
            pred = self.agent.inspect(case.content, data_id=case.data_id)
            expected = expected_violation_from_comment(case.comment)
            if pred.is_violation == expected:
                continue
            stats["conflicts"] += 1
            proposal = self.reflect(case.content, pred, case.comment)
            if self.apply(case.content, case.comment, proposal):
                stats["evolved"] += 1
                if verbose:
                    print(
                        f"  演进[{case.data_id}] -> {proposal.target_category} "
                        f"+词{proposal.new_keywords} | {proposal.reflection_notes}"
                    )
        return stats


def _first_json(text: str) -> str:
    import re

    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    return m.group(0) if m else "{}"
