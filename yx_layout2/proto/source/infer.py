# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from ..mrd import MrdDoc
from .base_rule import DocRule, DocTextRule, LineTextRule, SpanRule
from .base_source import OTHER, Source
from .lagou_rules import lg_rules
from .liepin_rules import lp_rules
from .qcwy_rules import qcwy_rules
from .zhilian_rules import zl_rules

all_rules = zl_rules + qcwy_rules + lg_rules + lp_rules


def apply_doc_rules(doc: MrdDoc, doc_rules: Sequence[DocRule]) -> Optional[Source]:
    for rule in doc_rules:
        source = rule.apply(doc)
        if source is not None:
            return source
    return None


def apply_doc_text_rules(
    doc_text: str, doc_text_rules: Sequence[DocTextRule]
) -> Optional[Source]:
    for rule in doc_text_rules:
        source = rule.apply(doc_text)
        if source is not None:
            return source
    return None


def apply_line_text_rules(
    line_texts: Sequence[str], line_text_rules: Sequence[LineTextRule]
) -> Optional[Source]:
    for line_text in line_texts:
        for rule in line_text_rules:
            source = rule.apply(line_text)
            if source is not None:
                return source
    return None


def apply_span_rules(doc: MrdDoc, span_rules: Sequence[SpanRule]) -> Optional[Source]:
    for line in doc:
        for span in line:
            for span_rule in span_rules:
                source = span_rule.apply(span)
                if source is not None:
                    return source
    return None


# no strict
def infer_source(doc: MrdDoc) -> Source:
    doc_text = doc.to_text("").replace(" ", "")
    doc_line_texts = [line_text.replace(" ", "") for line_text in doc.to_line_texts("")]

    doc_rules_map: Dict[Optional[Source], List[DocRule]] = defaultdict(list)
    doc_text_rules_map: Dict[Optional[Source], List[DocTextRule]] = defaultdict(list)
    line_text_rules_map: Dict[Optional[Source], List[LineTextRule]] = defaultdict(list)
    span_rules_map: Dict[Optional[Source], List[SpanRule]] = defaultdict(list)

    for rule in all_rules:
        if isinstance(rule, DocRule):
            doc_rules_map[rule.dep].append(rule)
        elif isinstance(rule, DocTextRule):
            doc_text_rules_map[rule.dep].append(rule)
        elif isinstance(rule, LineTextRule):
            line_text_rules_map[rule.dep].append(rule)
        elif isinstance(rule, SpanRule):
            span_rules_map[rule.dep].append(rule)

    applied_dep = set()
    dep = None
    dep_is_changed = False
    type_name_sequence = ["doc", "doc_text", "line_text", "span"]

    while dep not in applied_dep:
        # dep_is_changed = False
        for type_name in type_name_sequence:
            # if dep_is_changed:
            #     break

            if type_name == "doc":
                source = apply_doc_rules(doc, doc_rules_map[dep])
            elif type_name == "doc_text":
                source = apply_doc_text_rules(doc_text, doc_text_rules_map[dep])
            elif type_name == "line_text":
                source = apply_line_text_rules(doc_line_texts, line_text_rules_map[dep])
            elif type_name == "span":
                source = apply_span_rules(doc, span_rules_map[dep])
            else:
                raise ValueError(f"unknown type_name: {type_name}")

            if source is not None:
                if source.is_leaf:
                    return source
                else:
                    applied_dep.add(dep)
                    dep = source
                    # dep_is_changed = True
                    break
        else:
            applied_dep.add(dep)
    if dep is None:
        return OTHER
    return dep
