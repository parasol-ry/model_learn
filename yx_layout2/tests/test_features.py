import json

from latain.core import Block

with open("tests/record1.json", "r") as f:
    record = json.loads(f.read())
    doc = Block.from_dict(record)
    doc.generate_feature()


def test_record_feature():
    span_0_0 = doc[0][0]
    span_0_0_features = {
        "span_font": "NotoSansCJKsc-Bold",
        "span_is_bold": True,
        "span_color": "105,110,125",
        "span_normalized_fsize": 1.0,
        "span_id": 0,
        "span_width": 42.0,
        "span_left": 30.5,
        "span_right": 72.5,
        "span_is_most_left": True,
        "span_is_most_right": False,
        "span_left_margin": 0.0,
        "span_right_margin": 358.5,
        "span_is_align_with_prev_line": False,
        "span_is_align_with_next_line": False,
        "span_num_tokens": 4,
        "span_is_startswith_chinese": True,
        "span_first_token": "工",
        "span_first_token_tag": "<CJK>",
        "span_last_token": "历",
        "span_last_token_tag": "<CJK>",
        "span_leading_tag": "<CJK>",
        "span_text_is_time_range": False,
    }
    assert span_0_0.features == span_0_0_features

    span_1_0 = doc[1][0]
    span_1_0_features = {
        "span_font": "NotoSansCJKsc-Bold",
        "span_is_bold": True,
        "span_color": "37,42,51",
        "span_normalized_fsize": 0.0,
        "span_id": 0,
        "span_width": 86.74275207519531,
        "span_left": 35.75,
        "span_right": 122.49275207519531,
        "span_is_most_left": False,
        "span_is_most_right": False,
        "span_left_margin": 5.25,
        "span_right_margin": 12.257247924804688,
        "span_is_align_with_prev_line": False,
        "span_is_align_with_next_line": False,
        "span_num_tokens": 17,
        "span_is_startswith_chinese": False,
        "span_first_token": "2",
        "span_first_token_tag": "<NUM>",
        "span_last_token": "5",
        "span_last_token_tag": "<NUM>",
        "span_leading_tag": "<NUM>",
        "span_text_is_time_range": True,
    }
    assert span_1_0.features == span_1_0_features

    span_1_1 = doc[1][1]
    span_1_1_features = {
        "span_font": "NotoSansCJKsc-Bold",
        "span_is_bold": True,
        "span_color": "37,42,51",
        "span_normalized_fsize": 0.0,
        "span_id": 1,
        "span_width": 272.18548583984375,
        "span_left": 134.75,
        "span_right": 406.93548583984375,
        "span_is_most_left": False,
        "span_is_most_right": False,
        "span_left_margin": 12.257247924804688,
        "span_right_margin": 24.06451416015625,
        "span_is_align_with_prev_line": False,
        "span_is_align_with_next_line": False,
        "span_num_tokens": 31,
        "span_is_startswith_chinese": True,
        "span_first_token": "北",
        "span_first_token_tag": "<CJK>",
        "span_last_token": ")",
        "span_last_token_tag": "<POS>",
        "span_leading_tag": "<CJK>",
        "span_text_is_time_range": False,
    }
    assert span_1_1.features == span_1_1_features

    span_2_0 = doc[2][0]
    span_2_0_features = {
        "span_font": "NotoSansCJKsc-Bold",
        "span_is_bold": True,
        "span_color": "37,42,51",
        "span_normalized_fsize": 0.0,
        "span_id": 0,
        "span_width": 154.5,
        "span_left": 134.75,
        "span_right": 289.25,
        "span_is_most_left": False,
        "span_is_most_right": False,
        "span_left_margin": 104.25,
        "span_right_margin": 141.75,
        "span_is_align_with_prev_line": False,
        "span_is_align_with_next_line": True,
        "span_num_tokens": 23,
        "span_is_startswith_chinese": True,
        "span_first_token": "售",
        "span_first_token_tag": "<CJK>",
        "span_last_token": "月",
        "span_last_token_tag": "<CJK>",
        "span_leading_tag": "<NUM>",
        "span_text_is_time_range": False,
    }
    assert span_2_0.features == span_2_0_features

    span_n1_n1 = doc[-1][-1]
    span_n1_n1_features = {
        "span_font": "NotoSansCJKsc-Regular",
        "span_is_bold": False,
        "span_color": "37,42,51",
        "span_normalized_fsize": 0.0,
        "span_id": 0,
        "span_width": 141.75,
        "span_left": 133.25,
        "span_right": 275.0,
        "span_is_most_left": False,
        "span_is_most_right": False,
        "span_left_margin": 102.75,
        "span_right_margin": 156.0,
        "span_is_align_with_prev_line": True,
        "span_is_align_with_next_line": False,
        "span_num_tokens": 15,
        "span_is_startswith_chinese": False,
        "span_first_token": "3",
        "span_first_token_tag": "<NUM>",
        "span_last_token": ";",
        "span_last_token_tag": "<POS>",
        "span_leading_tag": "<CJK>",
        "span_text_is_time_range": False,
    }
    assert span_n1_n1.features == span_n1_n1_features


def test_line_feature():
    line_0 = doc[0]
    line_0_features = {
        "line_is_first": True,
        "line_is_last": False,
        "line_id": 0,
        "line_leading_fsize": 10.5,
        "line_max_fsize": 10.5,
        "line_is_same_fsize": True,
        "line_is_same_fsize_with_prev": True,
        "line_is_same_fsize_with_next": False,
        "line_is_same_font": True,
        "line_is_same_color": True,
        "line_num_spans": 1,
        "line_left_margin": 0.0,
        "line_right_margin": 358.5,
        "line_top_margin": 0,
        "line_down_margin": 19.08001708984375,
        "line_is_most_left": True,
        "line_is_most_right": False,
        "line_is_align_with_prev": False,
        "line_is_align_with_next": False,
        "line_max_width": 42.0,
        "line_max_interspace": 0,
        "line_average_width": 42.0,
        "line_average_interspace": 0,
        "line_num_tokens": 4,
        "line_is_startswith_chinese": True,
        "line_have_time_range": False,
    }

    assert line_0.features == line_0_features

    line_1 = doc[1]
    line_1_features = {
        "line_is_first": False,
        "line_is_last": False,
        "line_id": 1,
        "line_leading_fsize": 9.75,
        "line_max_fsize": 9.75,
        "line_is_same_fsize": True,
        "line_is_same_fsize_with_prev": False,
        "line_is_same_fsize_with_next": True,
        "line_is_same_font": True,
        "line_is_same_color": True,
        "line_num_spans": 2,
        "line_left_margin": 5.25,
        "line_right_margin": 24.06451416015625,
        "line_top_margin": 19.08001708984375,
        "line_down_margin": 2.07000732421875,
        "line_is_most_left": False,
        "line_is_most_right": False,
        "line_is_align_with_prev": False,
        "line_is_align_with_next": False,
        "line_max_width": 272.18548583984375,
        "line_max_interspace": 12.257247924804688,
        "line_average_width": 179.46411895751953,
        "line_average_interspace": 12.257247924804688,
        "line_num_tokens": 48,
        "line_is_startswith_chinese": False,
        "line_have_time_range": True,
    }

    assert line_1.features == line_1_features

    line_2 = doc[2]
    line_2_features = {
        "line_is_first": False,
        "line_is_last": False,
        "line_id": 2,
        "line_leading_fsize": 9.75,
        "line_max_fsize": 9.75,
        "line_is_same_fsize": True,
        "line_is_same_fsize_with_prev": True,
        "line_is_same_fsize_with_next": True,
        "line_is_same_font": True,
        "line_is_same_color": True,
        "line_num_spans": 1,
        "line_left_margin": 104.25,
        "line_right_margin": 141.75,
        "line_top_margin": 2.07000732421875,
        "line_down_margin": 2.07000732421875,
        "line_is_most_left": False,
        "line_is_most_right": False,
        "line_is_align_with_prev": False,
        "line_is_align_with_next": True,
        "line_max_width": 154.5,
        "line_max_interspace": 0,
        "line_average_width": 154.5,
        "line_average_interspace": 0,
        "line_num_tokens": 23,
        "line_is_startswith_chinese": True,
        "line_have_time_range": False,
    }
    assert line_2.features == line_2_features

    line_n_1 = doc[-1]
    line_n_1_features = {
        "line_is_first": False,
        "line_is_last": True,
        "line_id": 14,
        "line_leading_fsize": 9.75,
        "line_max_fsize": 9.75,
        "line_is_same_fsize": True,
        "line_is_same_fsize_with_prev": True,
        "line_is_same_fsize_with_next": True,
        "line_is_same_font": True,
        "line_is_same_color": True,
        "line_num_spans": 1,
        "line_left_margin": 102.75,
        "line_right_margin": 156.0,
        "line_top_margin": 0.57000732421875,
        "line_down_margin": 0,
        "line_is_most_left": False,
        "line_is_most_right": False,
        "line_is_align_with_prev": True,
        "line_is_align_with_next": False,
        "line_max_width": 141.75,
        "line_max_interspace": 0,
        "line_average_width": 141.75,
        "line_average_interspace": 0,
        "line_num_tokens": 15,
        "line_is_startswith_chinese": False,
        "line_have_time_range": False,
    }
    assert line_n_1.features == line_n_1_features
