import time
import json
from concurrent.futures import ThreadPoolExecutor

from utils.api_clients import run_claude, run_openai_chat_completions, google_websearch
from utils.prompt_io_utils import (
    extract_itemprompts_text_from_zip,
    make_outline_str,
    split_layout_dict_five,
    make_outline_layout_template_set,
    merge_lv5_outputs,
)
from utils.utils import load_json_to_dict, try_fix_and_parse_json


def lv1_outline_websearch(lv1_system_prompt, lv1_user_prompt_template, user_input_text):
    """
    개요 생성 전 웹서치 단계
    1. context를 가지고 검색 쿼리 1개 생성(LLM)
    2. 1번에서 생성한 검색 쿼리를 가지고 웹 서치 진행(gopopg), 상위 5개 항목 가져오기
    """
    
    lv1_user_prompt = lv1_user_prompt_template % user_input_text
    
    # 1. context를 가지고 검색 쿼리 1개 생성(LLM)
    start_time = time.time()
    response = run_claude(lv1_system_prompt, lv1_user_prompt)
    lv1_output = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    execution_time = time.time() - start_time
    
    search_query = json.loads(lv1_output.split('<json>')[1])
    
    # 2. 1번에서 생성한 검색 쿼리를 가지고 웹 서치 진행(gopopg), 상위 10개 항목 가져오기
    web_search_result = google_websearch(search_query[0], 10)
    
    print(f'✅ lv1(websearch) done. time: {execution_time}')
    
    return {
        "user_input_text": user_input_text,
        "lv1_outline_background_search": {
            "system_prompt": lv1_system_prompt,
            "user_prompt": lv1_user_prompt,
            "llm_output": lv1_output,
            "input_tokens": input_tokens, 
            "output_tokens": output_tokens,
            "time": execution_time
            
        },
        "lv1_websearch_result": {
            "search_query": search_query,
            "websearch_output": web_search_result,
        }
    }


def lv2_outline(lv2_system_prompt, lv2_user_prompt_template, 
                user_input_text, slide_count, audience, tone, extra_note, lv1_websearch_result_websearch_output):
    """
    개요 생성 단계
    """
    lv2_user_prompt = lv2_user_prompt_template % (slide_count, audience, tone, extra_note, user_input_text, lv1_websearch_result_websearch_output)
    
    start_time = time.time()
    response = run_claude(lv2_system_prompt, lv2_user_prompt)
    lv2_output = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    execution_time = time.time() - start_time
    
    print(f'✅ lv2(outline) done. time: {execution_time}')
    
    return {
        "user_inputs": {
            "user_input_text": user_input_text,
            "slide_count": slide_count,
            "audience": audience,
            "tone": tone,
            "extra_note": extra_note,
        },
        "lv2_outline": {
            "system_prompt": lv2_system_prompt,
            "user_prompt": lv2_user_prompt,
            "llm_output": lv2_output,
            "input_tokens": input_tokens, 
            "output_tokens": output_tokens,
            "time": execution_time
        }
    }
    
    
def lv2_2_template_recommendation(lv2_2_system_prompt, lv2_2_user_prompt_template,
                                  user_input_text, audience, tone, template_keyword_list_path, op_language):
    """
    사용자 입력 기반 템플릿 추천 단계
    """
    template_keyword_list = load_json_to_dict(template_keyword_list_path)
    lv2_2_user_prompt = lv2_2_user_prompt_template % (user_input_text, audience, tone, op_language, template_keyword_list)
    
    start_time = time.time()
    response = run_openai_chat_completions(lv2_2_system_prompt, lv2_2_user_prompt)
    lv2_2_output = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    execution_time = time.time() - start_time
    
    print(f'✅ lv2_2(template_recommendation) done. time: {execution_time}')
    
    return {
        "user_inputs": {
            "user_input_text": user_input_text,
            "audience": audience,
            "tone": tone,
        },
        "op_language": op_language,
        "template_keyword_list": template_keyword_list,
        "lv2_2_template_recommendation": {
            "system_prompt": lv2_2_system_prompt,
            "user_prompt": lv2_2_user_prompt,
            "llm_output": lv2_2_output,
            "input_tokens": input_tokens, 
            "output_tokens": output_tokens,
            "time": execution_time
        }
    }
    
    
def lv3_layout(lv3_system_prompt, lv3_user_prompt_template, item_prompt_path,
               user_input_text, slide_count, audience, tone, extra_note, lv2_outline_llm_output):
    """
    레이아웃 생성 단계
    """
    item_prompt_dict = extract_itemprompts_text_from_zip(item_prompt_path)
    outline_xml_str = make_outline_str(lv2_outline_llm_output)
    
    lv3_user_prompt = lv3_user_prompt_template % (slide_count, audience, tone, outline_xml_str, extra_note, user_input_text, item_prompt_dict['template_info_purpose'])
    
    start_time = time.time()
    response = run_claude(lv3_system_prompt, lv3_user_prompt)
    lv3_output = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    execution_time = time.time() - start_time
    
    print(f'✅ lv3(layout) done. time: {execution_time}')
    
    return {
        "user_inputs": {
            "user_input_text": user_input_text,
            "slide_count": slide_count,
            "audience": audience,
            "tone": tone,
            "extra_note": extra_note,
            "template_idx": item_prompt_dict['template_idx'],
        },
        "lv3_layout": {
            "system_prompt": lv3_system_prompt,
            "user_prompt": lv3_user_prompt,
            "llm_output": lv3_output,
            "input_tokens": input_tokens, 
            "output_tokens": output_tokens,
            "time": execution_time
        }
    }


def lv4_content_websearch(lv4_system_prompt, lv4_user_prompt_template, user_input_text, lv2_outline_llm_output):
    """
    내용 생성 전 웹서치 단계
    """
    outline_xml_str = make_outline_str(lv2_outline_llm_output)
    
    lv4_user_prompt = lv4_user_prompt_template % (user_input_text, outline_xml_str)
    
    start_time = time.time()
    response = run_claude(lv4_system_prompt, lv4_user_prompt)
    lv4_output = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    execution_time = time.time() - start_time
    
    web_search_result = ""
    search_query = json.loads(lv4_output.split('<json>')[1])
    
    try:
        for i in range(len(search_query)):
            web_search_result += f"{google_websearch(search_query[i], 3)}\n\n"
        web_search_result = web_search_result.rstrip('\n')
    except Exception as e:
        print(f'google search error: {e}')
        web_search_result = 'No Search Result'
        
    print(f'✅ lv4(websearch) done. time: {execution_time}')
    
    return {
        "user_input_text": user_input_text,
        "lv4_content_background_search": {
            "system_prompt": lv4_system_prompt,
            "user_prompt": lv4_user_prompt,
            "llm_output": lv4_output,
            "input_tokens": input_tokens, 
            "output_tokens": output_tokens,
            "time": execution_time
            
        },
        "lv4_websearch_result": {
            "search_query": search_query,
            "websearch_output": web_search_result,
        }
    }
    
    
def process_chunk(args):
    i, lv5_system_prompt, lv5_user_prompt_template, item_prompt_dict, user_input_text, slide_count, audience, tone, extra_note, lv2_outline_llm_output, layout_split, lv4_websearch_result_websearch_output, chunk_size = args
    
    outline_xml_str = make_outline_str(lv2_outline_llm_output)
    outline_layout_set = make_outline_layout_template_set(lv2_outline_llm_output, layout_split[i], item_prompt_dict)
    lv4_websearch_result_websearch_output = "No Search Result" if "No Search Result" in lv4_websearch_result_websearch_output else lv4_websearch_result_websearch_output
    lv5_user_formatted = lv5_user_prompt_template % (slide_count, audience, tone, outline_xml_str, extra_note, user_input_text, outline_layout_set, lv4_websearch_result_websearch_output)
    
    start_time = time.time()
    response = run_claude(lv5_system_prompt, lv5_user_formatted)
    lv5_output = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    execution_time = time.time() - start_time
    
    print(f'... chunk {i} done. time: {execution_time}')
    
    return i, {
        "system_prompt": lv5_system_prompt,
        "user_prompt": lv5_user_formatted,
        "llm_output": lv5_output,
        "input_tokens": input_tokens, 
        "output_tokens": output_tokens,
        "time": execution_time
    }
    

def lv5_content(lv5_system_prompt, lv5_user_prompt_template, item_prompt_path,
                user_input_text, slide_count, audience, tone, extra_note, 
                lv2_outline_llm_output, lv3_layout_llm_output, lv4_websearch_result_websearch_output, chunk_size=6):
    
    item_prompt_dict = extract_itemprompts_text_from_zip(item_prompt_path)
    layout_dict = json.loads(lv3_layout_llm_output.split("<json>")[1])
    layout_split = split_layout_dict_five(layout_dict, chunk_size)
    
    total_output = {
        "user_inputs": {
            "user_input_text": user_input_text,
            "slide_count": slide_count,
            "audience": audience,
            "tone": tone,
            "extra_note": extra_note,
            "template_idx": item_prompt_dict['template_idx'],
        },
        "lv5_content": {}
    }

    # 병렬 처리를 위한 인자 리스트 생성
    args_list = [(i, lv5_system_prompt, lv5_user_prompt_template, item_prompt_dict,
                  user_input_text, slide_count, audience, tone, extra_note, 
                  lv2_outline_llm_output, layout_split, lv4_websearch_result_websearch_output, chunk_size) 
                 for i in range(len(layout_split))]
    
    # ThreadPoolExecutor를 사용한 병렬 처리
    with ThreadPoolExecutor(max_workers=len(layout_split)) as executor:
        results = list(executor.map(process_chunk, args_list))
    
    execution_time = 0
    # 결과를 total_output에 저장
    for i, result in results:
        total_output['lv5_content'][f"lv5_content_{i}"] = result
        current_time = result['time']  # 현재 반복에서의 time 값
        if current_time > execution_time:
            execution_time = current_time
    
    try:
        total_output['lv5_content']['lv5_content_merge'] = merge_lv5_outputs(layout_split, total_output['lv5_content'])
        total_output['lv5_content']['execution_time'] = execution_time
    except Exception as e:
        
        # try:
        #     for i in range(len(layout_split)):
        #         try_fix_and_parse_json(total_output['lv5_content'][f"lv5_content_{i}"]["llm_output"].split('<json>')[1])
        # except Exception as e:
        # print(f'merge error: {e}')
            total_output['lv5_content']['lv5_content_merge'] = "merge error"
            total_output['lv5_content']['execution_time'] = execution_time
        
    print(f'✅ lv5(content) done. time: {execution_time}')
    
    return total_output 