import zipfile
import os
import re
import json
from typing import Dict, Any
import pandas as pd


def extract_itemprompts_text_from_zip(zip_path: str) -> Dict[str, Any]:
    """
    아이템 프롬프트 ZIP(.zip)을 읽어 다음 정보를 반환한다.
    
    - template_idx        : ZIP 경로에서 layout 숫자만 추출
    - template_info_all   : 모든 TXT 파일 전문을 순서대로 이어 붙인 문자열
    - template_info       : {파일번호(int): TXT 전문(str)} 딕셔너리
    - template_info_purpose : 각 파일에서 Purpose 블록만 추출해 모아 둔 문자열 -> lv3_layout 단계에서 사용된다.
    
    Purpose 블록은
    `<Purpose of the page>` … `<Purpose, number of text boxes, and text length of the content>`
    사이의 텍스트이며, 없으면 "Pattern not found" 로 표시된다.
    """
    extracted_texts: Dict[int, str] = {}
    merged_texts: list[str] = []
    purpose_lines: list[str] = []

    # 1) ZIP 열어서 내용 읽기 ---------------------------------------------------
    with zipfile.ZipFile(zip_path, "r") as zf:
        txt_files = sorted(
            [f for f in zf.namelist() if f.lower().endswith(".txt")]
        )

        for order, filename in enumerate(txt_files, start=1):
            with zf.open(filename) as f:
                key_int = int(os.path.splitext(os.path.basename(filename))[0])
                text = f.read().decode("utf-8")

                extracted_texts[key_int] = text
                merged_texts.append(f"[{order}]\n\"{text}\"")

                # Purpose 추출
                m = re.search(
                    r"<Purpose of the page>(.*?)<Purpose, number of text boxes, and text length of the content>",
                    text,
                    flags=re.DOTALL,
                )
                purpose = m.group(1).strip() if m else "Pattern not found"
                purpose_lines.append(f"[{key_int}]: {purpose}")

    # 2) layout 번호 추출 ------------------------------------------------------
    m_layout = re.search(r"layout(\d+)", zip_path)
    template_idx = m_layout.group(1) if m_layout else "unknown"

    # 3) 결과 반환 -------------------------------------------------------------
    return {
        "template_idx": template_idx,
        "template_info_all": "\n".join(merged_texts),
        "template_info": extracted_texts,
        "template_info_purpose": "\n".join(purpose_lines),
    }


def extract_formatprompts_from_folder(base_path):
    """
    지정된 폴더에서 format prompt 파일들을 읽어와 딕셔너리로 반환하는 함수
    
    각 하위 폴더에서 system-prompt.txt와 user-prompt.txt 파일을 찾아 읽어온다.
    파일이 존재하지 않으면 빈 문자열로 처리한다.
    
    Args:
        base_path (str): format prompt 폴더들이 있는 기본 경로
        
    Returns:
        dict: {폴더명: {"system-prompt": 내용, "user-prompt": 내용}} 형태의 딕셔너리
              폴더명은 정렬된 순서로 처리됨
              
    Example:
        >>> extract_formatprompts_from_folder("/path/to/prompts")
        {
            "folder1": {
                "system-prompt": "시스템 프롬프트 내용...",
                "user-prompt": "유저 프롬프트 내용..."
            },
            "folder2": {
                "system-prompt": "...",
                "user-prompt": "..."
            }
        }
    """
    
    prompt_data = {}
    
    for folder_name in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder_name)
        
        if os.path.isdir(folder_path):  # 폴더인지 확인
            system_prompt_path = os.path.join(folder_path, "system-prompt.txt")
            user_prompt_path = os.path.join(folder_path, "user-prompt.txt")
            
            prompt_data[folder_name] = {
                "system-prompt": open(system_prompt_path, encoding='utf-8').read() if os.path.exists(system_prompt_path) else "",
                "user-prompt": open(user_prompt_path, encoding='utf-8').read() if os.path.exists(user_prompt_path) else ""
            }
    
    return prompt_data


def make_template_keyword_csv_to_list(template_keyword_csv_path: str):
    """
    템플릿 키워드 리스트 생성
    """
    template_keyword_list = []
    
    df = pd.read_csv(template_keyword_csv_path)

    for _, row in df.iterrows():
        template_keyword_list.append({
            "template_idx": str(row['전용템플릿idx']),
            "keywords": row['원본 키워드']
        })
        
    return template_keyword_list


def make_outline_str(data: str):
    """
    JSON 형태의 outline 데이터를 XML 태그 형식의 문자열로 변환하는 함수
    
    입력된 JSON 문자열에서 outline 정보를 추출하여 각 페이지별로
    XML 태그 형식(<{page}page>내용</{page}page>)으로 포맷팅한다.
    
    Args:
        data (str): '<json>' 태그를 포함한 JSON 문자열
                   outline 키 하위에 페이지별 데이터가 있어야 함
                   각 페이지 데이터는 문자열이거나 {'title': str, 'contentList': list} 형태
    
    Returns:
        str: XML 태그 형식으로 포맷팅된 outline 문자열
             각 페이지는 <{페이지번호}page>내용</{페이지번호}page> 형태로 변환됨
             페이지 번호 순서대로 정렬되어 출력됨
    
    Example:
        >>> data = '<json>{"outline": {"1": "소개", "2": {"title": "본문", "contentList": ["내용1", "내용2"]}}}'
        >>> make_outline_str(data)
        '  <1page>소개</1page>\\n  <2page>본문: 내용1,내용2</2page>'
    """
    outputs = ''
    data = json.loads(data.split('<json>')[1])
    data = data['outline']
    
    keys = sorted(data.keys(), key=lambda x: int(x))
    
    for key in keys:
        if isinstance(data[key], str):
            outputs += f"  <{key}page>{data[key]}</{key}page>\n"
        else:
            title = data[key]['title']
            contents = ",".join(data[key]['contentList'])
            outputs += f"  <{key}page>{title}: {contents}</{key}page>\n"

    outputs = outputs.rstrip('\n ')

    return outputs


def split_layout_dict_five(input_dict, chunk_size=5):
    """
    input_data = {"1":1,"2":4,"3":6,"4":5,"5":10,"6":11,"7":9,"8":8,"9":7,"10":13,"11":12,"12":14,"13":15}
    output:
        {0: {'1': 1, '2': 4, '3': 6, '4': 5, '5': 10},
        1: {'6': 11, '7': 9, '8': 8, '9': 7, '10': 13},
        2: {'11': 12, '12': 14, '13': 15}}
    """
    keys = list(input_dict.keys())  # 키 리스트 추출
    chunked_dict = {}
    
    for i in range(0, len(keys), chunk_size):
        chunked_dict[i // chunk_size] = {k: input_dict[k] for k in keys[i:i + chunk_size]}
    
    return chunked_dict


def make_outline_layout_template_set(outline, layout: dict, template_info: dict):
    """
    outline, layout, template_info를 결합하여 파이프라인에서 사용할 형태로 변환
    """
    outline = json.loads(outline.split("<json>")[1])
    
    result = ''
    
    for page in layout:
        outline_page = outline['outline'][page]
        
        if type(outline_page) == str:
            result += f"<{page}page>{outline_page}</{page}page>"
        else:
            title = outline_page['title']
            contentList = ",".join(outline_page['contentList'])
            result += f"<{page}page>{title}: {contentList}</{page}page>\n"
            
        result += f"{template_info['template_info'][layout[page]]}\n\n"
        
    return result.rstrip('\n')


def merge_lv5_outputs(layout_split: dict, lv5_results: dict):
    """
    lv5 파이프라인의 청크별 결과를 병합하여 최종 결과 생성
    """
    merge_content_result = {}
    
    for chunk_num in layout_split:
        lv5_content_output_json = json.loads(lv5_results[f'lv5_content_{chunk_num}']['llm_output'].split("<json>")[1])
        for page in lv5_content_output_json:
            merge_content_result[page] = lv5_content_output_json[page]
        
    return json.dumps(merge_content_result, ensure_ascii=False)


def calculate_cost_usd(cache_write: int, cache_read: int, input_tokens: int, output_tokens: int):
    """
    입력 및 출력 토큰 수를 기반으로 비용을 계산하는 함수
    :param input_tokens: 입력 토큰 수
    :param output_tokens: 출력 토큰 수
    :param input_cost_per_mtok: 입력 토큰당 비용 (1M 토큰 기준, 기본값: $3.00)
    :param output_cost_per_mtok: 출력 토큰당 비용 (1M 토큰 기준, 기본값: $15.00)
    :return: 총 비용 (달러 단위)
    """
    input_cost_per_mtok = 3.00
    output_cost_per_mtok = 15.00
    cache_write_cost_per_mtok = 3.75
    cache_read_cost_per_mtok = 0.3
    
    cache_write_cost = (cache_write / 1_000_000) * cache_write_cost_per_mtok
    cache_read_cost = (cache_read / 1_000_000) * cache_read_cost_per_mtok
    input_cost = (input_tokens / 1_000_000) * input_cost_per_mtok
    output_cost = (output_tokens / 1_000_000) * output_cost_per_mtok
    total_cost = cache_write_cost + cache_read_cost + input_cost + output_cost
    
    total_cost_dict = {
        "cache_write_cost": cache_write_cost,
        "cache_read_cost": cache_read_cost,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }
    
    return total_cost_dict 