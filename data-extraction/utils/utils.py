import zipfile
import os
import json
import chardet


def save_dict_to_json(data: dict, filename: str, indent: int = 4):
    """
    딕셔너리를 JSON 파일로 저장하는 함수.
    
    :param data: 저장할 딕셔너리
    :param filename: 저장할 파일 이름 (예: "data.json")
    :param indent: JSON 파일의 들여쓰기 (기본값: 4)
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        print(f"JSON 파일 저장 완료: {filename}")
    except Exception as e:
        print(f"오류 발생: {e}")


def load_json_to_dict(filename: str) -> dict:
    """
    JSON 파일을 읽어와서 딕셔너리로 변환하는 함수.

    :param filename: 불러올 JSON 파일 이름 (예: "data.json")
    :return: JSON 데이터를 담은 딕셔너리
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"JSON 파일 로드 완료: {filename}")
        return data
    except Exception as e:
        print(f"오류 발생: {e}")
        return {}


def is_valid_json(json_string: str) -> bool:
    """
    입력된 문자열이 올바른 JSON 형식인지 확인하는 함수.
    
    :param json_string: JSON 형식으로 확인할 문자열
    :return: 올바른 JSON 형식이면 True, 아니면 False
    """
    try:
        json.loads(json_string)
        return True
    except json.JSONDecodeError:
        print(f'🚫 json 파싱 에러')
        return False


def zip_txt_files(source_folder: str, destination_folder: str, template_idx: int):
    """
    지정된 폴더 안의 txt 파일들만 압축하여, 지정된 형식의 zip 파일로 저장
    
    :param source_folder: txt 파일들이 있는 원본 폴더 경로
    :param destination_folder: 압축된 파일을 저장할 폴더 경로
    :param template_idx: 압축 파일명에 사용할 인덱스 값
    """
    if not os.path.isdir(source_folder):
        raise ValueError("원본 폴더가 존재하지 않습니다.")
    
    if not os.path.isdir(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)
    
    zip_filename = f"layout{template_idx}-prompt-test.zip"
    zip_path = os.path.join(destination_folder, zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_folder):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, source_folder))
    
    print(f"압축 완료: {zip_path}")


def extract_and_rezip_txt(zip_file_path: str, destination_folder: str):
    """
    1. zip 파일을 압축 해제
    2. txt 파일들만 다시 압축 (폴더 구조 없이 파일만 포함)
    3. 기존 zip 파일명을 유지하여 사용자가 지정한 폴더에 저장
    
    :param zip_file_path: 기존 zip 파일 경로
    :param destination_folder: 새로운 zip 파일을 저장할 폴더 경로
    """
    if not os.path.isfile(zip_file_path):
        raise ValueError("지정된 zip 파일이 존재하지 않습니다.")
    
    if not os.path.isdir(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)
    
    extract_folder = os.path.join(destination_folder, "extracted")
    os.makedirs(extract_folder, exist_ok=True)
    
    # Zip 파일명 추출
    zip_filename = os.path.basename(zip_file_path)
    zip_name, _ = os.path.splitext(zip_filename)
    new_zip_path = os.path.join(destination_folder, f"{zip_name}.zip")
    
    # Zip 파일 압축 해제
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    
    # txt 파일만 다시 압축 (폴더 구조 없이 파일만 포함)
    with zipfile.ZipFile(new_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(extract_folder):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.basename(file))
    
    # 임시 폴더 삭제
    for root, dirs, files in os.walk(extract_folder, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    os.rmdir(extract_folder)
    
    print(f"새로운 압축 파일 생성 완료: {new_zip_path}")


def read_text_file_auto_encoding(file_path):
    """파일의 인코딩을 자동 감지하여 문자열로 읽어오는 함수"""
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read()

        detected = chardet.detect(raw_data)
        encoding = detected.get("encoding", "utf-8")

        with open(file_path, "r", encoding=encoding) as file:
            return file.read()
    except Exception as e:
        print(f"파일 읽기 중 오류 발생: {e}")
    return None 


def try_fix_and_parse_json(data_str):
    """
    문자열 타입으로 된 json 데이터를 파싱하는 함수
    
    Args:
        data_str (str): JSON 형태의 문자열
    
    Returns:
        dict or str: 파싱된 JSON 객체 또는 "merge_error"
    """
    # 1. json.loads(data_str)로 1차 시도
    try:
        json.loads(data_str)
        return data_str
    except json.JSONDecodeError:
        pass
    
    try:
        data_str_2 = data_str + '}'
        json.loads(data_str_2)
        return data_str_2
    except json.JSONDecodeError:
        pass

    return "merge_error"